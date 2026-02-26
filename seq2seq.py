import math
import copy
import time
import torch
import random
import Attention
import collections
from torch import nn
import torch.nn.functional as F
import sec_machine_translation
from d2l import torch as d2l

class SeqEncoder(d2l.Encoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, 
                 dropout=0.0, **kwargs):
        super(SeqEncoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          dropout=dropout)
        
    
    def forward(self, X, *args):
        X = self.embedding(X)
        X = X.permute(1, 0, 2)
        output, state = self.rnn(X)

        return output, state
    
encoder = SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,
                     num_layers=2)
encoder.eval()
X = torch.zeros((4, 7), dtype=torch.long) # batch_size = 4, steps = 7
output, state = encoder(X)
# print(output.shape, state.shape) # steps/layers, batch, hiddens. 

class SeqDecoder(d2l.Decoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0.0, **kwargs):
        super(SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,
                          dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]
    
    def forward(self, X, state):
        X = self.embedding(X).permute(1, 0, 2)
        context = state[-1].repeat(X.shape[0], 1, 1)
        X_and_context = torch.cat((X, context), 2)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).permute(1, 0, 2)

        return output, state

class AttentionDecoder(d2l.Decoder):
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    def attention_weights(self):
        raise NotImplementedError

class AttentionSeqDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(AttentionSeqDecoder, self).__init__(**kwargs)
        self.attention = Attention.AdditiveAttention(num_hiddens, num_hiddens, 
                                                     num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens,
                          num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_len, *args):
        outputs, hidden_state = enc_outputs
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_len)
    
    def forward(self, X, state):
        '''
        outputs: batch_size, num_steps, num_hiddens
        hidden_state: num_layers, batch_size, num_hiddens
        X: num_steps, batch_size, embed_size
        x: batch_size, embed_size
        query: batch_size, 1, num_hiddens
        context: batch_size, 1, num_hiddens
        '''
        enc_outputs, hidden_state, enc_valid_len= state
        X = self.embedding(X).permute(1, 0, 2)
        outputs, self._attention_weights = [], []

        for x in X:
            
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            context = self.attention(query, enc_outputs, enc_outputs, enc_valid_len)
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state,
                                          enc_valid_len]

    def attention_weights(self):
        return self._attention_weights
    

# decoder = SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
#                      num_layers=2)
decoder = AttentionSeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
                     num_layers=2)
decoder.eval()
state = decoder.init_state(encoder(X), None)
output, state = decoder(X, state)
# print(output.shape, state.shape) #batch, steps, vocab/hiddens

def sequence_mask(X, valid_len, value=0): # Tag
    maxlen = X.size(1) #Tag
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

X = torch.tensor([[1, 2, 3], [4, 5, 6]])
# print(sequence_mask(X, torch.tensor([1, 2])))

class MaskedSoftmaxLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label, dtype=torch.float32, device=pred.device)
        weights = sequence_mask(weights, valid_len)  # mask: 1 for valid tokens
        loss_all = F.cross_entropy(pred.permute(0,2,1), label, reduction='none')  # (batch, seq_len)
        loss_masked = (loss_all * weights).sum(dim=1) / (weights.sum(dim=1) + 1e-8)  # avg per sequence
        return loss_masked  # (batch,)

loss = MaskedSoftmaxLoss()

import random
import torch
import torch.nn.functional as F
from torch import nn

def train(net: nn.Module, data_iter, lr, num_epochs, tar_vocab, device):
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0, std=0.01)
        if isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)

    # scheduled sampling schedule
    teacher_forcing_start = 1.0
    teacher_forcing_end = 0.0
    total_epochs = max(1, num_epochs)

    net.apply(init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = MaskedSoftmaxLoss()  # 假定你已按建议实现了 MaskedSoftmaxLoss
    net.train()

    eos_id = tar_vocab['<eos>']

    def avg_eos_prob_last_valid(y_hat, y_valid_len, eos_id):
        with torch.no_grad():
            probs = F.softmax(y_hat, dim=-1)
            batch = y_hat.size(0)
            vals = []
            for i in range(batch):
                last = int(y_valid_len[i].item()) - 1
                if last < 0:
                    vals.append(0.0)
                else:
                    vals.append(probs[i, last, eos_id].item())
            return sum(vals) / len(vals)

    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # loss_sum, token_num
        time_st = time.time()

        tf_ratio = teacher_forcing_start + (teacher_forcing_end - teacher_forcing_start) * (epoch / (total_epochs - 1))

        for batch_idx, batch in enumerate(data_iter):
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]

            # ---- encoder ----
            enc_outputs = net.encoder(X, X_valid_len)  # 通常返回 (output, state)
            dec_state = net.decoder.init_state(enc_outputs, X_valid_len)

            # stepwise decoding with scheduled sampling
            batch_size, seq_len = Y.shape[0], Y.shape[1]
            dec_input_t = torch.full((batch_size, 1), tar_vocab['<bos>'], dtype=torch.long, device=device)  # (batch,1)
            outputs = []  # will store (batch,1,vocab) per step

            for t in range(seq_len):
                # decoder expects (batch, seq) -> here seq=1
                Y_hat_t, dec_state = net.decoder(dec_input_t, dec_state)
                outputs.append(Y_hat_t)  # (batch,1,vocab)
                use_teacher = random.random() < tf_ratio
                if use_teacher:
                    # next input is gold token at time t
                    dec_input_t = Y[:, t].unsqueeze(1)
                else:
                    # next input is model's prediction (detach to avoid backprop through sampling op)
                    dec_input_t = Y_hat_t.argmax(dim=2).detach()

            # concat outputs to (batch, seq_len, vocab)
            Y_hat = torch.cat(outputs, dim=1)

            # compute masked loss (loss_fn returns per-seq loss)
            l = loss_fn(Y_hat, Y, Y_valid_len)  # (batch,)
            l.sum().backward()
            d2l.grad_clipping(net, 1)
            optimizer.step()

            num_tokens = int(Y_valid_len.sum().item())
            with torch.no_grad():
                metric.add(l.sum().item(), num_tokens)


        # 每 epoch 打印 summary（你原来的打印保留）
        cost = time.time() - time_st
        print(f'epoch [{epoch + 1}/{num_epochs}]: '
              f'loss {metric[0] / metric[1]:.3f}, '
              f'num_tokens {metric[1]:.2f}, '
              f'cost {cost:.3f} sec')

        # 可选：每若干 epoch 做 validation / sample predict
        if (epoch + 1) % 10 == 0:
            predict()
            net.train()

    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec on {str(device)}')

    
embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 64, 10
lr, num_epochs, device = 0.005, 300, d2l.try_gpu()

train_iter, src_vocab, tar_vocab = sec_machine_translation.load_data(batch_size, num_steps)
encoder = SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers,
                        dropout)
decoder = SeqDecoder(len(tar_vocab), embed_size, num_hiddens, num_layers,
                        dropout)
net = d2l.EncoderDecoder(encoder, decoder)

def _clone_state(state):
    """Clone decoder state safely (Tensor or tuple/list of Tensors)."""
    if state is None:
        return None
    if isinstance(state, torch.Tensor):
        return state.clone()
    if isinstance(state, (tuple, list)):
        return type(state)(_clone_state(s) for s in state)
    # fallback: deep copy
    return copy.deepcopy(state)

def beam_search_predict(net, enc_outputs, dec_state, tgt_vocab,
                        beam_size=4, max_steps=20, device='cpu',
                        alpha=0.7, enc_valid_len=None):
    """
    Returns best token id list found by beam search.
    - net: model with .decoder
    - enc_outputs: whatever encoder returns (passed to decoder if needed)
    - dec_state: initial decoder state
    - enc_valid_len: optional tensor for masking (pass to decoder if needed)
    """
    # beam element: (seq_tokens_list, state, score)
    beam = [([tgt_vocab['<bos>']], _clone_state(dec_state), 0.0)]
    completed = []

    for _ in range(max_steps):
        new_beam = []
        for seq, state, score in beam:
            last_tok = seq[-1]
            dec_X = torch.tensor([[last_tok]], device=device, dtype=torch.long)  # (1,1)

            # call decoder; try both common signatures
            try:
                Y_hat, new_state = net.decoder(dec_X, state, enc_outputs, enc_valid_len)
            except TypeError:
                Y_hat, new_state = net.decoder(dec_X, state)

            # Y_hat: (batch=1, seq=1, vocab)
            logp = torch.log_softmax(Y_hat[:, -1, :], dim=-1).squeeze(0)  # (vocab,)

            topk_logp, topk_idx = torch.topk(logp, beam_size)
            for k in range(topk_idx.size(0)):
                tok = int(topk_idx[k].item())
                new_seq = seq + [tok]

                # length penalty (so longer reasonable sequences not overly penalized)
                length_penalty = ((5 + len(new_seq)) / 6) ** alpha
                new_score = (score * length_penalty + float(topk_logp[k].item())) / length_penalty

                # clone state for safe branching
                state_clone = _clone_state(new_state)

                if tok == tgt_vocab['<eos>']:
                    completed.append((new_seq, new_score, state_clone))
                else:
                    new_beam.append((new_seq, state_clone, new_score))

        # keep top beam_size
        beam = sorted(new_beam, key=lambda x: x[2], reverse=True)[:beam_size]
        if not beam:
            break

    # choose best completed (by score) or best partial
    if completed:
        best_seq, best_score, best_state = max(completed, key=lambda x: x[1])
    else:
        best_seq, best_score, best_state = beam[0]

    return best_seq, best_state  # return token id list and final state (if needed)


def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    net.eval()
    src_sentence = sec_machine_translation.preprocess(src_sentence)
    
    tokens = [t for t in src_sentence.lower().split(' ') if t]
    src_tokens = [src_vocab[t] for t in tokens] + [src_vocab['<eos>']]
    
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    dec_X = torch.unsqueeze(torch.tensor(
        [tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        if pred == tgt_vocab['<eos>'] or pred == tgt_vocab['<pad>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq

def bleu(pred_seq, label_seq, k):  #@save
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score

def predict():
    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .', 'pass the graduate school entrance exam !', 
            'pass the postgraduate entrance exam !', 'I scored 400 points on the postgraduate entrance exam .']
    cmns = ['快 跑 。', '我 迷 路 了 。', '他 很 冷 静 。', '我 到 家 了 。', 
            '通 过 考 研 !', '通 过 研 究 生 考 试 !', '我 研 究 生 考 试 得 了 400 分 。']
    for eng, cmn in zip(engs, cmns):
        translation, attention_weight_seq = predict_seq2seq(
            net, eng, src_vocab, tar_vocab, num_steps, device)
        
        readable_translation = translation.replace(' ', '')
        score = bleu(translation, cmn, k=2)
        print(f'英文: {eng}')
        print(f'标签: {cmn}')
        print(f'预测: {readable_translation}')
        print(f'BLEU: {score:.3f}')
        print('-' * 20)

train(net, train_iter, lr, num_epochs, tar_vocab, device)

print("saving model...")
torch.save(net.state_dict(), 'seq2seq.pth')
print("save successfully")



# engs = 'pass the graduate school entrance exam'
# translation, attention_weight_seq = predict_seq2seq(
#         net, engs, src_vocab, tar_vocab, num_steps, device)
# readable_translation = translation.replace(' ', '')
# print(f'{engs} => {translation}')
# print(readable_translation)