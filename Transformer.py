import math
import copy
import time
import torch
import random
import Attention
import collections
import pandas as pd
from torch import nn
from d2l import torch as d2l
import sec_machine_translation

class PositionEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, 
                 num_heads, dropout, bias=False, **kwargs) -> None:
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = Attention.DotproductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)
        
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0
            )

        output = self.attention(queries, keys, values, valid_lens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)

class FFN(nn.Module):
    def __init__(self, ffn_input, ffn_hiddens, ffn_outputs, **kwargs):
        super(FFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_input, ffn_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_hiddens, ffn_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

class EncoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_input, ffn_hiddens, num_heads, dropout,
                 use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(key_size, query_size, value_size, 
                                            num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = FFN(ffn_input, ffn_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)
    
    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))

class TransformerEncoder(d2l.Encoder):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_input, ffn_hiddens, num_heads,
                 num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block" + str(i),
                                 EncoderBlock(key_size, query_size, value_size,
                                              num_hiddens, norm_shape, ffn_input, ffn_hiddens,
                                              num_heads, dropout, use_bias))
    
    def forward(self, X, valid_lens, *args):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
        return X

class DecoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_input, ffn_hiddens, num_heads,
                 dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = FFN(ffn_input, ffn_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)
    
    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]

        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), axis=1) # type: ignore
        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            dec_valid_lens = torch.arange(
                1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None

        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state

class TransformerDecoder(d2l.AttentionDecoder):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                DecoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, i))
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]
    
    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
        return self.dense(X), state

def transpose_qkv(X, num_heads):
    # shape of X: batch_size, key/query_size, num_hiddens
    # trans X ->: batch_size, key/query_size, num_heads, num_hiddens/num_heads
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    # trans X ->: batch_size, num_heads, k/q_size, num_hiddens/num_heads
    X = X.permute(0, 2, 1, 3)
    # trnas X ->: (batch_size * num_heads, k/q_size, num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])

def transpose_output(X, num_heads):
    # trans X ->: batch_size, num_heads, 
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)

'''
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

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

    net.apply(init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = MaskedSoftmaxLoss()  
    net.train()

    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2) 
        time_st = time.time()

        for batch_idx, batch in enumerate(data_iter):
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]

            bos = torch.tensor([tar_vocab['<bos>']] * Y.shape[0], 
                               device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], dim=1)

            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss_fn(Y_hat, Y, Y_valid_len) 
            l.sum().backward()
            d2l.grad_clipping(net, 1)
            optimizer.step()

            num_tokens = int(Y_valid_len.sum().item())
            with torch.no_grad():
                metric.add(l.sum().item(), num_tokens)


        cost = time.time() - time_st
        print(f'epoch [{epoch + 1}/{num_epochs}]: '
              f'loss {metric[0] / metric[1]:.3f}, '
              f'num_tokens {metric[1]:.2f}, '
              f'cost {cost:.3f} sec')

        if (epoch + 1) % 5 == 0:
            predict()
            net.train()

    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec on {str(device)}')
'''
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

num_hiddens, num_layers, dropout, batch_size, num_steps = 64, 3, 0.1, 128, 25
lr, num_epochs, device = 0.001, 80, d2l.try_gpu()
ffn_num_input, ffn_num_hiddens, num_heads = 64, 128, 4
key_size, query_size, value_size = 64, 64, 64
norm_shape = [64]

train_iter, src_vocab, tar_vocab = sec_machine_translation.load_data(batch_size, num_steps)
encoder = TransformerEncoder(
    len(src_vocab), key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout)
decoder = TransformerDecoder(
    len(tar_vocab), key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout)
net = d2l.EncoderDecoder(encoder, decoder)


'''
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
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
    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .', 'pass the post-graduate school entrance exam !', 
            'pass the post-graduate entrance exam !', 'i want to buy a car .', 'how are you doing today ?']
    cmns = ['快 跑 。', '我 迷 路 了 。', '他 很 冷 静 。', '我 到 家 了 。', 
            '通 过 考 研 !', '通 过 研 究 生 考 试 !', '我 想 买 辆 车 。', '你 今 天 怎 么 样？']
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
'''
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''

print("saving model...")
torch.save(net.state_dict(), 'Transformer.pth')
print("save successfully")