import os
import torch
from d2l import torch as d2l

def read_data():
    data_dir = r'D:\data\cmn-eng'

    text = []
    with open(os.path.join(data_dir, 'cmn-eng.txt'), 'r',
              encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split('\t')

        if len(parts) >= 2:
            text.append(f'{parts[0]}\t{parts[1]}')            
     
    return '\n'.join(text)

text = read_data()

def preprocess(text):
    def is_chinese(char):
        return '\u4e00' <= char <= '\u9fa5'
    
    def no_space(char, prev_char):
        return char in set(',.!?，。！？') and prev_char != ' '
    
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()

    text_with_spaces = []
    for char in text:
        if is_chinese(char):
            text_with_spaces.append(' ' + char + ' ')
        else:
            text_with_spaces.append(char)
    text = ''.join(text_with_spaces)

    out = [' ' + char 
           if i > 0 and no_space(char, text[i - 1]) 
           else char
           for i, char in enumerate(text)]

    return ''.join(out)

text = preprocess(read_data())
# print(text[:90])

def tokenize(text, num_examples=None):
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break

        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target

source, target = tokenize(text)
# print(len(source), len(target))
# print(source[-1:])
# print(target[-1:])


def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
    d2l.set_figsize()
    _, _, patches = d2l.plt.hist(
        [[len(l) for l in xlist], [len(l) for l in ylist]])
    d2l.plt.xlabel(xlabel)
    d2l.plt.ylabel(ylabel)
    for patch in patches[1].patches:
        patch.set_hatch('/')
    d2l.plt.legend(legend)

show_list_len_pair_hist(['source', 'target'], '# tokens per sequence',
                        'count', source, target);
# d2l.plt.show()

src_vocab = d2l.Vocab(source, min_freq=2,
                      reserved_tokens=['<pad>', '<bos>', '<eos>'])
# print(len(src_vocab))

def truncate_pad(line, num_steps, padding_token):
    if len(line) > num_steps:
        return line[:num_steps]
    return line + [padding_token] * (num_steps - len(line))

# print(truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>']))

def build_array(lines, vocab, num_steps):
    lines_ids = [[vocab[token] for token in line] for line in lines]

    eos_id = vocab['<eos>']
    lines_ids = [ids + [eos_id] for ids in lines_ids]

    array = torch.tensor([truncate_pad(ids, num_steps, vocab['<pad>']) for ids in lines_ids],
                         dtype=torch.long)
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)

    return array, valid_len

def load_data(batch_size, num_steps, num_examples=None):
    text = preprocess(read_data())
    source, target = tokenize(text)
    source = [[token for token in line if token.strip() != ''] for line in source]
    target = [[token for token in line if token.strip() != ''] for line in target]

    src_vocab = d2l.Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tar_vocab = d2l.Vocab(target, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])

    src_array, src_valid_len = build_array(source, src_vocab, num_steps)
    tar_array, tar_valid_len = build_array(target, tar_vocab, num_steps)

    data_arrays = (src_array, src_valid_len, tar_array, tar_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)

    return data_iter, src_vocab, tar_vocab
