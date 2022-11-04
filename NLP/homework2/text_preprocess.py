import collections
import re
import numpy as np
import time
from zhon.hanzi import punctuation as cn_punc
from string import punctuation as en_punc
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch


parser = argparse.ArgumentParser("data preprocess")
parser.add_argument('--padding', type=bool, default=False, help="whether do padding for sentence")
args = parser.parse_args()
PADDING = args.padding


def load_book_tokenize():
    def filter_chinese(x):
        punc = cn_punc + en_punc
        tmp = re.sub("[{}]+".format(punc), "", x) # remove punctutation
        remove_white_spaces = re.sub(r'[\r|\n|\t]', '', tmp) # remove white space
        # pure_words = re.sub(r'[0-9]+', '', remove_white_spaces) # remove numbers
        pure_words = re.sub(r'[^\u4e00-\u9fa5_^0-9]', '', remove_white_spaces) # filter out other characters except chinese
        
        return pure_words
    
    with open('./data.txt', 'r') as f:
        text = f.readlines()
    lines = list(map(filter_chinese, text))
            
    return lines


def check_line_length(lines):
    length = [len(line) for line in lines]
    plt.hist(length, bins = len(set(length)))
    plt.grid()
    plt.xlabel("length")
    plt.ylabel("number")
    plt.show()


# if need to padd, filter the lines in order to avoid large padding.
def filter_lines(lines):
    length = [len(line) for line in lines]
    freq = collections.Counter(length)
    freq = dict(sorted(freq.items(), key=lambda x: x[1], reverse=True))
    res = []
    for line in lines:
        line_length = len(line)
        if freq[line_length] >= 10 and line_length >= 10  and line_length < 400:
            res.append(line)
            
    return res


def padding(lines):
    lines = filter_lines(lines)
    length = [len(line) for line in lines]
    max_length = max(length) + 1 # last one is end of sentence 'E'
    res = []
    for line in lines:
        padding_size = max_length - 1 - len(line)
        line += 'E' + 'P' * padding_size# padding token is 'P'
        res.append(line)
    
    return res
    

def count_corpus(lines):
    flattened_tokens = [c for line in lines for c in line]
    
    return collections.Counter(flattened_tokens)


class Vocab:
    def __init__(self, min_freq=10, reserved_tokens=['U'], tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
            
        freq = count_corpus(tokens)
        self.freq_tokens = sorted(freq.items(), key=lambda x: x[1], reverse=True)

        res = ['P'] + reserved_tokens # padding index will be 0
        self.unk = res.index('U')
        res += [token for token, freq in self.freq_tokens if freq > min_freq and token not in res]

        self.index_to_token, self.token_to_index = [], {}
        for token in res:
            self.index_to_token.append(token)
            self.token_to_index[token] = len(self.index_to_token) - 1

    def __len__(self):
        return len(self.index_to_token)
    
    def __getitem__(self, token):
        if not isinstance(token, (list, tuple)):
            return self.token_to_index.get(token, self.unk) # if not exist in vocab return 'U'(unknow)
        return [self.__getitem__(item) for item in token]

    def to_tokens(self, index):
        if not isinstance(index, (list, tuple)):
            return self.index_to_token[index]
        return [self.to_tokens(item) for item in index]
    

def divide_corpus(train_size=0.7, val_size=0.2):
    lines = load_book_tokenize()
    all_tokens = [c for line in lines for c in line]
    vocab = Vocab(tokens=all_tokens, reserved_tokens=['E', 'U'])
    if PADDING:
        lines = padding(lines)
        corpus = []
        for line in lines:
            tokens = [token for token in line]
            corpus.append(vocab[tokens]) # nest list
    else:
        lines = filter_lines(lines)
        corpus = [vocab[token] for line in all_tokens for token in line]
        
    corpus_size = len(lines) * len([lines[0]]) if PADDING else len(corpus)
    train_size = int(train_size * corpus_size)
    val_size = int(val_size * corpus_size)
    train_corpus = corpus[:train_size]
    val_corpus = corpus[train_size :  train_size + val_size]
    test_corpus = corpus[train_size + val_size: ]
    
    return train_corpus, val_corpus, test_corpus, vocab


def generate_dataset(corpus, step):
    data_num = len(corpus) // step * step
    data = np.array(corpus[: data_num], dtype=np.int16).reshape(-1, step) # when load into nn.Embedding, remember to convert to Long type.
        
    return data

        
def FNNloader(data, index, step):
    x = torch.tensor(data[index][: step - 1])
    y = torch.tensor(data[index][step - 1])
    
    return x, y


def RNNloader(data, index, step):
    x = torch.tensor(data[index])
    
    # remember that padding index is 0, end index is 1, unknow index is 2
    y = torch.tensor(np.concatenate([data[index][1:], np.array([0], dtype=np.int32)]))
    
    return x, y
    

class MyDataset(Dataset):
    def __init__(self, data, step, index, loader):
        self.data = data
        self.step = step
        self.loader = loader
        self.index = index #list
        
    def __getitem__(self, index):
        data_index = self.index[index]
        x, y = self.loader(self.data, data_index, self.step)
        
        return x, y
    
    def __len__(self):
        return len(self.index)


def load_data_end2end(step, batch_size, loader, num_workers):
    print("loading data")
    trainset, valset, testset, vocab = divide_corpus()
    print("load finish")
    
    if not PADDING:
        trainset = generate_dataset(trainset, step)
        valset = generate_dataset(valset, step)
        testset = generate_dataset(testset, step)
    
    train_index = list(range(len(trainset)))
    val_index = list(range(len(valset)))
    test_index = list(range(len(testset)))
        
    trainset = MyDataset(trainset, step=step, index=train_index, loader=loader)
    valset = MyDataset(valset, step=step, index=val_index, loader=loader)
    testset = MyDataset(testset, step=step, index=test_index, loader=loader)
        
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    
    return trainloader, valloader, testloader, vocab


# if __name__ == "__main__":
#     path = './padding_data/'
#     step = 7
#     batch_size = 128
#     loader = RNNloader
#     t1 = time.time()
#     trainloader, valloader, testloader, vocab = load_data_end2end(step, batch_size, loader)
#     print(time.time() - t1)
#     x, y = iter(trainloader).next()
#     print(x.shape, y.shape)
#     print(x[0])
#     print(y[0])