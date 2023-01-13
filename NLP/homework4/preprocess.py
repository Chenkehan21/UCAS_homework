import collections
import re
import numpy as np
from zhon.hanzi import punctuation as cn_punc
from string import punctuation as en_punc
from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence


def load_data(path='./NLP_TC/traindata.txt'):
    punc = cn_punc + en_punc
    with open(path, 'r') as f:
        text = f.readlines()
    
    sentences = []
    labels = []
    for line in text:
        line = line.split('\t')
        label = line[0]
        content = line[1].strip()
        content = re.sub("[{}]+".format(punc), '', content)
        content = re.sub(r'[\r|\n|\t|]', '', content)
        content = content.split(' ')
        content = [i for i in content if i != '']
        content.append('<E>')
        sentences.append(content)
        labels.append(label)
    label_type = list(set(labels))
    label_map = {key:0+i for i, key in enumerate(label_type)}
    labels = [label_map[item] for item in labels]
        
    return sentences, labels


class Vocab:
    def __init__(self, min_freq=10, reserved_tokens=['<U>'], tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
            
        freq = collections.Counter(tokens)
        self.freq_tokens = sorted(freq.items(), key=lambda x: x[1], reverse=True)

        res = reserved_tokens # padding index will be 0
        self.unk = res.index('<U>')
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
    
    
def tokenlize(sentences, vocab):
    corpus = []
    for sentence in sentences:
        tokens = [token for token in sentence]
        corpus.append(vocab[tokens][:512])
    data_pad = pad_sequence([torch.from_numpy(np.array(x)) for x in corpus], batch_first=True).float()
    for item in data_pad:
        if item[-1] != 0:
            item[-1] = 2
        
    return data_pad


class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.label[index]
        
        return x, y
    
    def __len__(self):
        return len(self.data)
    
    
def prepare_data(batch_size, num_workers):
    print("loading data")
    train_path = "./NLP_TC/traindata.txt"
    dev_path = "./NLP_TC/devdata.txt"
    test_path = "./NLP_TC/testdata.txt"
    train_data, train_label = load_data(train_path)
    dev_data, dev_label = load_data(dev_path)
    test_data, test_label = load_data(test_path)
    
    print("trainset size: %d"%len(train_data))
    print("devset size: %d"%len(dev_data))
    print("testset size: %d"%len(test_data))
    
    all_tokens = [token for sentence in train_data for token in sentence]
    vocab = Vocab(min_freq=10, reserved_tokens=['<P>','<U>','<E>'], tokens=all_tokens)
    train_data, dev_data, test_data = tokenlize(train_data, vocab), tokenlize(dev_data, vocab), tokenlize(test_data, vocab)
    
    trainset = MyDataset(train_data, train_label)
    devset = MyDataset(dev_data, dev_label)
    testset = MyDataset(test_data, test_label)
    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    devloader = DataLoader(devset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    print("load finish")
    
    return trainloader, devloader, testloader, vocab