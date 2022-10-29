from data_preprocess import load_data as load_raw_data
import pandas as pd
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader


def generate_dataset(corpus, step, filename, path='./data_prepare/'):
    data_num = len(corpus) // step * step
    data = np.array(corpus[: data_num]).reshape(-1, step)
    data_pd = pd.DataFrame(data)
    data_pd.to_csv(path + '%s.csv'%filename)
    

# def generate_RNNdataset(corpus, step, filename, path='./data_prepare/'):
#     data_num = len(corpus) // step
#     data = np.array(corpus[: data_num]).reshape(-1, step)
#     data_pd = pd.DataFrame(data)
#     data_pd.to_csv(path + '%s_data.csv'%filename)
    
#     label = np.array(corpus[1 : data_num + 1]).reshape(-1, step)
#     label_pd = pd.DataFrame(label)
#     label_pd.to_csv(path + '%s_label.csv'%filename)
    

def FNNloader(data, index, step):
    x = torch.tensor(data[index][: step - 1])
    y = torch.tensor(data[index][step - 1])
    
    return x, y


def RNNloader(data, index, step):
    x = torch.tensor(data[index][ : step - 1])
    y = torch.tensor(data[index][1: ])
    
    return x, y
    

class MyDataset(Dataset):
    def __init__(self, data, step, index, loader):
        self.data = data
        self.step = step
        self.loader = loader
        self.index = index
        
    def __getitem__(self, index):
        data_index = self.index[index]
        x, y = self.loader(self.data, data_index, self.step)
        
        return x, y
    
    def __len__(self):
        return len(self.index)


def prepare_data(steps=[7, 9, 11, 13], path='./data_prepare/'):
    vocab, corpus,\
    train_corpus, train_vocab,\
    val_vocab, val_corpus,\
    test_corpus, test_vocab = load_raw_data()
    print("load finish")
    with open(path + 'train_vocab.pkl', 'wb') as f:
        pickle.dump(train_vocab, f)
    with open(path + 'val_vocab.pkl', 'wb') as f:
        pickle.dump(val_vocab, f)
    with open(path + 'test_vocab.pkl', 'wb') as f:
        pickle.dump(test_vocab, f)
    for step in steps:
        print(step)
        generate_dataset(train_corpus, step, 'trainset_%dgram'%step, path)
        generate_dataset(val_corpus, step, 'valset_%dgram'%step, path)
        generate_dataset(test_corpus, step, 'testset_%dgram'%step, path)


def load_data(path, step, batch_size, loader):
    with open(path + 'trainset_%dgram.csv'%step, 'r') as f:
        trainset = pd.read_csv(f, index_col=0).to_numpy()
    with open(path + 'valset_%dgram.csv'%step, 'r') as f:
        valset = pd.read_csv(f, index_col=0).to_numpy()
    with open(path + 'testset_%dgram.csv'%step, 'r') as f:
        testset = pd.read_csv(f, index_col=0).to_numpy()
        
    with open(path + 'train_vocab.pkl', 'rb') as f:
        train_vocab = pickle.load(f)
    with open(path + 'val_vocab.pkl', 'rb') as f:
        val_vocab = pickle.load(f)
    with open(path + 'test_vocab.pkl', 'rb') as f:
        test_vocab = pickle.load(f)
    
    train_index = list(range(len(trainset)))
    val_index = list(range(len(valset)))
    test_index = list(range(len(testset)))
        
    trainset = MyDataset(trainset, step=step, index=train_index, loader=loader)
    valset = MyDataset(valset, step=step, index=val_index, loader=loader)
    testset = MyDataset(testset, step=step, index=test_index, loader=loader)
        
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)
    
        
    return trainloader, valloader, testloader,\
           train_vocab, val_vocab, test_vocab
           

# if __name__ == "__main__":
#     prepare_data()