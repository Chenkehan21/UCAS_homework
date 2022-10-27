from turtle import forward
import torch
import torch.nn as nn
import tqdm
from utils import trainer
from NLP.homework2.read_long_sequence import load_data_iter


class FNNLM(nn.Module):
    def __init__(self, v, d, step, hidden=50):
        super().__init__()
        # input shape: [batch_size, step] (step is the number of sentence)
        self.v = v # number of words
        self.d = d # dimension of a word's feature 
        self.step = step
        self.hidden = hidden
        
        # self.net = nn.Sequential(
        #     nn.Embedding(self.v, self.d), # shape: [batch size, step, d]
        #     nn.Flatten(), # shape: [batch size, step * d]
        #     nn.Linear(self.step * self.d, 50), nn.Tanh(), # shape: [batch size, 50]
        #     nn.Linear(512, 512), nn.ReLU(),
        #     nn.Linear(512, 512), nn.ReLU(),
        #     nn.Linear(512, self.step) # nn.CrossEntropy() has softmax
        # )
        
        self.C = nn.Embedding(self.v, self.d) # shape: [batch size, step, d]
        self.flat = nn.Flatten()# shape: [batch size, step * d]
        self.lin1 = nn.Linear(self.step * self.d, self.hidden) # shape: [batch size, 50]
        self.tanh = nn.Tanh()
        self.lin2 = nn.Linear(self.step * self.d, self.v)
        self.lin3 = nn.Linear(self.hidden, self.v)
        
    def forward(self, x):
        embedding_x = self.C(x)
        flatted_x = self.flat(embedding_x)
        res1 = self.tanh(self.lin1(flatted_x))
        res2 = self.lin2(flatted_x)
        res3 = self.lin3(res1)
        res = res2 + res3
        
        return res
    

def init_weights(layer):
    if type(layer) == nn.Linear:
        nn.init.normal_(layer.weight, mean=0, std=0.01)
        
        
def train(net, data_iter, optimizer, loss_func, epoches, device):
    total_perplexity = []
    for epoch in range(epoches):
        net.train()
        total_loss, n = 0, 0
        for feature, label in tqdm(data_iter):
            feature, label = feature.to(device), label.to(device)
            y_hat = net(feature)
            loss = loss_func(y_hat, label)
            total_loss += loss * label.numel()
            n += label.numel()
            loss.backward()
            optimizer.zero_grad()
            optimizer.step()
        perplexity = torch.exp(total_loss / n).item()
        total_perplexity.append(perplexity)
        print("epoch: %d|perplexity: %.3f" % (epoch + 1, perplexity))
    
    
def main(batch_size=64, step=256, lr=1e-1, epoches=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data_iter, test_data_iter = load_data_iter(batch_size=batch_size, setp=step, use_FNNML=True, use_random_sample=False)
    v = len(train_data_iter.vocab)
    d = 50
    net = FNNLM(v, d, step).to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr)
    
    train(net, train_data_iter, optimizer, loss_func, epoches, device)
    

if __name__ == "__main__":
    main()