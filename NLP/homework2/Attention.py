import os
os.environ["CUDA_VISIBLE_DEVICES"]='2'
from text_preprocess import RNNloader, load_data_end2end
import math
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('./Attention_log')


class MultiHeadSelfAttention(nn.Module):
    dim_in: int  # input dimension
    dim_k: int   # key and query dimension
    dim_v: int   # value dimension
    num_heads: int  # number of heads, for each head, dim_* = dim_* // num_heads

    def __init__(self, dim_in, dim_k, dim_v, num_heads, device):
        super(MultiHeadSelfAttention, self).__init__()
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.device = device
        
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / math.sqrt(dim_k // num_heads)


    def forward(self, x):
        batch, n, dim_in = x.shape # dim_in = look up table size
        assert dim_in == self.dim_in

        nh = self.num_heads
        dk = self.dim_k // nh  # dim_k of each head
        dv = self.dim_v // nh  # dim_v of each head

        q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)
        k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)
        v = self.linear_v(x).reshape(batch, n, nh, dv).transpose(1, 2)
        shape = (q.shape[0],q.shape[1],q.shape[2],q.shape[2])
        mask = -(1e9) * torch.tensor(np.triu(np.ones(shape), k=1))

        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact
        dist += mask.to(self.device) # only use previous words to predict
        dist = torch.softmax(dist, dim=-1)  # batch, nh, n, n

        att = torch.matmul(dist, v)  # batch, nh, n, dv
        att = att.transpose(1, 2).reshape(batch, n, self.dim_v)  # batch, n, dim_v
        
        return att


class Self_Attention(nn.Module):
    def __init__(self, vocab_size, look_up_table_size, num_heads, device):
        super(Self_Attention, self).__init__()
        self.vocab_size = vocab_size
        self.d = look_up_table_size
        self.num_heads=num_heads
        self.device = device
        
        self.embed = nn.Embedding(self.vocab_size, self.d)
        self.attention = MultiHeadSelfAttention(self.d, self.d, self.d, self.num_heads, self.device)#K,Q,V,dim_in,headers
        self.layer_norm1 = nn.LayerNorm(self.d, eps=1e-12)
        self.relu = nn.ReLU()
        self.layer_norm2 = nn.LayerNorm(self.d, eps=1e-12)
        self.dense1 = nn.Linear(self.d, self.d)
        self.dense2 = nn.Linear(self.d, self.d)
        self.dense = nn.Linear(self.d, self.vocab_size)


    def forward(self, inputs): # inputs: (batch, seq_len)
        num_steps = inputs.shape[1]
        self.valid_lens = torch.tensor(list(range(1, num_steps + 1)))
        X = self.embed(inputs) # X.shap=[batch size, sentence length, look up table size]
        Y1=self.attention(X)
        # residual 
        Y1 =Y1+X
        Y1 = self.layer_norm1(Y1)
        Y = self.dense1(Y1)
        Y = self.relu(Y)
        Y = self.dense2(Y)
        Y =Y+Y1
        Y = self.layer_norm2(Y)
        output=self.dense(Y.view(-1,Y.shape[-1]))

        return output
    

def run_one_epoch(net, data_iter, optimizer, loss_func, device):
    total_loss, n = 0, 0 # loss in one epoch
    for feature, label in tqdm(data_iter, ncols=80):
        feature, label = feature.long().to(device), label.long().to(device)
        label = label.flatten()
        mask = label >= 1
        valid_label = label[mask]
        y_hat = net(feature)
        valid_y_hat = y_hat[mask]
        loss = loss_func(valid_y_hat, valid_label)
        total_loss += loss * feature.shape[0]
        if net.training:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    perplexity = 2 ** (total_loss / len(data_iter.dataset)).item() # total_loss / n is the average cross entropy loss of n grams
    
    return perplexity


@torch.no_grad()
def test_net(path='./Attention_model_weights/Attention_params_13.751.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1024
    step = 7
    lr = 1e-3
    d = 512 # look up table feature size
    num_headers = 8
    print("device: ", device)
    print("batch size: %d\ngram: %d\nlr: %.3f\nlook up table: %d\n"
          %(batch_size, step, lr, d))

    trainloader, valloader, testloader, vocab = load_data_end2end(step, batch_size, loader=RNNloader, num_workers=16)
    v = len(vocab)
    print("vocab size: ", v)
    net = Self_Attention(v, d, num_headers, device).to(device)
    net.load_state_dict(torch.load(path))
    net.eval()
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr)
    ppl = run_one_epoch(net, testloader, optimizer, loss_func, device)
    print(ppl)

    return ppl
    
    
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1024
    step = 7
    lr = 1e-3
    best_ppl = 1e10
    d = 512 # look up table feature size
    hidden_size = 128
    num_headers = 8
    print("device: ", device)
    print("batch size: %d\ngram: %d\nlr: %.3f\nlook up table: %d\nRNN hidden size: %d\nnum headers: %d\n"
          %(batch_size, step, lr, d, hidden_size, num_headers))
    
    trainloader, valloader, testloader, vocab = load_data_end2end(step, batch_size, loader=RNNloader, num_workers=16)
    v = len(vocab)
    print("vocab size: %d"%v)
    
    net = Self_Attention(v, d, num_headers, device).to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr)
    
    total_perplexity_train, total_perplexity_val = [], []
    epoch = 0
    while True:
        epoch += 1
        net.train()
        print("train:")
        train_perplexity = run_one_epoch(net, trainloader, optimizer, loss_func, device)
        total_perplexity_train.append(train_perplexity)
        if train_perplexity < best_ppl:
            best_ppl = train_perplexity
            torch.save(net.state_dict(), './Attention_model_weights/Attention_params_%.3f.pth'%best_ppl)
        
        # validation
        net.eval()
        print("validate:")
        with torch.no_grad():
            val_perplexity = run_one_epoch(net, valloader, optimizer, loss_func, device)
        total_perplexity_val.append(val_perplexity)
        print("epoch: %d|train ppl: %.3f|validation ppl: %.3f" % (epoch, train_perplexity, val_perplexity))
        writer.add_scalar('Perplexity/train', train_perplexity, epoch)
        writer.add_scalar('Perplexity/validation', val_perplexity, epoch)
        writer.add_scalars('train_val',{'train': train_perplexity,
                                       'val': val_perplexity}, epoch)

        if epoch % 5 == 0:
            net.eval()
            with torch.no_grad():
                test_perplexity = run_one_epoch(net, testloader, optimizer, loss_func, device)
            print("epoch: %d|test ppl: %.3f"%(epoch, test_perplexity))

if __name__ == "__main__":
    main()