import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import torch
import torch.nn as nn
from tqdm import tqdm
from text_preprocess import FNNloader, load_data_end2end
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter('./FNNlog_5gram2')


class FNNLM(nn.Module):
    def __init__(self, v, d, step, hidden=8):
        super().__init__()
        # input shape: [batch_size, step] (step is the number of sentence)
        self.v = v # number of words
        self.d = d # dimension of a word's feature 
        self.step = step
        self.hidden = hidden
        
        self.C = nn.Embedding(self.v, self.d) # shape: [batch size, step, d]
        self.flat = nn.Flatten()# shape: [batch size, step * d]
        self.lin1 = nn.Linear(self.step * self.d, self.hidden) # shape: [batch size, 50]
        self.tanh = nn.Tanh()
        self.lin2 = nn.Linear(self.step * self.d, self.v)
        self.lin3 = nn.Linear(self.hidden, self.v)
        
    def forward(self, x): # x shape: [batch_size, step]
        embedding_x = self.C(x)
        flatted_x = self.flat(embedding_x)
        res1 = self.tanh(self.lin1(flatted_x))
        res2 = self.lin2(flatted_x)
        res3 = self.lin3(res1)
        res = res2 + res3
        
        return res
        
        
def run_one_epoch(net, data_iter, optimizer, loss_func, device, hidden_state=None):
    total_loss, n = 0, 0 # loss in one epoch
    for feature, label in tqdm(data_iter, ncols=80):
        feature, label = feature.long().to(device), label.long().to(device)
        # print(feature, feature.shape)
        # print(label, label.shape)
        if hidden_state == None:
            y_hat = net(feature)
        else:
            hidden_state.detach_() # hidden state 
            y_hat, hidden_state = net(feature, hidden_state)
            label = label.flatten()
        loss = loss_func(y_hat, label)
        total_loss += loss * feature.shape[0] # label.numel() = batchsize, loss in one batch
        # print("feature.shape[0]: ", )
        # n += 1
        if net.training:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    perplexity = 2 ** (total_loss / len(data_iter.dataset)).item() # total_loss / n is the average cross entropy loss of n grams
    
    return perplexity


@torch.no_grad()
def test_net(path='./FNN_model_weights2/FNN_params_20.877.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1024
    step = 5
    lr = 1e-3
    d = 64 # look up table feature size
    print("device: ", device)
    print("batch size: %d\ngram: %d\nlr: %.3f\nlook up table: %d\n"
          %(batch_size, step, lr, d))

    trainloader, valloader, testloader, vocab = load_data_end2end(step, batch_size, loader=FNNloader, num_workers=16)
    v = len(vocab)
    print("vocab size: ", v)
    net = FNNLM(v, d, step-1).to(device)
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
    step = 5
    lr = 1e-3
    best_ppl = 1e10
    d = 64 # look up table feature size
    print("device: ", device)
    print("batch size: %d\ngram: %d\nlr: %.3f\nlook up table: %d\n"
          %(batch_size, step, lr, d))

    trainloader, valloader, testloader, vocab = load_data_end2end(step, batch_size, loader=FNNloader, num_workers=16)
    v = len(vocab)
    print("vocab size: ", v)
    net = FNNLM(v, d, step-1).to(device)
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
            torch.save(net.state_dict(), './FNN_model_weights2/FNN_params_%.3f.pth'%best_ppl)
        
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
    # main()
    test_net()