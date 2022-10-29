import torch
import torch.nn as nn
from tqdm import tqdm
from end2end_dataload import load_data_iter
from prepare_dataloader import load_data, FNNloader
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter('./FNNlog')


class FNNLM(nn.Module):
    def __init__(self, v, d, step, hidden=50):
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
        
    def forward(self, x):
        embedding_x = self.C(x)
        flatted_x = self.flat(embedding_x)
        res1 = self.tanh(self.lin1(flatted_x))
        res2 = self.lin2(flatted_x)
        res3 = self.lin3(res1)
        res = res2 + res3
        
        return res
        
        
def run_one_epoch(net, data_iter, optimizer, loss_func, device):
    total_loss, n = 0, 0 # loss in one epoch
    for feature, label in tqdm(data_iter, ncols=80):
        feature, label = feature.to(device), label.to(device)
        y_hat = net(feature)
        loss = loss_func(y_hat, label)
        total_loss += loss * label.numel() # label.numel() = batchsize, loss in one batch
        n += label.numel()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    perplexity = torch.exp2(total_loss / n).item() # total_loss / n is the average cross entropy loss of n grams
    
    return perplexity    
        
        
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    step = 7
    lr = 1e-3
    best_ppl = 1e10
    d = 100 # look up table feature size
    
    # train_data_iter, val_data_iter,_ , _, _ = load_data_iter(batch_size=batch_size, step=step, use_FNNML=True, use_random_sample=False)
    trainloader, valloader, testloader, train_vocab, val_vocab, test_vocab = load_data(path='././data_prepare/', step=step, batch_size=batch_size, loader=FNNloader)
    v = len(train_vocab)
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
            torch.save(net.state_dict(), 'FNN_params_%.3f.pth'%best_ppl)
        
        # validation
        net.eval()
        print("validate:")
        val_perplexity = run_one_epoch(net, valloader, optimizer, loss_func, device)
        total_perplexity_val.append(val_perplexity)
        print("epoch: %d|train ppl: %.3f|validation ppl: %.3f" % (epoch, train_perplexity, val_perplexity))
        writer.add_scalar('Perplexity/train', train_perplexity, epoch)
        writer.add_scalar('Perplexity/validation', val_perplexity, epoch)
        writer.add_scalars('train_val',{'train': train_perplexity,
                                       'val': val_perplexity}, epoch)
    

if __name__ == "__main__":
    main()