import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'
import sys
sys.path.append('../')
from text_preprocess import load_padding_dataset, RNNloader, load_data_end2end
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter('./RNNlog_padding2')


class RNNLM(nn.Module):
    def __init__(self, vocab_size, d, step, hidden_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.d = d
        self.step = step
        self.hidden_size = hidden_size
        self.C = nn.Embedding(self.vocab_size, self.d) # shape: [batch_size, step, d]
        self.rnn = nn.RNN(self.d, self.hidden_size, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.vocab_size)
        self.flat = nn.Flatten(start_dim=0, end_dim=1)
        
    def forward(self, x): # x shape: [batch_size, sentence length] input are sentences!
        # x = F.one_hot(x.T, self.vocab_size).type(torch.float32).to(self.device)
        x = self.C(x) # [batch_size, step, d]
        y, hidden_state = self.rnn(x) 
        # y.shape=[batch_size, step, D * hidden_size] (D=2 if use bidirectional RNN)
        # hidden_state.shape = [D * num_layers, batch_size, hidden_size] (num_layers=1 default)
        
        # y.shape=(batch_size, num_steps, hidden_size)
        # hidden_state.shape=(D*num_layers, N, H) D=2 if bidirectional=True otherwise 1
        y = self.flat(y) # y.shape=[batch_size * step, D * hidden_size] = [batch_size * step, hidden_size]
        y = self.linear(y) # y.shape = [batch_size * step, vocab_size]
        
        return y

    def init_hidden_state(self, batch_size):
        return torch.zeros(self.rnn.num_layers, batch_size, self.hidden_size)
    

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
    perplexity = 2 ** (total_loss / len(data_iter.dataset)).item()
    
    return perplexity


@torch.no_grad()
def test_net(path='./RNN_model_weights2/RNN_params_17.868.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1024
    step = 7
    lr = 1e-3
    d = 512 # look up table feature size
    hidden_size = 128
    print("device: ", device)
    print("batch size: %d\ngram: %d\nlr: %.3f\nlook up table: %d\n"
          %(batch_size, step, lr, d))

    trainloader, valloader, testloader, vocab = load_data_end2end(step, batch_size, loader=RNNloader, num_workers=16)
    v = len(vocab)
    print("vocab size: ", v)
    net = RNNLM(v, d, step - 1, hidden_size).to(device)
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
    print("device: ", device)
    print("batch size: %d\ngram: %d\nlr: %.3f\nlook up table: %d\nRNN hidden size: %d\n"
          %(batch_size, step, lr, d, hidden_size))
    
    trainloader, valloader, testloader, vocab = load_data_end2end(step, batch_size, loader=RNNloader, num_workers=16)
    v = len(vocab)
    print("vocab size: %d"%v)
    net = RNNLM(v, d, step - 1, hidden_size).to(device)
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
            torch.save(net.state_dict(), './RNN_model_weights2/RNN_params_%.3f.pth'%best_ppl)
        
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
    # test_net()