import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from tqdm import tqdm
from preprocess import prepare_data
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('./LSTM_CLF_batchsize128')


class MaskedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first, 
                       bias=True, dropout=0, bidirectional=False):
        super().__init__()
        self.batch_first = batch_first
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first, 
                            bias, dropout, bidirectional)
    
    def forward(self, x, seq_length):
        # x:[batch_size, max_seq_length, d] x if a embedded tensor; look-up-table size: [vocab size, d]
        total_length = x.shape[1]
        packed_x = pack_padded_sequence(x, lengths=seq_length, batch_first=self.batch_first, enforce_sorted=False)
        y, (h, c) = self.lstm(packed_x)
        y_padded, length = pad_packed_sequence(y, batch_first=self.batch_first, total_length=total_length)
        
        return y_padded, (h, c), length


class LSTM_CLF(nn.Module):
    '''
    directin: 1
    input: [batchsize, 512]
    output: [batchsize, 5]
    '''
    def __init__(self, v, d, output_size, hidden_size, num_layers):
        super().__init__()
        self.v = v # vocab size
        self.d = d # look up table size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(v, d)
        self.lstm = MaskedLSTM(input_size=d, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.linear1 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
    
    def forward(self, x):
        seq_lengths = torch.where(x == 2)[1].to('cpu')
        # print(x, seq_lengths)
        # x:[batch_size, len(sentence)=512 (after padding)]
        x = self.embedding(x) # [batch_size, 512, d]
        output, (h, c), length = self.lstm(x, seq_lengths)
        '''
        output:[batch_size, 512, hiddensize]
        512 is the max sequence length, 
        h: [num_layers, batch_size, hidden_size]
        c: [num_layers, batch_size, hidden_size]
        '''
        # res = self.linear1(self.dropout(h))
        output = self.maxpool(output.permute([0, 2, 1])).permute([0, 2, 1])
        res = self.linear1(output)
        
        return res[:, 0]
    
    def init_hidden_state(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)
    
    
def run_one_epoch(net, data_iter, optimizer, loss_func, device):
    total_loss, n, correct = 0, 0, 0 # loss in one epoch
    for feature, label in tqdm(data_iter, ncols=80):
        feature, label = feature.long().to(device), label.long().to(device)
        y_hat = net(feature)
        label_hat = torch.argmax(F.softmax(y_hat, dim=1), dim=1)
        correct += torch.sum(label_hat == label).item()
        loss = loss_func(y_hat, label)
        total_loss += loss # loss in a batch
        n += 1
        if net.training:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    loss_a_epoch = total_loss / n
    acc = correct / (n * feature.shape[0])
    
    return loss_a_epoch, acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    num_workers = 1
    lr = 1e-3
    d = 128 # look up table feature size
    hidden_size = 64
    best_acc = -1.0
    
    trainloader, devloader, testloader, vocab = prepare_data(batch_size, num_workers)
    v = len(vocab)
    print("device: ", device)
    print("batch size: %d\nlr: %.3f\nlook up table size: (%d, %d)\nLSTM hidden size: %d\n"
          %(batch_size, lr, v, d, hidden_size))
    
    net = LSTM_CLF(v=v, d=d, output_size=5, hidden_size=hidden_size, num_layers=1).to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr)
    
    epoch = 0
    print("STRAT TO TRAIN!")
    while True:
        epoch += 1
        net.train()
        train_loss_a_epoch, train_acc = run_one_epoch(net, trainloader, optimizer, loss_func, device)
        print("epoch: %d|train loss: %.3f|train acc: %.3f"%(epoch, train_loss_a_epoch, train_acc))
        
        # validation
        net.eval()
        with torch.no_grad():
            val_loss_a_epoch, val_acc = run_one_epoch(net, devloader, optimizer, loss_func, device)
        print("epoch: %d|val loss: %.3f|val acc: %.3f"%(epoch, val_loss_a_epoch, val_acc))
            
        if epoch % 5 == 0:
            print("===== TEST =====")
            net.eval()
            with torch.no_grad():
                test_loss_a_epoch, test_acc = run_one_epoch(net, testloader, optimizer, loss_func, device)
            print("epoch: %d|test loss: %.3f|test acc: %.3f"%(epoch, test_loss_a_epoch, test_acc))
            if test_acc > best_acc:
                best_acc = val_acc
                save_path = "./model_weights/batchsize_%d_test_acc%.3f.tar"%(batch_size, val_acc)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr': lr,
                    'batch_size': batch_size,
                    'hidden_size': hidden_size,
                    'look_up_tabel': d,
                    'train_loss': train_loss_a_epoch,
                    'val_loss': val_loss_a_epoch, 
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                }, save_path)
            
            writer.add_scalars('loss', {'train_loss': train_loss_a_epoch,
                                        'validation_loss': val_loss_a_epoch,
                                        'test_loss': test_loss_a_epoch}, epoch)
            writer.add_scalars('acc', {'train_acc': train_acc,
                                        'validation_acc': val_acc,
                                        'test_acc': test_acc}, epoch)


if __name__ == "__main__":
    main()