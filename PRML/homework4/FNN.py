import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt


def prepare_data():
    class1 = [
        [ 1.58, 2.32, -5.8], [ 0.67, 1.58, -4.78], [ 1.04, 1.01, -3.63], 
        [-1.49, 2.18, -3.39], [-0.41, 1.21, -4.73], [1.39, 3.16, 2.87],
        [ 1.20, 1.40, -1.89], [-0.92, 1.44, -3.22], [ 0.45, 1.33, -4.38],
        [-0.76, 0.84, -1.96]
    ]
    label1 = []
    for _ in range(len(class1)):
        label1.append([1.0, 0.0, 0.0])
    data1 = list(zip(class1, label1))
        
    class2 = [
        [ 0.21, 0.03, -2.21], [ 0.37, 0.28, -1.8], [ 0.18, 1.22, 0.16], 
        [-0.24, 0.93, -1.01], [-1.18, 0.39, -0.39], [0.74, 0.96, -1.16],
        [-0.38, 1.94, -0.48], [0.02, 0.72, -0.17], [ 0.44, 1.31, -0.14],
        [ 0.46, 1.49, 0.68]
    ]
    label2 = []
    for _ in range(len(class2)):
        label2.append([0.0, 1.0, 0.0])
    data2 = list(zip(class2, label2))
        
    class3 = [
        [-1.54, 1.17, 0.64], [5.41, 3.45, -1.33], [ 1.55, 0.99, 2.69], 
        [1.86, 3.19, 1.51], [1.68, 1.79, -0.87], [3.51, -0.22, -1.39],
        [1.40, -0.44, -0.92], [0.44, 0.83, 1.97], [ 0.25, 0.68, -0.99],
        [ 0.66, -0.45, 0.08]
    ]
    label3 = []
    for _ in range(len(class3)):
        label3.append([0.0, 0.0, 1.0])
    data3 = list(zip(class3, label3))
    
    return data1, data2, data3


def generate_dataset():
    data1, data2, data3 = prepare_data()
    
    return data1 + data2 + data3


def myloader(data, index):
    x = data[index][0]
    y = data[index][1]
    
    return x, y


class MyDataset(Dataset):
    def __init__(self, data, loader) -> None:
        self.data = data
        self.loader = loader
    
    def __getitem__(self, index):
        x, y = self.loader(self.data, index)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        
        return x, y
    
    def __len__(self):
        return len(self.data)
    

def load_data(batch_size, loader=myloader):
    trainset = generate_dataset()
    trainset = MyDataset(trainset, loader)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    return train_loader
    

class FNN(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fnn = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 3),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        out = self.fnn(x)
        
        return out


def main():
    torch.cuda.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 15 # 1, 30
    lr = 0.1 # 0.001, 0.01, 0.1
    epochs = 1000
    hidden_size = 12 # 4, 8, 12, 16
    
    trainloader = load_data(batch_size)
    
    net = FNN(hidden_size).to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr)
    loss_fn = nn.MSELoss()
    
    loss_list, acc_list = [], []

    # train
    for epoch in range(epochs):
        train_loss = 0
        correct, counter= 0, 0
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            y_hat = net(x)
            mask = torch.max(y_hat, dim=1)[1]
            loss = loss_fn(y_hat, y)
            train_loss += loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            correct += torch.sum(y[list(range(len(mask))), mask]).item()
            counter += len(mask)
        acc = correct / counter
        loss_list.append(train_loss / len(trainloader))
        acc_list.append(acc)
        print("epoch: %d|train loss: %.3f|correct %.3f"%(epoch + 1, train_loss / len(trainloader), acc))
        if acc >= 0.8:
            state = {
                'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }
            torch.save(state, './FNN_%.3f.pth'%acc)
    
    plt.plot(list(range(len(loss_list))), loss_list)
    plt.grid()
    plt.title('train loss(hidden neurons: %d, learning rate: %.3f, batch_size: %d)'%(hidden_size, lr, batch_size))
    plt.savefig('./loss_%d_%.3f_%d.png'%(hidden_size, lr, batch_size))
    plt.cla()
    plt.plot(list(range(len(acc_list))), acc_list)
    plt.title('train accuracy(hidden neurons: %d, learning rate: %.3f, batch_size: %d)'%(hidden_size, lr, batch_size))
    plt.grid()
    plt.savefig('./acc_%d_%.3f_%d.png'%(hidden_size, lr, batch_size))
    
    
if __name__ == "__main__":
    main()