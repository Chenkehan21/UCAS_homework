import numpy as np
import pandas as pd


def get_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        dataset = pd.read_csv(f)
    
    train_data = []
    label_pool = []
    for i in range(1, 5):
        w_x = list(dataset['x%d'%i][:8])
        w_y = list(dataset['y%d'%i][:8])
        ones = [1] * len(w_x)
        train_data.append(np.array([w_x, w_y, ones]))
        
        c_label = np.array([0] * 4)
        c_label[i -1] = 1
        label = np.tile(c_label, (8, 1))
        label_pool.append(label)
        
    train_set = np.concatenate(train_data, axis=1)
    labels = np.concatenate(label_pool, axis=0).T
    
    test_data = []
    for i in range(1, 5):
        w_x = list(dataset['x%d'%i][-2:])
        w_y = list(dataset['y%d'%i][-2:])
        ones = [1] * len(w_x)
        test_data.append(np.array([w_x, w_y, ones]))
    test_set = np.concatenate(test_data, axis=1)
    
    return train_set, test_set, labels


def mse(X, Y):
    w = np.dot(np.dot(np.linalg.inv(np.dot(X, X.T)), X), Y.T)

    return w

def test(train_data, label, test_data):
    w = mse(train_data, label) #(3, 4)
    y_hat = np.dot(w.T, test_data) # (4, 3) (3, 8) = (4, 8)
    res = np.argmax(y_hat, axis=0) + 1 # (1, 8)
    y = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    correct = np.sum(res == y)
    acc = correct / 8
    
    return acc

if __name__ == "__main__":
    path = './data.csv'   
    train_set, test_set, labels = get_data(path)
    acc = test(train_set, labels, test_set)
    print(acc)