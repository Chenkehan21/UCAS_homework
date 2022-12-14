import numpy as np
import scipy.io
from tqdm import tqdm
import matplotlib.pyplot as plt

np.random.seed(1234)


def load_data(path='./vehicle.mat'):
    all_data = scipy.io.loadmat(path)['UCI_entropy_data'][0, 0]['train_data'].T #(846, 19)
    np.random.shuffle(all_data)
    n, _ = all_data.shape
    split = int(0.8 * n)
    train_data = all_data[:split, :]
    test_data = all_data[split:, :]
    
    return train_data, test_data


def data_preprocess(data):
    x1, x2, x3, x4 = [], [], [], []
    n, _ = data.shape
    for i in range(n):
        label = data[i][-1]
        if label == 1:
            x1.append(data[i])
        if label == 2:
            x2.append(data[i])
        if label == 3:
            x3.append(data[i])
        if label == 4:
            x4.append(data[i])
    
    return np.array(x1), np.array(x2), np.array(x3), np.array(x4)


def calculate_s(x, u):
    n = u.shape[0]
    res = np.zeros((n, n))
    for i in range(x.shape[0]):
        t = x[i][:-1].reshape(-1, 1)
        res += np.dot((t - u), (t - u).T)
    
    return res


def LDA(data, k):
    x1, x2, x3, x4 = data_preprocess(data) # train data: (676 , 19)
    mean = np.mean(data[:, :-1], axis=0)
    mean1 = np.mean(x1[:, :-1], axis=0)
    mean2 = np.mean(x2[:, :-1], axis=0)
    mean3 = np.mean(x3[:, :-1], axis=0)
    mean4 = np.mean(x4[:, :-1], axis=0)
    
    # S_t = calculate_s(data, mean)
    S_w1 = calculate_s(x1, mean1)
    S_w2 = calculate_s(x2, mean2)
    S_w3 = calculate_s(x3, mean3)
    S_w4 = calculate_s(x4, mean4)
    S_w = S_w1 + S_w2 + S_w3 + S_w4
    # S_b = S_t - S_w
    
    n = mean.shape[0]
    S_b = np.zeros((n, n))
    for u, x in [(mean1, x1), (mean2, x2), (mean3, x3), (mean4, x4)]:
        m = (u - mean).reshape(-1, 1)
        S_b += np.dot(m, m.T) * len(x)

    S = np.dot(np.linalg.inv(S_w), S_b)
    eig_val, eig_vec = np.linalg.eig(S)
    eig_val, eig_vec = np.real(eig_val), np.real(eig_vec)
    sorted_index = np.argsort(-eig_val)[:k]
    w = eig_vec[:, sorted_index]
    res = np.dot(data[:, :-1], w)
    
    return res, w


def KNN(data, x): # k=1, data (n_samples, n_dim)
    d = np.linalg.norm(data - x, ord=2, axis=1)
    index = np.argmin(d)
    
    return index


def test(train_data, test_data, k):
    train_data_LDA, w = LDA(train_data, k)
    test_data_LDA = np.dot(test_data[:, :-1], w)
    n, _ = test_data.shape
    index = []
    for i in range(n):
        index.append(KNN(train_data_LDA, test_data_LDA[i, :].reshape(1, -1)))
    knn_label = train_data[index, -1]
    acc = np.sum(knn_label == test_data[:, -1]) / n
    print(acc)
    
    return acc


def main():
    train_data, test_data = load_data()
    LDA_K = [1, 2, 3] # k <= C - 1
    acc = []
    for k in LDA_K:
        acc.append(test(train_data, test_data, k))
        
    print(acc)
    plt.plot(LDA_K, acc)
    plt.xlabel('dimension')
    plt.ylabel('accuracy')
    plt.title('LDA')
    plt.grid()
    plt.show()
    
    
if __name__ == "__main__":
    main()