import numpy as np
import scipy.io
from tqdm import tqdm
import matplotlib.pyplot as plt

np.random.seed(1234)


def load_data(path='./ORLData_25.mat'):
    all_data = scipy.io.loadmat(path)['ORLData'].T
    np.random.shuffle(all_data)
    all_data = all_data.T
    _, n = all_data.shape
    split = int(0.8 * n)
    train_data = all_data[:, : split]
    test_data = all_data[:, split:]
    
    return train_data, test_data


def PCA(data: np.array, k):
    mean = np.mean(data, axis=0)
    data = data.astype(np.float64)
    data -= mean
    cov = np.cov(data)
    eigen_val, eigen_vec = np.linalg.eig(cov)
    eigen_val, eigen_vec = np.real(eigen_val), np.real(eigen_vec)
    sorted_index = np.argsort(-eigen_val)[:k]
    w = eigen_vec[:, sorted_index]
    res = np.dot(w.T, data)
    
    return res, w


def KNN(data, x): # k=1
    d = np.linalg.norm(data - x, ord=2, axis=0)
    index = np.argmin(d)
    
    return index


def test(train_data, test_data, pca_dim = 15):
    train_data_pca, w = PCA(train_data[:-1, :], pca_dim)
    test_data_pca = np.dot(w.T, test_data[:-1, :])
    _, n = test_data.shape
    index = []
    for i in range(n):
        index.append(KNN(train_data_pca, test_data_pca[:, i].reshape(pca_dim, 1)))
    knn_label = train_data[-1, index]
    acc = np.sum(knn_label == test_data[-1, :]) / n
    
    return acc


def main():
    train_data, test_data = load_data()
    pca_dims = list(range(5, 105, 5))
    acc = []
    for k in tqdm(pca_dims):
        acc.append(test(train_data, test_data, k))
    test(train_data, test_data, 10)
        
    print(acc)
    plt.plot(pca_dims, acc)
    plt.xticks(pca_dims)
    plt.grid()
    plt.xlabel("dimension")
    plt.ylabel("accuracy")
    plt.title("PCA")
    plt.show()


if __name__ == "__main__":
    main()