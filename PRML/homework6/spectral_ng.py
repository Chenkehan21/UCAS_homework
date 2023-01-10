import numpy as np
import matplotlib.pyplot as plt
from kmeans import evaluate
import pickle
from sklearn.cluster import KMeans


def knn(data, x, k):
    d = np.linalg.norm(data - x, ord=2, axis=1)
    neighbour_id = d.argsort()[1 : k + 1]
    neighbour = data[neighbour_id]
    
    return neighbour_id, neighbour


def Laplacian(data, k, sigma):
    n = len(data)
    w = np.zeros((n, n))
    gaussian = lambda x, y: np.exp(-np.linalg.norm(x - y, ord=2)**2 / (2 * sigma**2))
    for i in range(n):
        x = data[i]
        neighbour_id, neighbour = knn(data, x, k)
        for j, y in zip(neighbour_id, neighbour):
            w[i, j] = gaussian(x, y)
    W = (w + w.T) / 2
    # np.savetxt('W.out', W, delimiter=',')
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    # np.savetxt('L5.out', L, delimiter=',')
    D_sqrt = np.diag(np.sum(W, axis=1)**(-1/2))
    L_sym = np.dot(D_sqrt, np.dot(L, D_sqrt))
    # L_sym = np.eye(*W.shape) - np.dot(D_sqrt, np.dot(W, D_sqrt))
    # np.savetxt('L_sym5.out', L_sym, delimiter=',')
    
    return L_sym


def normalize(x):
    partition_function = np.sqrt(1e-5 + np.sum(x**2, axis=1, keepdims=True))
    return x / partition_function


def Ng(data, k_graph, sigma):
    L_sym = Laplacian(data, k_graph, sigma)
    eig_val, eig_vec = np.linalg.eig(L_sym)
    eig_val, eig_vec = np.real(eig_val), np.real(eig_vec)
    sorted_index = np.argsort(eig_val)[:k_graph]
    U = eig_vec[:, sorted_index]
    U = normalize(U)
    kmeans = KMeans(n_clusters=2, n_init=20, max_iter=100).fit(U)
    labels = kmeans.labels_
    res = {0:[], 1:[]}
    for i, label in enumerate(labels):
        res[label].append(data[i])
    
    res_eval = evaluate_cluster(data, labels)
        
    color_list = ['g', 'b']
    color_id = 0
    for val in res.values():
        val = np.array(val)
        plt.scatter(val[:, 0], val[:, 1], alpha=0.8, c=color_list[color_id], label='data' + str(color_id))
        color_id += 1
    plt.legend()
    plt.grid()
    plt.show()
    
    return res_eval
    
    
def evaluate_cluster(data, label):
    raw_data_query = {tuple(key):None for key in data}
    cluster_query = {tuple(key):None for key in data}
    for i, key in enumerate(list(raw_data_query.keys())):
        raw_data_query[key] = i // 100
        cluster_query[key] = label[i]
    res = evaluate(data, raw_data_query, cluster_query, 'FM')
    
    return res


def check_different_sigma():
    sigma_1, sigma_16, sigma_31 = [], [], []
    for k in range(1, 30):
        res1 = Ng(data, k, 1)
        res16 = Ng(data, k, 16)
        res31 = Ng(data, k, 31)
        sigma_1.append(res1)
        sigma_16.append(res16)
        sigma_31.append(res31)
        
    plt.plot(sigma_1, label='sigma=1')
    plt.plot(sigma_16, label='sigma=16')
    plt.plot(sigma_31, label='sigma=31')
    plt.legend()
    plt.grid()
    plt.xlabel('k')
    plt.ylabel('Jaccard')
    plt.title('Jaccard index with different k')
    
    plt.show()
    
    k1, k5, k10 = [], [], []
    for sigma in range(1, 30):
        res1 = Ng(data, 1, sigma)
        res5 = Ng(data, 5, sigma)
        res10 = Ng(data, 15, sigma)
        k1.append(res1)
        k5.append(res5)
        k10.append(res10)
    
    plt.plot(k1, label='k=1')
    plt.plot(k5, label='k=5')
    plt.plot(k10, label='k=15')
    plt.legend()
    plt.grid()
    plt.xlabel('sigma')
    plt.ylabel('Jaccard')
    plt.title('Jaccard index with different sigma')
    
    plt.show()


def check_different_k():
    k1, k5, k10 = [], [], []
    for sigma in range(1, 30):
        res1 = Ng(data, 1, sigma)
        res5 = Ng(data, 5, sigma)
        res10 = Ng(data, 15, sigma)
        k1.append(res1)
        k5.append(res5)
        k10.append(res10)
    
    plt.plot(k1, label='k=1')
    plt.plot(k5, label='k=5')
    plt.plot(k10, label='k=15')
    plt.legend()
    plt.grid()
    plt.xlabel('sigma')
    plt.ylabel('Jaccard')
    plt.title('Jaccard index with different sigma')
    
    plt.show()
    

if __name__ == "__main__":
    with open('./spectral_cluster_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    # check_different_k()
    
    eval = Ng(data, 17, 1)
    print(eval)