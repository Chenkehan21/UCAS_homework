import numpy as np
import matplotlib.pyplot as plt
import copy
import math
import pickle


def generate_data( 
            means=[[1., -1.],
                   [5.5, -4.5],
                   [1., 4.],
                   [6., 4.5],
                   [9., 0.]],
            sigma=[[.1, 0.], [0., 1.]]
            ):
    all_data = []
    for mean in means:
        all_data.append(np.random.multivariate_normal(mean, sigma, 200))
    data = np.array(all_data).reshape(-1, 2)
    with open('./all_data.pkl', 'wb') as f:
        pickle.dump(all_data, f)
        
    with open('./data.pkl', 'wb') as f:
        pickle.dump(data, f)
    

def kmeans(k, data, init='random'):
    means_random = []
    if init == 'random':
        index = np.random.choice(data.shape[0], k)
        means_update = data[index]
        print("random init means: ", means_update)
        means_random = data[index]
    else:
        means_update = copy.deepcopy(init)
        print("init means: ", means_update)
        
    while True:
        res = {key:[] for key in list(range(k))}
        query = {tuple(key):None for key in data}
        for d in data:
            distance = np.linalg.norm(means_update - d, ord=2, axis=1)
            c = np.argmin(distance)
            res[c].append(d)
            query[tuple(d)] = c
        means_init = copy.deepcopy(means_update)
        for key, val in res.items():
            val = np.array(val)
            means_update[key] = np.mean(val, axis=0)
        error = np.sum(np.linalg.norm(means_update - means_init, ord=2, axis=1))
        if error <= 1e-5:
            break
    
    return res, means_update, query, means_random


def evaluate(D, P, C, metric='Jaccard'):
    SS, SD, DS, DD = 0, 0, 0, 0
    n = len(D)
    for i in range(n):
        for j in range(i + 1, n):
            di, dj = D[i], D[j]
            pi, pj = P[tuple(di)], P[tuple(dj)]
            ci, cj = C[tuple(di)], C[tuple(dj)]
            if ci == cj and pi == pj:
                SS += 1
            if ci == cj and pi != pj:
                SD += 1
            if ci != cj and pi == pj:
                DS += 1
            if ci != cj and pi != pj:
                DD += 1
    if metric == 'Jaccard':
        return SS / (SS + SD + DS)
    if metric == 'Rand':
        return (SS + DD)/ (SS + SD + DS + DD)
    if metric == 'FM':
        return math.sqrt(SS**2 / ((SS + SD) * (SS + DS)))
    
    
def distance_matrix(X):
    n = X.shape[1]
    G = np.dot(X.T, X)
    H = np.tile(np.diag(G), (n, 1))
    
    return H + H.T - 2 * G


def Hungarian(distance_matrix, index):
    n = len(distance_matrix)
    
    # initialize
    query = {key:[] for key in list(range(n))}
    '''
    index matrix:
    C_a      2      1      0      3      4
    
    C_b      0      3      1      2      4
    
    C_c      4      1      0      2      3
    
    C_d      3      0      4      1      2
    
    C_e      0      1      2      3      4
    
    flag_index = [2, 0, 2, 1, 0]
    
    query = 
    {
        P_a: [C_b, C_e],
        P_b: [],
        P_c: [C_a],
        P_d: [C_d],
        P_e: [C_c]
    }
     =>
    {
        0: [1, 4],
        1: [],
        2: [0],
        3: [3],
        4: [2]
    }
    '''
    for key in query.keys():
        query[key] += [i for i in np.where(index[:, 0] == key)[0]]
    bias = [0] * n
    # update
    while True:
        counter = 0
        for key in query.keys():
            if len(query[key]) > 1:
                d = distance_matrix.T[key][query[key]]
                winner = query[key][np.argmin(d)]
                for candidate in copy.deepcopy(query[key]):
                    if candidate != winner:
                        candidate_index = index[candidate]
                        bias[candidate] += 1
                        new_flag_index = candidate_index[bias[candidate]]
                        query[key].remove(candidate)
                        query[new_flag_index].append(candidate)
            if len(query[key]) == 1:
                counter += 1
        if counter == n:
            return query
        
    
def match_cluster(cluster_means, raw_means):
    n = cluster_means.shape[1]
    all_means = np.hstack([cluster_means, raw_means])
    distance = distance_matrix(all_means)[: n][:, n:] # distance matirx[i,j] = [cluster_mean_i, raw_mean_j]
    index = np.argsort(distance)
    query = Hungarian(distance, index)
    
    return query


def main():
    means=np.array([[1., -1.],
           [5.5, -4.5],
           [1., 4.],
           [6., 4.5],
           [9., 0.]])
    
    with open('all_data.pkl', 'rb') as f:
        all_data = pickle.load(f)
    with open('data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    raw_data_query = {tuple(key):None for key in data}
    for i, key in enumerate(list(raw_data_query.keys())):
        raw_data_query[key] = i // 200
    
    init_means = np.array([all_data[i][10] for i in range(len(all_data))])
    res, cluster_means, cluster_query, means_random = kmeans(5, data, 'random')
    print("cluster mean: ", cluster_means)
    acc = evaluate(data, raw_data_query, cluster_query, 'FM')
    print("acc: ", acc)
    query = match_cluster(cluster_means.T, means.T)
    print(query)
    for key in query.keys():
        error = np.linalg.norm(means[key] - cluster_means[query[key]], ord=2)
        print("error %d: "%key, error)
    
    color_list = ['g', 'b', 'r', 'orange', 'purple']
    fig = plt.figure()
    ax = fig.add_subplot(121)
    color_id = 0
    for item in all_data:
        plt.scatter(item[:, 0], item[:, 1], alpha=0.2, c=color_list[color_id], label='data' + str(color_id))
        color_id += 1
    plt.grid()
    plt.title("Data")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    
    ax = fig.add_subplot(122)
    color_id = 0
    for val in res.values():
        val = np.array(val)
        plt.scatter(val[:, 0], val[:, 1], alpha=0.1, c=color_list[color_id], label='data' + str(color_id))
        color_id += 1
    plt.scatter(cluster_means[:, 0], cluster_means[:, 1], alpha=1, marker='*', c=color_list, s=[100]*5, label='cluster means')
    if len(means_random) > 0:
        plt.scatter(means_random[:, 0], means_random[:, 1], alpha=1, marker='+', c=color_list, s=[100]*5, label='initial means')
    else:
        plt.scatter(init_means[:, 0], init_means[:, 1], alpha=1, marker='+', c=color_list, s=[100]*5, label='initial means')
    plt.title("K-means(K=5)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    main()
    # distance_matrix = np.array([
    # [21, 1, 10, 33, 44],
    # [2, 30, 40, 4, 50],
    # [19, 4, 30, 40, 11],
    # [4, 15, 18, 1, 10],
    # [20, 21, 22, 23, 24]
    # ])
    # index = np.argsort(distance_matrix)
    # print(index)
    # query = Hungarian(distance_matrix, index)