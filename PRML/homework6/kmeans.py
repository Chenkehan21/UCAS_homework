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


def main():
    means=[[1., -1.],
           [5.5, -4.5],
           [1., 4.],
           [6., 4.5],
           [9., 0.]]
    
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
    print("error: ", np.linalg.norm(cluster_means - means, ord=2, axis=1))
    acc = evaluate(data, raw_data_query, cluster_query)
    print("acc: ", acc)
    
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