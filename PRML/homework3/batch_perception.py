import numpy as np
import pandas as pd


def get_data(path, w1, w2):
    with open(path, 'r', encoding='utf-8') as f:
        dataset = pd.read_csv(f)
    
    w1_x = list(dataset['x%d'%w1])
    w1_y = list(dataset['y%d'%w1])
    data1 = np.array([w1_x, w1_y])
    
    w2_x = list(dataset['x%d'%w2])
    w2_y = list(dataset['y%d'%w2])
    data2 = np.array([w2_x, w2_y])
    
    #normalized augmented feature vector
    data = np.concatenate([data1, -data2], axis=1)
    data = np.concatenate([data, np.array([[1]* 2 * len(data1.T)])], axis=0)
    
    return data    


def calculate_Y_k(a, y):
    res = np.dot(a, y)
    index = np.where(res <= 0)
    Y_k = np.array([y[:, i] for i in index[1]]).T
    
    return Y_k 


def do_batch_perception(data, a=np.array([[0., 0., 0.]]), eta=1, k=0):
    # n = len(data.T)
    iteration_times = 0
    
    while True:
        # k = (k + 1) % n
        Y_k = calculate_Y_k(a, data)
        if len(Y_k) == 0:
            break
        gradient = np.sum(Y_k, axis=1)
        a += eta * gradient
        iteration_times += 1
        
    return a, iteration_times

if __name__ == "__main__":
    path = 'D:\\UCAS_homework\\PRML\\homework3\\data.csv'   
    data1 = get_data(path, 1, 2)
    a1, iteration_times1 = do_batch_perception(data1, a=np.array([[0., 0., 0.]]), eta=1, k=0)
    print(a1, iteration_times1)
    
    data2 = get_data(path, 3, 2)
    a2, iteration_times2 = do_batch_perception(data2, a=np.array([[0., 0., 0.]]), eta=1, k=0)
    print(a2, iteration_times2)