from turtle import end_fill
import numpy as np
from batch_perception import get_data


def calculate_pesudo_inv(x, epsilon):
    tmp = np.dot(x.T, x)
    tmp2 = tmp + epsilon * np.eye(tmp.shape)
    res = np.dot(np.linalg.inv(tmp2), x.T)
    
    return res


def hk(y, a=np.array([[0., 0., 0.]]), b=np.array([[1., 1., 1.]]), 
                     eta=1., k=0, b_min=0.1, k_max=2):
    n = len(y.T)
    iteration_times = 0
    
    while True:
        k = (k + 1) % n
        if k == k_max:
            print("no solution found")
        else:
            e = np.dot(y.T, a) - b.T
            e_t = 0.5 * (e + abs(e))
            b += 2 * eta * e_t
            y_pesudo_inv = calculate_pesudo_inv(y)
            a = np.dot(y_pesudo_inv.T, b)
            iteration_times += 1
            if abs(e) <= b_min:
                return a, b, iteration_times
    

if __name__ == "__main__":
    path = 'D:\\UCAS_homework\\PRML\\homework3\\data.csv'   
    data1 = get_data(path, 1, 3)
    a, b, iteration_times = hk(data1, a=np.array([[0., 0., 0.]]), b=np.array([[1., 1., 1.]]), 
                               eta=1., k=0, b_min=0.1, k_max=18)