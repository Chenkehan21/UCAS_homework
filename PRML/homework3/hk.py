import numpy as np
from batch_perception import get_data, plot_result

np.random.seed(0)


def calculate_pesudo_inv(x):
    tmp = np.dot(x.T, x)
    res = np.dot(np.linalg.inv(tmp), x.T)
    
    return res


def hk(y, eta=0.1, b_min=1e-8, k_max=1e3):
    y = y.T #(20, 3)
    iteration_times = 0
    y_pesudo_inv = calculate_pesudo_inv(y)
    b = np.ones((20, 1)) * 1e-2
    a = np.dot(y_pesudo_inv, np.ones((20, 1)))
    
    while True:
        if iteration_times == k_max:
            print("no solution found")
            return a, b, iteration_times
        else:
            e = np.dot(y, a) - b
            e_t = 0.5 * (e + np.abs(e))
            b += 2 * eta * e_t
            a = np.dot(y_pesudo_inv, b)
            iteration_times += 1
            if (np.abs(e) <= b_min).all():
                return a, b, iteration_times
    

if __name__ == "__main__":
    path = './data.csv'   
    data1, w1, w3 = get_data(path, 1, 3)
    a1, b1, iteration_times1 = hk(data1, eta=0.1, b_min=1e-10, k_max=1e5)
    print(a1, '\n\n', b1, '\n\n', iteration_times1)

    data2, w2, w4 = get_data(path, 2, 4)
    a2, b2, iteration_times2 = hk(data2, eta=0.1, b_min=1e-8, k_max=1e5)
    print(a2, '\n\n', b2, '\n\n', iteration_times2)
    
    plot_result(a1, w1, w3, 'w1', 'w3', 'Ho-Kashyap algorithm')
    plot_result(a2, w2, w4, 'w2', 'w4', 'Ho-Kashyap algorithm')