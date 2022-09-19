import numpy as np


def corr1d(x, k):
    padding = np.array([0] * int((len(k) - 1) / 2))
    padded_x = np.hstack([padding, x, padding])
    k_size = len(k)
    res = []
    for i in range(len(padded_x) - k_size + 1):
        res.append((padded_x[i : i + k_size] * k).sum())
    
    return res


def corr2d(x, k):
    h, w = x.shape
    m, n = k.shape
    h_padding = int((m - 1) / 2)
    w_padding = int((n - 1) / 2)
    padded_x = np.pad(x, ((h_padding, h_padding), (w_padding, w_padding)), 'constant', constant_values=(0, 0))
    print("\npadded_x:\n", padded_x)
    h_y, w_y = h - m + 1 + 2 * h_padding, w - n + 1 + 2 * w_padding
    h_x, w_x = padded_x.shape
    res = []
    for i in range(h_x - m + 1):
        for j in range(w_x - n + 1):
            res.append((padded_x[i : i + m, j : j + n] * k).sum())
    res = np.array(res).reshape(h_y, w_y)

    return res


if __name__ == "__main__":
    x1 = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])
    k1 = np.array([2, 0, -2])
    conv_k1 = np.flip(k1, axis=0)
    res1 = corr1d(x1, conv_k1)
    print("\nres1:\n", res1)

    x2 = np.array([[1, 3, 2, 0, 4],
                  [1, 0, 3, 2, 3],
                  [0, 4, 1, 0, 5],
                  [2, 3, 2, 1, 4],
                  [3, 1, 0, 4, 2]])
    k2 = np.array([[-1, 0, 1],
                  [-2, 0, 2],
                  [-1, 0, 1]])
    conv_k2 = np.flip(np.flip(k2, axis=0), axis=1)
    print("\nconv k:\n", conv_k2)
    res2 = corr2d(x2, conv_k2)
    print("\nres2:\n", res2)