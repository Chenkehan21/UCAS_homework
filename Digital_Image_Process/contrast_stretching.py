import numpy as np
import matplotlib.pyplot as plt


def contrast_stretching_function(E, m, r):
    return 1 / (1 + np.exp(-E * r + E * m))


def contrast_stretching_function2(E, m, r):
    return 1 / (1 + (m / r)**E)


def safe_contrast_stretching_function(E, m, r):
    x = E * (r - m)
    x_neg = x[x < 0]
    x_else = x[x >= 0]
    res_else = 1 / (1 + np.exp( -1 * x_else))
    res_neg = (np.exp(x_neg)) / (1 + np.exp(x_neg))
    res = np.append(res_neg, res_else)

    return res


def plot_function(E, L, m):
    epslion = 1
    r = np.arange(epslion, L, 0.1)
    s = safe_contrast_stretching_function(E, m, r)
    plt.plot(r, s, label="E=%.3f, L=%d, m=%d" % (E, L, m))
    plt.title('sigmoid function family')
    plt.xlabel('r', fontsize=15)
    plt.ylabel('s', fontsize=15)
    plt.grid(True)
    plt.legend()


def plot_function2(E, L, m):
    epslion = 1
    r = np.arange(epslion, L, 0.1)
    s = contrast_stretching_function2(E, m, r)
    plt.plot(r, s, label="E=%d, L=%d, m=%d" % (E, L, m))
    plt.title('teacher\'s ppt function family')
    plt.xlabel('r', fontsize=15)
    plt.ylabel('s', fontsize=15)
    plt.grid(True)
    plt.legend()
    

def plot_function_family(L, m):
    fig = plt.figure()
    plt.suptitle("Contrast Stretching Function Family(with different E)")
    
    # plt.xticks(range(0, L, 1))
    ax = fig.add_subplot(1, 2, 1)
    for E in np.arange(0.020, 0.13, 0.01):
        plot_function(E, L, m)
    ax = fig.add_subplot(1, 2, 2)
    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')
    for E in range(2, 22, 2):
        plot_function2(E, L, m)
    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')
    
    plt.show()


if __name__ == "__main__":
    plot_function_family(L=256, m=128)