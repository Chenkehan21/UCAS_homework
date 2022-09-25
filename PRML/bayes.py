from re import L
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import pyplot


plt.style.use('seaborn-whitegrid')
palette = pyplot.get_cmap('Set1')
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 18,
}


def generate_samples(mu, sigma, n):
    return np.random.multivariate_normal(mu, sigma, n)


def plot_samples(samples: list, n):
    samples1, samples2 = samples[0], samples[1]
    plt.plot(samples1[0, :], samples1[1, :], '.', alpha=0.5, color='b', label='samples1')
    plt.plot(samples2[0, :], samples2[1, :], '.', alpha=0.5, color='g', label='samples2')
    plt.axvline(x=0, ymin=-5, ymax=5, linestyle='-.', linewidth=3, color='r', alpha=0.6, label='decision boundary')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('2-dimension gaussian distribution samples(n1=n2=%d)'%n)
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.show()


def calculate_experience_error(samples, P_omega1=0.5, P_omega2=0.5):
    samples1, samples2 = samples[0], samples[1]
    num1 = len(samples1.T)
    num2 = len(samples2.T)
    n1, n2 = 0, 0
    for item in samples1[0]:
        if item < 0:
            n1 += 1      
    for item in samples2[0]:
        if item > 0:
            n2 += 1            
    p1_error = n1 / num1
    p2_error = n2 / num2

    res = p1_error * P_omega1 + p2_error * P_omega2
    
    return res


def test(mu1, sigma1, mu2, sigma2, N):
    errors = []
    for n in N:
        samples1 = generate_samples(mu1, sigma1, n).T
        samples2 = generate_samples(mu2, sigma2, n).T
        error = calculate_experience_error([samples1, samples2])
        errors.append(error)

    return np.array(errors)


def main(test_times=10):
    mu1 = np.array([1, 0])
    sigma1 = np.eye(2)
    mu2 = np.array([-1, 0])
    sigma2 = np.eye(2)
    N = list(range(50, 501, 50))
    x = [2 * i for i in N]
    all_errors = []

    for _ in range(test_times):
        error = test(mu1, sigma1, mu2, sigma2, N)
        all_errors.append(error)
    print(all_errors)
    all_errors = np.array(all_errors)
    avg = np.mean(all_errors, axis=0)
    std = np.std(all_errors, axis=0)
    upper_bound = list(map(lambda x: x[0] + x[1], zip(avg, std)))
    lower_bound = list(map(lambda x: x[0] - x[1], zip(avg, std)))
    plt.plot(x, avg, color='b', label='average experience error')
    plt.fill_between(x, upper_bound, lower_bound, color='b', alpha=0.2)
    plt.xticks(x)
    plt.xlabel('n')
    plt.ylabel('Experience Error')
    plt.title('Experience Error with Different Size of Samples (test time=%d)'%test_times)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # mu1 = np.array([1, 0])
    # sigma1 = np.eye(2)
    # mu2 = np.array([-1, 0])
    # sigma2 = np.eye(2)
    # n = 500
    # samples1 = generate_samples(mu1, sigma1, n).T
    # samples2 = generate_samples(mu2, sigma2, n).T
    # plot_samples([samples1, samples2], n=500)
    main(test_times=1)