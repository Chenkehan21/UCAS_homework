import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize():
    class1 = np.array([
        [ 1.58, 2.32, -5.8], [ 0.67, 1.58, -4.78], [ 1.04, 1.01, -3.63], 
        [-1.49, 2.18, -3.39], [-0.41, 1.21, -4.73], [1.39, 3.16, 2.87],
        [ 1.20, 1.40, -1.89], [-0.92, 1.44, -3.22], [ 0.45, 1.33, -4.38],
        [-0.76, 0.84, -1.96]
    ])

    class2 = np.array([
        [ 0.21, 0.03, -2.21], [ 0.37, 0.28, -1.8], [ 0.18, 1.22, 0.16], 
        [-0.24, 0.93, -1.01], [-1.18, 0.39, -0.39], [0.74, 0.96, -1.16],
        [-0.38, 1.94, -0.48], [0.02, 0.72, -0.17], [ 0.44, 1.31, -0.14],
        [ 0.46, 1.49, 0.68]
    ])

    class3 = np.array([
        [-1.54, 1.17, 0.64], [5.41, 3.45, -1.33], [ 1.55, 0.99, 2.69], 
        [1.86, 3.19, 1.51], [1.68, 1.79, -0.87], [3.51, -0.22, -1.39],
        [1.40, -0.44, -0.92], [0.44, 0.83, 1.97], [ 0.25, 0.68, -0.99],
        [ 0.66, -0.45, 0.08]
    ])

    x1 = class1[:, 0]
    y1 = class1[:, 1]
    z1 = class1[:, 2]

    x2 = class2[:, 0]
    y2 = class2[:, 1]
    z2 = class2[:, 2]

    x3 = class3[:, 0]
    y3 = class3[:, 1]
    z3 = class3[:, 2]

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x1, y1, z1, label='data1')
    ax.scatter(x2, y2, z2, label='data2')
    ax.scatter(x3, y3, z3, label='data3')

    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    plt.legend()
    plt.show()
    
    
if __name__ == "__main__":
    visualize()