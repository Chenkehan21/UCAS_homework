import numpy as np
import pandas as pd


def get_data(path, w):
    with open(path, 'r', encoding='utf-8') as f:
        dataset = pd.read_csv(f)
    
    w_x = list(dataset['x%d'%w])
    w_y = list(dataset['y%d'%w])
    data = np.array([w_x, w_y]).T
    
    return data


if __name__ == "__main__":
    path = './data.csv'   
    data1, w1, w3 = get_data(path, 1, 3)