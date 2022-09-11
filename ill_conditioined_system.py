import numpy as np


def calculate_rank(matrix: np.array, n_precision):
    rank = None
    multiplier = round(matrix[1][0] / matrix[0][0], n_precision)
    E = np.array([matrix[0], np.around(matrix[1] - multiplier * matrix[0], decimals=n_precision)])
    if np.all(E[1] == 0):
        rank = 1
    else:
        rank = 2
    
    return rank, E


def check_solvable(matrix: np.array, n_precision: int):
    r, E = calculate_rank(matrix, n_precision)
    n_row, n_column = matrix.shape
    n_column -= 1
    
    if r == n_row and r == n_column:
        print("one solution")
        return True
    elif r == n_row and r < n_column:
        print("infinity solutions")
        return True
    elif r == n_column and r < n_row:
        print("one solution or no solution")
    elif r < n_row and r < n_column:
        print("infinity solutions or no solution")
    
    
def run(n_precision=5):
    '''
    equations:
    0.835x + 0.667y = 0.168
    0.333x + 0.266y = 0.067
    '''
    print("precision = ", n_precision)
    A = np.array([[0.835, 0.667], [0.333, 0.266]])
    B = np.array([0.168, 0.067]).reshape(2, 1)
    aug_A = np.append(A, B, axis=1)
    print("coefficient matrix:\n", A)
    print("augmented matrix:\n", aug_A)
    
    rank_A, E_A = calculate_rank(A, n_precision)
    rank_aug_A, E_aug_A = calculate_rank(aug_A, n_precision)
    print("r(A) = ", rank_A)
    print("r([A|b]) = ", rank_aug_A)
    print("row echelon form of cofficient matrix A:\n", E_A)
    print("row echelon form of augmented matrix [A|b]:\n", E_aug_A)
    
    check_solvable(aug_A, n_precision)
    

def main():
    run(5)
    print()
    run(6)
    

if __name__ == "__main__":
    main()