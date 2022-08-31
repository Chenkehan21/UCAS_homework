import numpy as np
from sympy import solve, symbols


def solve_eqs_numpy():
    'solve Ax = b'
    A = np.matrix('47 28; 89 53')
    b = np.matrix('19, 36').T    
    res = A.I * b

    return res

def solve_eqs_sympy():
    x, y = symbols('x y')
    eq1 = 47 * x + 28 * y - 19
    eq2 = 89 * x + 53 * y - 36
    eqs = [eq1, eq2]
    vars = [x, y]
    res = solve(eqs, vars, set=True)
    
    return res


def solve_eqs(round_ndigits=6):
    A = np.array([[47, 28, 19], [89, 53, 36]])
    eliminated_row2 = A[1] - A[0] * round(A[1][0] / A[0][0], round_ndigits)
    eliminated_A = np.array([A[0], eliminated_row2])
    print("eliminated_A:", eliminated_A)
    
    return None
    
    
if __name__ == "__main__":
    res1 = solve_eqs_sympy()
    res2 = solve_eqs_numpy()
    solve_eqs()
    
    print(res1)
    print(res2)
    # print(res3)