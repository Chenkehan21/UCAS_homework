from tabnanny import check
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


def solve_eqs(round_ndigits=3):
    A = np.array([[47, 28, 19], [89, 53, 36]])
    eliminated_row2 = A[1] - A[0] * round(A[1][0] / A[0][0], round_ndigits)
    coff_y = eliminated_row2[1]
    b_prime = eliminated_row2[2]
    y = round(b_prime / coff_y, round_ndigits)
    x = round((A[0][2] - A[0][1] * y) / A[0][0], round_ndigits)
    res = [x, y]
    
    return res


def check_precision(threshold=1e-50):
    n = 1
    x, y = solve_eqs(round_ndigits=n)
    while abs(x - 1.0) > threshold or abs(y + 1.0) > threshold:
        n += 1
        x, y = solve_eqs(n)
    print("final n:", n)
    print(x, y)
    
    
if __name__ == "__main__":
    res1 = solve_eqs_sympy()
    res2 = solve_eqs_numpy()
    res3 = solve_eqs(1)
    
    print(res1)
    print(res2)
    print(res3)

    check_precision()