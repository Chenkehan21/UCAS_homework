import numpy as np
import math


def get_matrix():
    n = eval(input("Please input the size of the square matirx A:"))
    print("Please input numbers seperated by space and input one row of the matrix at one time.")
    A = np.zeros((n, n))
    for i in range(n):
        a_i = input().split(' ')
        A_i = [eval(item) for item in a_i]
        A[i] = A_i
        
    b = input("Please input b seperated by space, it should be the same size of A!\n").split(' ')
    b = np.array([eval(item) for item in b]).reshape(n, 1)
    
    return A, b


class Matrix():
    def __init__(self, A:np.ndarray, b:np.ndarray):
        self.A = A.astype(np.float64)
        self.b = b.astype(np.float64)
        self.n = A.shape[0]
    
    def calculate_2norm(self, x):
            res = 0
            for item in x:
                res += item**2
            
            return math.sqrt(res)
        
    def LU_decomposition(self):
        L = np.eye(self.n)
        A = self.A.copy()
        row = 0
        while row < self.n - 1:
            if row != 0:
                pivot_pos = (A[row] != 0).argmax(axis=0)
                pivot = A[row, pivot_pos]
            else:
                pivot = A[0, 0]
            if pivot == 0:
                print("No LU_decomposition!")
                return None, None
            for i in range(self.n - 1 , row, -1):
                l = A[i][row] / pivot
                A[i] -= l * A[row]
                L[i][row] = l
            row += 1
        
        return L, A
    
    def QR_decomposition(self):
        Q = np.zeros((self.n, self.n))
        R = np.zeros((self.n, self.n))
        cnt = 0 # set a counter to count how many vectors have been done.
        A = self.A.copy()
        x1 = A[:,0]
        # r1 = np.linalg.norm(x1, ord=2)
        r1 = self.calculate_2norm(x1)
        R[0, 0] = r1
        q1 = x1 / r1
        Q[:, 0] = q1
        cnt += 1
        
        for col in range(1, self.n):
            x = A[:, col] # x.shape = (self.n,)
            for i in range(cnt):
                r = np.dot(x.T, Q[:, i])
                R[i, col] = r
                x -= r * Q[:, i]
            r = self.calculate_2norm(x)
            R[col, col] = r
            q = x / r
            Q[:, col] = q
            cnt += 1
        
        return Q, R
        
    def Householder_reduction(self):
        R = np.eye(self.n)
        tmp = self.A.copy()
        for _ in range(self.n - 1):
            x = tmp[:, 0]
            x_2norm = self.calculate_2norm(x)
            I = np.eye(x.shape[0])
            e1 = I[:, 0]
            u = x - x_2norm * e1
            u = u.reshape(u.shape[0], -1)
            R_i_hat = I - 2 * (np.dot(u, u.T)) / (np.dot(u.T, u))
            R_i = np.eye(self.n)
            R_i[self.n - R_i_hat.shape[0]:, self.n - R_i_hat.shape[1]:] = R_i_hat
            R = np.dot(R_i, R)
            T = np.dot(R, self.A)
            tmp = np.dot(R_i_hat, tmp)
            tmp = np.delete(tmp, 0, axis=0)
            tmp = np.delete(tmp, 0, axis=1)

        Q = R.T
        R = T
        
        return Q, R
    
    def Givens_reduction(self):
        tmp = self.A.copy()
        
        P = np.eye(self.n)
        for _ in range(self.n - 1):
            x = tmp[:, 0]
            for i in range(1, x.shape[0]):
                d = math.sqrt(x[0]**2 + x[i]**2)
                c = x[0] / d
                s = x[i] / d
                P_i_hat = np.eye(x.shape[0])
                P_i_hat[0, 0] = c
                P_i_hat[i, 0] = -s
                P_i_hat[0, i] = s
                P_i_hat[i, i] = c
                P_i = np.eye(self.n)
                P_i[self.n - x.shape[0]:, self.n - x.shape[0]:] = P_i_hat
                P = np.dot(P_i, P)
                T = np.dot(P, self.A)
                tmp = np.dot(P_i_hat, tmp)
                x = tmp[:, 0]
            tmp = np.delete(tmp, 0, axis=0)
            tmp = np.delete(tmp, 0, axis=1)
            
        Q = P.T
        R = T
        
        return Q, R
            
    def solve_equation(self, Q, R):
        # Ax = b => QRx = b => Rx = (Q.T)b
        b = np.dot(Q.T, self.b)
        res = []
        res.append(b[-1] / R[-1, -1])
        for i in range(self.n - 2, -1, -1):
            tmp = b[i]
            for j, x in enumerate(res):
                tmp -= x * R[i, self.n - 1- j]
            res.append(tmp / R[i, i])
        res = [x.item() for x in res]
        res.reverse()
        
        return res
            
    def calculate_determinant(self, x):
        if x.shape == (2, 2):
            return x[0, 0] * x[1, 1] - x[0, 1] * x[1, 0]
        else:
            res = 0
            for i in range(x.shape[0]):
                tmp = x.copy()
                alpha = (-1)**i * x[i, 0]
                tmp = np.delete(tmp, i, axis=0)
                tmp = np.delete(tmp, 0, axis=1)
                res += alpha * self.calculate_determinant(tmp)

            return res

def main():
    A, b = get_matrix()
    matrix = Matrix(A, b)
    print("Please choose the decopmosition you need. You can choose more than one, please seperate them by space:\n")
    print("1:LU_decomposition\n2:QR_decomposition\n3:Householder_reduction\n4:Givens_reduction\n")
    decomposition_id = input().split(' ')
    decomposition_id = [eval(item) for item in decomposition_id]
    
    print("A:\n", A, "\nb:\n", b , '\n')
    for i in decomposition_id:
        if i == 1:
            print("\nLU decomposition:\n")
            L, U = matrix.LU_decomposition()
            print("L:\n", L, "\nU:\n", U)
        if i == 2:
            print("\nQR decomposition:\n")
            Q, R = matrix.QR_decomposition()
            print("Q:\n", Q, "\nR:\n", R)
        if i == 3:
            print("\nHouseholder reduction:\n")
            Q, R = matrix.Householder_reduction()
            print("Q:\n", Q, "\nR:\n", R)
        if i == 4:
            print("\nGivens reduction:\n")
            Q, R = matrix.Givens_reduction()
            print("Q:\n", Q, "\nR:\n", R)
    
    x = matrix.solve_equation(Q, R)
    print("Solution of Ax=b is: ", x)
    b_hat = np.dot(A, np.array(x).reshape(3, -1))
    print("check Ax:\n", b_hat)
    
    determinant = matrix.calculate_determinant(A)
    print("det(A): ", determinant)
    print("check det(A): ", np.linalg.det(A))
        

if __name__ == "__main__":
    main()