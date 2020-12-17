import numpy as np
from numpy import linalg as la
from typing import Union

def HouseholderQR(x: np.ndarray) -> Union[np.ndarray, int]:
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy()

    for j in range(n):
        x = R[j:, j]
        normx = np.linalg.norm(x)
        rho = -np.sign(x[0])
        u1 = x[0] - rho * normx
        u = x / u1
        u[0] = 1
        beta = -rho * u1 / normx

        R[j:, :] = R[j:, :] - beta * np.outer(u, u).dot(R[j:, :])
        Q[:, j:] = Q[:, j:] - beta * Q[:, j:].dot(np.outer(u, u))
        
    return Q, R


if(__name__=="__main__"):
    A=np.random.rand(4,3)
    Q,R=HouseholderQR(A)
    print("The following matrix should be upper triangular:")
    print(R)
    print("If the solution consitutes a decomposition, the following is near zero:")
    print(la.norm(A-np.dot(Q,R)))
    print("If Q is unitary, the following is near zero:")
    print(la.norm(np.dot(Q.T,Q)-np.eye(Q.shape[0])))
