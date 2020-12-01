import numpy as np
from numpy.linalg import eig
from math import sqrt

def ComputeSVD(A):
    """Given a matrix A use the eigen value decomposition to compute a SVD
       decomposition. It returns a tuple (U,Sigma,V) of np.matrix objects."""


def PseudoInverse(A):
    """Given a matrix A use the SVD of A to compute the pseudo inverse. It returns the pseudo inverse as a np.matrix object."""
    

def LinearSolve(A, b):
    """Given a matrix A and a vector b this function solves the linear equations
       A*x=b by solving the least squares problem of minimizing |A*x-b| and
       returns the optimal x."""

if (__name__ == "__main__"):
    # Try the SVD decomposition
    A = np.matrix([
        [1.0,1.0,1.0],
        [1.0,2.0,3.0],
        [1.0,4.0,9.0],
        [1.0,8.0,27.0]
    ])
    U, Sigma, V = ComputeSVD(A)
    print U
    print Sigma
    print V
    print("If the following numbers are nearly zero, SVD seems to be working.")
    print(np.linalg.norm(U*Sigma*V.H - A))
    print(np.linalg.norm(U.H*U-np.eye(3)))
    print(np.linalg.norm(V.H*V-np.eye(3)))
    # Try solving a least squares system
    b = np.matrix([1.0,2.0,3.0,4.0]).T
    x = LinearSolve(A,b)
    print("If the following number is nearly zero, linear solving seems to be working.")
    print(np.linalg.norm(x-np.linalg.lstsq(A,b)[0]))

