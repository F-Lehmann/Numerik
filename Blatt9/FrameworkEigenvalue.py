import scipy.io
import numpy as np
from matplotlib import pyplot

iterations = 100

def PowerIteration(A, v):
    """Given a square matrix A and a column vector v of matching size to initialize
       the iteration this function implements the power iteration to approximate the
       largest Eigenvalue of A. It returns a list of successive approximations where
       the last approximation is the best one."""
    approx = []
    vector = v
    Av = A.dot(vector)
    for _ in range(iterations):
        vector = Av / np.linalg.norm(Av)
        Av = A.dot(vector)
        val = vector.flatten().dot(Av)
        approx.append(val)
    return approx


def RayleighQuotientIteration(A, v):
    """Given a square matrix A and a column vector v of matching size to initialize
       the iteration this function implements the Rayleigh quotient iteration to
       approximate the Eigenvalue of A whose Eigenvector is closest to v. It returns
       a list of successive approximations where the last approximation is the best
       one."""
    approx = []
    vector = v
    I = np.eye(A.shape[0])
    for _ in range(iterations):
        u = vector/np.linalg.norm(vector)
        val = np.dot(u.flatten(), np.dot(A, u))
        approx.append(val)
        vector = np.linalg.solve(A-val*I, u)
    u = vector/np.linalg.norm(vector)
    val = np.dot(u.flatten(), np.dot(A, u))
    approx.append(val)
    return approx


Matrizen = scipy.io.loadmat("Matrizen.mat")
A1 = Matrizen["A1"]
A2 = Matrizen["A2"]
Startvektor = Matrizen["Startvektor"]

pyplot.suptitle('Power Iteration')
LambdaList = PowerIteration(A1, Startvektor)
pyplot.plot(LambdaList, color="g")
LambdaList = PowerIteration(A2, Startvektor)
pyplot.plot(LambdaList, color="r")

pyplot.figure()
pyplot.suptitle("Rayleigh Quotient Iteration")
LambdaList = RayleighQuotientIteration(A1, Startvektor)
pyplot.plot(LambdaList, "g")
LambdaList = RayleighQuotientIteration(A2, Startvektor)
pyplot.plot(LambdaList, "r")

pyplot.show()
