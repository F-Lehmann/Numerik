from math import sqrt
import numpy as np

def cholesky(A):
    n = len(A)
    L = np.zeros_like(A)

    for i in range(n):
        for k in range(i+1):
            tmp_sum = sum(L[i,j] * L[k,j] for j in range(k))
            
            if (i == k):
                L[i,k] = sqrt(A[i,i] - tmp_sum)
            else:
                L[i,k] = (1.0 / L[k,k] * (A[i,k] - tmp_sum))
    return L
 
A = np.array( [[6, 3, 4, 8], [3, 6, 5, 1], [4, 5, 10, 7], [8, 1, 7, 25]],dtype=np.double)
L = cholesky(A)
R = L.transpose()
print("A:")
print(A)

print("L:")
print(L)

print("R:")
print(R)

print("R* R")
print(R.transpose()@R)