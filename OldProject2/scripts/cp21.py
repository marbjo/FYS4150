import numpy as np
import matplotlib as plt

n = np.int( input('What is n: ') )
A = np.zeros((n,n))

rho_0 = 0
rho_N = 1


h = (rho_N - rho_0 ) / n

for i in range(n):

    d = 2/h**2
    a = -1/h**2

    A[i,i] = d

    if i<(n-1):
        A[i,i+1] = a
        A[i+1,i] = a


# diagonalize and obtain eigenvalues, not necessarily sorted
EigValues, EigVectors = np.linalg.eig(A)
# sort eigenvectors and eigenvalues
permute = EigValues.argsort()
EigValues = EigValues[permute]
EigVectors = EigVectors[:,permute]

print(EigValues)
