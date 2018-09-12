import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lu_factor, lu_solve
import time

n = int(  input('Give n (Grid point precision, #): ') )
start = 0.
end = 1.
h = (end-start)/float(n+1)
x = np.linspace(start,end,n+2)
f_value = 100.*np.exp(-10*x)

#Boundary conditions
f_value[-1]=0; f_value[0] = 0.
#Setting up matrix
A = np.zeros((n,n))

#End points set manually
A[0,0] = 2
A[-1,-1] = 2
A[1,0] = -1
A[-2,-1] = -1

for i in range(1, n-1):
    #Diagonal is 2, diagonal above and over is -1
    A[i,i] = 2
    A[i-1,i] = -1
    A[i+1,i] = -1

start_time = time.time()

#LU decomp and solver
lu, piv = lu_factor(A)
x_sol = lu_solve((lu, piv), f_value[1:-1])

end_time = time.time()
time_spent = end_time - start_time

#Making the final plotting vectors
f_value[1:-1] = x_sol
num_values = h**2 *f_value

def u(x):
    value = 1 - (1-np.exp(-10))*x -np.exp(-10*x)
    return value

a_values = u(x) #Analytic values
"""
plt.scatter(x,num_values,color='red',s=5,marker='s',label='Numerical values')
plt.plot(x,a_values,'b-' ,label='Analytical values')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.legend()
plt.show()
"""

print('Time elapsed on algorithm: %s ' %time_spent)
