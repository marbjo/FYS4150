import numpy as np
import matplotlib.pyplot as plt
from pandas import *
#import tkinter

np.random.seed(1)
n = int(  input('Give n (Grid point precision, #): ') )
start = 0
end = 1.
h = (end-start)/float(n)
x = np.linspace(start,end,n+1)
f_value = 100.*np.exp(-10*x)

#Setting up matrix
A = np.zeros((n+1,n+1))

#End points set manually
A[0,0] = np.random.randint(1,10)
A[-1,-1] = np.random.randint(1,10)
A[1,0] = np.random.randint(1,10)
A[-2,-1] = np.random.randint(1,10)



for i in range(1,n-1):
    #Diagonal is 2, diagonal above and over is -1
    A[i,i] = np.random.randint(1,10)
    A[i-1,i] = np.random.randint(1,10)
    A[i+1,i] = np.random.randint(1,10)


A = np.c_[A, f_value] #Adding column of function values to A matrix

print(DataFrame(A)) #Pretty print of starting matrix
print('---------------')

#Removing the elements BELOW the pivots
for k in range(1,n):
    A[k,k] = A[k,k] - A[k-1,k]*A[k,k-1]/A[k-1,k-1]
    A[k,-1] = A[k,-1] - A[k-1,-1]*A[k,k-1]/A[k-1,k-1]
    A[k,k-1] = 0.

#Handling end points separately
A[-1,-2] = A[-1,-2] - A[-1,-3]*A[-2,-2]/A[-2,-3]
A[-1,-1] = A[-1,-1] - A[-1,-3]*A[-2,-1]/A[-2,-3]
A[-1,-3] = 0.

#print(DataFrame(A))

#Removing the elements ABOVE the pivots
for l in range(n,0,-1):
    A[l-1,-1] = A[l-1,-1] - A[l,-1]*A[l-1,l]/A[l,l]
    A[l-1,l] = 0.

#Normalizing diagonal
for m in range(n+1):
    A[m,-1] = A[m,-1]/A[m,m]
    A[m,m] = 1.

num_values = h**2 * A[:,-1]

def u(x):
    value = 1 - (1-np.exp(-10))*x -np.exp(-10*x)
    return value

a_values = u(x)

plt.plot(x,num_values,label='Numerical values')
plt.plot(x,a_values, label='Analytical values')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.legend()
plt.show()
