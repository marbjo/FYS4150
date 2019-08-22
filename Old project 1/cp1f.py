import numpy as np
import matplotlib.pyplot as plt
from pandas import *

np.random.seed(1)
n = int(  input('Give n (Grid point precision, #): ') )
start = 0.
end = 1.
h = (end-start)/float(n)
x = np.linspace(start,end,n+2)
print(len(x))
print(x)
print(h)

f_value = 100.*np.exp(-10.*x)

f_value[0]=0; f_value[-1] = 0


diag = 2*np.ones(n) #np.random.randint(1,5,n)
u_diag = -np.ones(n) #np.random.randint(1,5,n-1)
l_diag = -np.ones(n) #np.random.randint(1,5,n-1)

l_diag[0] = 0; u_diag[-1] = 0

"""
print('Diagonal: ')
print(diag)
print('Lower diagonal: ')
print(l_diag)
print('Upper diagonal: ')
print(u_diag)
print('Function values: ')
print(f_value)
"""

#Forward substitution
for i in range(1,n):
    print('Round %d: ' %i)
    diag[i] = diag[i] - l_diag[i]*u_diag[i-1]/diag[i-1]
    f_value[i] = f_value[i] - l_diag[i]*f_value[i-1]/u_diag[i-1]

#print('Function values: ')
#print(f_value)
u = f_value

#Backward substitution
for j in range(n,0,-1):
    print('Round %d: ' %j)
    u[j-1] = (f_value[j-1] - u_diag[j-1]*u[j] ) / diag[j-1]

#u = f_value


print('Diagonal: ')
print(diag)
print('Lower diagonal: ')
print(l_diag)
print('Upper diagonal: ')
print(u_diag)
print('Function values: ')
print(f_value)


"""
for k in range(1,n+1):
    #Normalizing diagonal
    u[k] = u[k]/diag[k-1]
    diag[k-1] = 1.
"""
"""
print('Diagonal: ')
print(diag)
print('Lower diagonal: ')
print(l_diag)
print('Upper diagonal: ')
print(u_diag)
print('Function values: ')
print(f_value)
"""

def u_a(y):
    #Analytic solution
    value = 1. - (1.-np.exp(-10))*y -np.exp(-10.*y)
    return value

a_values = u_a(x)
num_values = u*h**2

plt.figure()
plt.plot(x,num_values,label='Numerical values')
#plt.scatter(x,num_values,s=4,color='red',marker='s',label='Numerical values')
plt.plot(x,a_values, label='Analytical values')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.legend()
plt.show()
