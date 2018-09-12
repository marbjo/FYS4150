import numpy as np
import matplotlib.pyplot as plt
from pandas import *
import time

#np.random.seed(1)
n = int(  input('Give n (Grid point precision, #): ') )
start = 0.
end = 1.
h = (end-start)/float(n+1)
x = np.linspace(start,end,n+2)

#Setting function values and endpoints
f_value = 100.*np.exp(-10.*x)
f_value[0]=0; f_value[-1] = 0

#Diagonal vectors. These can be random numbers
#Note: no error handling for linearly dependent column vectors
#(Leads to division by zero in loops)

diag = 2*np.ones(n) #np.random.randint(1,10,n)
u_diag = -np.ones(n-1) #np.random.randint(1,10,n-1)
l_diag = -np.ones(n-1) #np.random.randint(1,10,n-1)


start_time = time.time()


#Forward substitution
for i in range(0,n-1):
    diag[i+1] = diag[i+1] - l_diag[i]*u_diag[i]/diag[i]
    f_value[i+2] = f_value[i+2] - l_diag[i]*f_value[i+1]/diag[i]

#Backward substitution
for j in range(n,1,-1):
    f_value[j-1] = f_value[j-1] - f_value[j]*u_diag[j-2]/diag[j-1]

u = f_value

#Normalizing diagonal
for k in range(1,n+1):
    u[k] = u[k]/diag[k-1]
    diag[k-1] = 1.

end_time = time.time()
time_spent = end_time - start_time

#Analytic solution
def u_a(y):
    value = 1. - (1.-np.exp(-10))*y -np.exp(-10.*y)
    return value

a_values = u_a(x)
num_values = u*h**2

"""
plt.figure()
plt.plot(x,num_values,label='Numerical values')
plt.scatter(x,num_values,s=10,color='red',marker='s',label='Numerical values')
plt.xlabel('x',fontsize=15)
plt.ylabel('u(x)',fontsize=15)

plt.legend()
plt.show()
"""
print('Time elapsed on algorithm: %s ' %time_spent)
