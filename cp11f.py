import numpy as np
import matplotlib.pyplot as plt
from pandas import *

#np.random.seed(1) #Seed for testing random matrices
start = 0.
end = 1.
n_list = [10**c for c in range(1,4)]
h_list = []
x_list = []
plot_list = []

for n in n_list:
    h = (end-start)/float(n+1)
    x = np.linspace(start,end,n+2)
    x_list.append(x)
    h_list.append(h)
    #Setting function values and endpoints
    f_value = 100.*np.exp(-10.*x)
    f_value[0]=0; f_value[-1] = 0

    #Diagonal vectors. These can be random numbers
    #Note: no error handling for linearly dependent column vectors
    #(Leads to division by zero in loops)

    diag = 2*np.ones(n) #np.random.randint(1,10,n)
    u_diag = -np.ones(n-1) #np.random.randint(1,10,n-1)
    l_diag = -np.ones(n-1) #np.random.randint(1,10,n-1)

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

    num_values = u*h**2
    print(num_values.shape)
    plot_list.append(num_values)
    #print()

#Analytic solution
def u_a(y):
    value = 1. - (1.-np.exp(-10))*y -np.exp(-10.*y)
    return value

a_values = u_a(x)

plt.figure()
plt.suptitle('Numerical solution to the Poisson equation',fontsize=20)

N = 3
for l in range(N):
    plt.subplot(N, 1, l+1)
    plt.title('n=%d' %n_list[l], fontsize=20)
    plt.scatter(x_list[l],plot_list[l],s=8-2*l,color='b',marker='s',label='Numerical values')
    plt.plot(x,a_values,'r',label='Analytical values')
    plt.xlabel('x',fontsize=20)
    plt.ylabel('u(x)',fontsize=20)
    plt.legend()

plt.tight_layout()
plt.show()
