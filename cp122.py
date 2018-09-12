import numpy as np
import matplotlib.pyplot as plt
import time

n = int(  input('Give n (Grid point precision, #): ') )
start = 0.
end = 1.
h = 1./(n+1)
x = np.linspace(start,end,n+2)
f_value = 100.*np.exp(-10.*x)

inv_diag = np.zeros(n)
f_value[0] = 0
f_value[-1] = 0
f_tilde = f_value

#Calculating diagonal prefactors
for i in range(0,n):
    inv_diag[i] = i/(i+1)

start_time = time.time()
for k in range(1,n+1):
    f_tilde[k] = f_value[k] + inv_diag[k-1] *f_tilde[k-1]

#Making u array
u = f_tilde

#Removing elements above diagonal
for j in range(n,0,-1):
    u[j-1] = inv_diag[j-1] * (f_tilde[j-1] + u[j])

end_time = time.time()
time_spent = end_time - start_time

def u_a(y):
    #Analytic solution
    value = 1. - (1.-np.exp(-10))*y -np.exp(-10.*y)
    return value

a_values = u_a(x)
num_values = u*h**2

"""
plt.figure()
plt.scatter(x,num_values,s=5,color='red',marker='s',label='Numerical values')
plt.plot(x,a_values, label='Analytical values')
plt.xlabel('x',fontsize=15)
plt.ylabel('u(x)',fontsize=15)
plt.suptitle('Numerical solution to the Poisson equation',fontsize=20)
plt.legend()
plt.show()
"""
print('Time elapsed on algorithm: %s ' %time_spent)
