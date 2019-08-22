import numpy as np
import matplotlib.pyplot as plt

n = int(  input('Give n (Grid point precision, #): ') )
start = 0.
end = 1.
h = 1./(n+1)
x = np.linspace(start,end,n+2)
#print(len(x))
f_value = 100.*np.exp(-10.*x)

diag = 2*np.ones(n)
f_value[0] = 0
f_value[-1] = 0
f_tilde = f_value

#Calculating diagonal and f_tilde
for i in range(1,n):
    diag[i] = (i+1)/i
    f_tilde[i] = f_value[i] + ((i-1)/i) *f_tilde[i-1]

#Making u array
u = f_tilde

#Removing elements above diagonal
for j in range(n,0,-1):
    u[j-1] = (j-1)/j * (f_tilde[j-1] + u[j])

def u_a(y):
    #Analytic solution
    value = 1. - (1.-np.exp(-10))*y -np.exp(-10.*y)
    return value

a_values = u_a(x)
num_values = u*h**2

plt.figure()
plt.plot(x,num_values,label='Numerical values')
plt.plot(x,a_values, label='Analytical values')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.legend()
plt.show()
