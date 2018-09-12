import numpy as np
import matplotlib.pyplot as plt

start = 0.
end = 1.
n_list = [10**c for c in range(1,8)]
error_list = []
h_list = []
for n in n_list:
    print("Now looping for n= %d " % (n) )
    #Initializing values
    h = (end-start)/(n)
    h_list.append(h)
    x = np.linspace(start,end,n+1)
    f_value = 100.*np.exp(-10.*x)

    diag = 2*np.ones(n+1)
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

    error =  np.abs( (num_values[1:-1] - a_values[1:-1])/a_values[1:-1] )
    print('Max error is: %s' %(np.nanmax(error)) )
    log_error = np.log10(error)
    max_error = np.nanmax(log_error[log_error != np.inf]) #np.nanmax(error)
    error_list.append(max_error)

#print(h_list)
#print(error_list)

plt.plot(np.log10(h_list),error_list)
plt.xlabel(r'$log_{10}(h)$',fontsize=15)
plt.ylabel(r'$log_{10}(\epsilon$)', fontsize=15)
plt.suptitle('Relative error as a function of step size',fontsize=20)
plt.show()
