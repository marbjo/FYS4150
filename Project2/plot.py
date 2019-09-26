import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import sys

#i = int(sys.argv[1])

#Run program as python3 plot.py 0 / 1 / 2 / 3 for plotting the
#ground state for different omegas

name_list = ["R_Coulomb_0.010000", "R_Coulomb_0.500000", "R_Coulomb_1.000000", "R_Coulomb_5.000000"]
omega_list = [0.01, 0.5, 1.0, 5.0]

#indices for first eigenvalue are 49,50, 34, 9
#for omega = 0.01, 0.5, 1.0, 5.0 respectively
ind_list = [49,50,34,9]

#mat = np.loadtxt(name_list[i])
#rho_vec = np.linspace(0,4.78,len(mat[:,ind_list[i]]))
#plt.plot(rho_vec,mat[:,ind_list[i]])

for k in range(len(name_list)):
    mat = np.loadtxt(name_list[k])
    rho_vec = np.linspace(0,4.78,len(mat[:,ind_list[k]]))
    name = "omega = " + str(omega_list[k])
    plt.plot(rho_vec,mat[:,ind_list[k]],label=name)
    plt.xlabel("Rho")
    plt.ylabel("Eigenvectors")

buckbeam_min_ind = 16

mat = np.loadtxt("R_quantum_dot")
plt.plot(rho_vec,mat[:,buckbeam_min_ind], 'k--', label="No Coulomb")
plt.legend()
plt.show()
