import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import sys

#i = int(sys.argv[1])

#Run program as python3 plot.py 0 / 1 / 2 / 3 for plotting the
#ground state for different omegas
name_list1 = ["Eig_Coulomb_0.010000", "Eig_Coulomb_0.500000","Eig_Coulomb_1.000000", "Eig_Coulomb_5.000000"]
name_list2 = ["R_Coulomb_0.010000", "R_Coulomb_0.500000", "R_Coulomb_1.000000", "R_Coulomb_5.000000"]
omega_list = [0.01, 0.5, 1.0, 5.0]

for k in range(len(name_list1)):
    mat1 = np.loadtxt(name_list1[k])
    min_ind = np.argmin(mat1)
    mat1 = np.sort(mat1)
    print(mat1[0:3])

    mat2 = np.loadtxt(name_list2[k])
    rho_vec = np.linspace(0,4.78, len(mat2[:,min_ind])) #len(mat2[:,min_ind]))
    name = "omega = " + str(omega_list[k])
    plt.plot(rho_vec,abs(mat2[:,min_ind])**2,label=name)

mat1 = np.loadtxt("Eig_quantum_dot")
min_ind = np.argmin(mat1)
mat2 = np.loadtxt("R_quantum_dot")
plt.plot(rho_vec,abs(mat2[:,min_ind])**2, 'k--', label="No Coulomb")
plt.xlabel(r"$\rho$")
plt.ylabel(r"$|\Psi|^2$")
plt.legend()
plt.show()
