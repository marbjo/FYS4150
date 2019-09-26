import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import sys

i = int(sys.argv[1])

#Run program as python3 plot.py 0 / 1 / 2 / 3 for plotting the
#different omegas

name_list = ["R0.010000", "R0.500000", "R1.000000", "R5.000000"]
mat = np.loadtxt(name_list[i])

rho_vec = np.linspace(0,4.78,len(mat[:,0]))
plt.plot(rho_vec,mat[:,0])
plt.xlabel("Rho")
plt.ylabel("Eigenvectors")
plt.show()
