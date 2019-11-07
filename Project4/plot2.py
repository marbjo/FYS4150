import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import sys


mat = np.loadtxt("expec_values.txt",skiprows=1)
L = 20 #Number of spins
temp = mat[:,0]
E = mat[:,1]
M = mat[:,2]
C_v = mat[:,3]
chi = mat[:,4]

plt.plot(temp,E)
plt.xlabel("Temperature")
plt.ylabel("Energy")
plt.show()

plt.plot(temp,M)
plt.xlabel("Temperature")
plt.ylabel("Magnetic Moment")
plt.show()

plt.plot(temp,C_v)
plt.xlabel("Temperature")
plt.ylabel("Heat capacity")
plt.show()

plt.plot(temp,E)
plt.xlabel("Temperature")
plt.ylabel("Susceptibility")
plt.show()
