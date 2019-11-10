import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import sys

#Normalized by number of spins
mat1 = np.loadtxt("L40_expec_values.txt",skiprows=1) / (40**2)
mat2 = np.loadtxt("L60_expec_values.txt",skiprows=1) / (60**2)
mat3 = np.loadtxt("L80_expec_values.txt",skiprows=1) / (80**2)
mat4 = np.loadtxt("L100_expec_values.txt",skiprows=1) / (100**2)

temp = mat1[:,0]*40**2 #Temp is not normalized

plt.plot(temp,mat1[:,1], label="L = 40")
plt.plot(temp,mat2[:,1], label="L = 60")
plt.plot(temp,mat3[:,1], label="L = 80")
plt.plot(temp,mat4[:,1], label="L = 100")
plt.xlabel("Temperature")
plt.ylabel("Energy")
plt.legend()
plt.show()

plt.plot(temp,mat1[:,2], label="L = 40")
plt.plot(temp,mat2[:,2], label="L = 60")
plt.plot(temp,mat3[:,2], label="L = 80")
plt.plot(temp,mat4[:,2], label="L = 100")
plt.xlabel("Temperature")
plt.ylabel("Magnetic Moment")
plt.legend()
plt.show()

plt.plot(temp,mat1[:,3], label="L = 40")
plt.plot(temp,mat2[:,3], label="L = 60")
plt.plot(temp,mat3[:,3], label="L = 80")
plt.plot(temp,mat4[:,3], label="L = 100")
plt.xlabel("Temperature")
plt.ylabel("Heat capacity")
plt.legend()
plt.show()

plt.plot(temp,mat1[:,4], label="L = 40")
plt.plot(temp,mat2[:,4], label="L = 60")
plt.plot(temp,mat3[:,4], label="L = 80")
plt.plot(temp,mat4[:,4], label="L = 100")
plt.xlabel("Temperature")
plt.ylabel("Susceptibility")
plt.legend()
plt.show()
