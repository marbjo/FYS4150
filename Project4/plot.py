import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import sys


import pandas as pd
import seaborn as sns
from scipy import stats

#Loading text file with temperature in front, skipping header.
#mat = np.loadtxt("1_MC_results.txt",skiprows=1)
mat = np.loadtxt("2.4_MC_results.txt",skiprows=1)
L = 20 #Number of spins
E_norm_by_spins = mat[:,1] / L**2
E_of_MC = mat[:,2]
M = mat[:,3]
accept_flips = mat[:,4]

print("We got this far")
N = len(mat[:,0])
plot_vec = np.linspace(0,N,N)

plt.title("|E| versus MonteCarlo steps")
plt.plot(plot_vec,E_of_MC, label="Energy")
plt.xlabel("log(N)")
plt.ylabel("E")
plt.legend(loc="lower left")
plt.show()

plt.title("|M| versus MonteCarlo steps")
plt.plot(plot_vec, M,label="Magnetic moment")
plt.xlabel("log(N)")
plt.ylabel("|M|")
plt.legend(loc="lower left")
plt.show()


plt.title("Number of accepted steps versus MonteCarlo steps")
plt.plot(plot_vec, accept_flips,label="Accepted")
plt.xlabel("log(N)")
plt.ylabel("#")
plt.show()

eq_index = 10000 #Equilibrium index, assuming equilibrium at N=10^4 (see from graph)

#Computing statistical properties
n,bins = np.histogram(E_norm_by_spins[eq_index:])
mids = 0.5*(bins[1:] + bins[:-1])
mean = np.average(mids, weights=n)
var = np.average((mids - mean)**2, weights=n)
print("Variance: %g" %var)
print("Mean: %g" %mean)
std = np.sqrt(var)

#Creating distribution with computed values, and plotting for comparison
#x = np.random.normal(loc=mean, scale=std, size=eq_index)
#sns.distplot(x);
#plt.show()

plt.hist(E_norm_by_spins[eq_index:], bins="auto", histtype='bar', ec='black', color = 'green', label='Energy', density = True)
#plt.plot(x)
plt.xlabel('E', fontsize=20)
plt.ylabel('Occurences (P(E))', fontsize=20)
plt.title('', fontsize=20)
#plt.legend()
plt.show()
