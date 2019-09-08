import numpy as np
import matplotlib.pyplot as plt
import sys

#Reading the txt file. Text file is chosen by commmand line argument j, as n=10^j.
#I.E. python3 program.py 2 reads the text file for n=10^2 and plots
#n = str(sys.argv[1])

name = "gaussian_"
name = name + str(sys.argv[1])
name = name + ".txt"
list = open(name).read()
list = [item.split() for item in list.split('\n')[:-1]]

#Converting to numpy array
matrix = np.array(list)

#Converting string elements to float
x = matrix[:,0].astype(float)
computed = matrix[:,1].astype(float)
exact = matrix[:,2].astype(float)
rel_err = matrix[:,3].astype(float)

#Calculating a simple average using the first and last value of the relative
#error matrix
avg_err = (rel_err[0]+rel_err[-1] ) / 2.


#concatenate adds Dirichlet conditions manually on each side of the vectors
x_new = np.concatenate([[0],x,[1]])
computed_new = np.concatenate([[0],computed,[0]])
exact_new = np.concatenate([[0],exact,[0]])

#Plotting
plt.plot(x_new,computed_new)
plt.plot(x_new,exact_new)
plt.legend(["Computed","Exact"])
plt.show()



"""
* Fix such that you can have headers for the columns and ignore the first line
 when reading
* Make several command line arguments make the program read several text
files and plot them in a subplot.
"""

#Github sharing
#Time?
#Mean error?
