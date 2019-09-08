import numpy as np
import matplotlib.pyplot as plt
import sys

#Reading the txt file. Text file is chosen by commmand line argument, as n.
#I.E. python3 program.py 10 reads the text file for n=10 and plots

"""
* Fix such that you can have headers for the columns and ignore the first line
 when reading
* Make several command line arguments make the program read several text
files and plot them in a subplot.
"""

n = str(sys.argv[1])
list = open(n).read()
list = [item.split() for item in list.split('\n')[:-1]]

#Converting to numpy array
matrix = np.array(list)

#Converting string elements to float
x = matrix[:,0].astype(float)
computed = matrix[:,1].astype(float)
exact = matrix[:,2].astype(float)
rel_err = matrix[:,3].astype(float)

#concatenate adds Dirichlet conditions manually on each side of the vectors
x_new = np.concatenate([[0],x,[1]])
computed_new = np.concatenate([[0],computed,[0]])
exact_new = np.concatenate([[0],exact,[0]])

#Plotting
plt.plot(x_new,computed_new)
plt.plot(x_new,exact_new)
plt.legend(["Computed","Exact"])
plt.show()
