import numpy as np
import matplotlib.pyplot as plt
import sys

#Reading the txt file. Text file is chosen by commmand line argument j, as n=10^j.
#I.E. python3 program.py 2 reads the text file for n=10^2 and plots
<<<<<<< HEAD


#Comment in this block if you want the figures for the general algorithm
name = "gaussian_"
arg = name + str(sys.argv[1])
name = arg + ".txt"

"""
#Comment in this block if you want the figures for the specialized algorithm
name = "special_gaussian_"
arg = name + str(sys.argv[1])
name = arg + "_fast.txt"
"""

=======
#n = str(sys.argv[1])

name = "gaussian_"
name = name + str(sys.argv[1])
name = name + ".txt"
>>>>>>> master
list = open(name).read()
list = [item.split() for item in list.split('\n')[:-1]]

#Converting to numpy array
matrix = np.array(list)

#Converting string elements to float
x = matrix[:,0].astype(float)
computed = matrix[:,1].astype(float)
exact = matrix[:,2].astype(float)
rel_err = matrix[:,3].astype(float)

<<<<<<< HEAD
=======
#Calculating a simple average using the first and last value of the relative
#error matrix
avg_err = (rel_err[0]+rel_err[-1] ) / 2.

>>>>>>> master

#concatenate adds Dirichlet conditions manually on each side of the vectors
x_new = np.concatenate([[0],x,[1]])
computed_new = np.concatenate([[0],computed,[0]])
exact_new = np.concatenate([[0],exact,[0]])

#Plotting
<<<<<<< HEAD

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

plt.rc('font', **font)

plt.figure(1)
plt.plot(x_new,computed_new,'s')
plt.plot(x_new,exact_new)
plt.legend(["Computed","Exact"])
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title("Numerical vs analytical solution")
plt.savefig(arg, bbox_inches='tight')
plt.show()


#Plotting relative error from data files
name2 = "standard_gaussian_error.txt"
list2 = open(name2).read()
list2 = [item.split() for item in list2.split('\n')[:-1]]

matrix2 = np.array(list2)
error_values = matrix2[:].astype(float)
step_vec = [1./10**(k+1) for k in range(7)]

plt.figure(2)
plt.loglog(step_vec,error_values)
plt.grid()
plt.xlabel("Step size, h")
plt.ylabel("Error values")
plt.title("Relative error as a function of step size.")
plt.savefig("RelativeError", bbox_inches='tight')
plt.show()

name3 = "specialized_gaussian_error.txt"
list3 = open(name3).read()
list3 = [item.split() for item in list3.split('\n')[:-1]]

matrix3 = np.array(list3)
error_values = matrix3[:].astype(float)
step_vec = [1./10**(k+1) for k in range(7)]

plt.figure(3)
plt.loglog(step_vec,error_values)
plt.grid()
plt.xlabel("Step size, h")
plt.ylabel("Error values")
plt.title("Relative error as a function of step size.")
plt.savefig("RelativeErrorSpecial", bbox_inches='tight')
plt.show()
=======
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
>>>>>>> master
