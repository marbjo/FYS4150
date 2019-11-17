import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import sys


import pandas as pd
import seaborn as sns
from scipy import stats

plot_b = False
if(plot_b == True):
    partb = np.loadtxt("analytic_comparison_part_b.txt", skiprows = 1)

    E_MC = partb[:-1,0]
    M_MC = partb[:-1,1]
    C_V_MC = partb[:-1,2]
    Chi_MC = partb[:-1,3]

    a_E = partb[-1,0]
    a_M = partb[-1,1]
    a_C_V = partb[-1,2]
    a_Chi = partb[-1,3]

    a_E_vec = np.ones_like(E_MC)*a_E
    a_M_vec = np.ones_like(M_MC)*a_M
    a_C_V_vec = np.ones_like(C_V_MC)*a_C_V
    a_Chi_vec = np.ones_like(Chi_MC)*a_Chi

    plot_len = np.arange(E_MC.shape[0])

    sns.set_style('dark', {'axes.grid':True,'axes.edgecolor':'black', 'font.family':['serif'], 'font.serif':['Roman'],
                           'xtick.bottom':True, 'ytick.left':True})
    print(sns.axes_style())
    fig, ax= plt.subplots()
    ax.set_title('T = 1.0 kT/J')
    ax.set_xlabel('Monte Carlo Cycles')
    ax.semilogx(plot_len, a_E_vec, label = 'Analytical Solution')
    ax.semilogx(plot_len, E_MC, label = 'Computed Solution')
    ax.set_ylabel(r'Mean Energy $\langle E(T) \rangle$')
    ax.legend()
    plt.savefig('partb_energy')

    fig, ax= plt.subplots()
    ax.set_title('T = 1.0 kT/J')
    ax.semilogx(plot_len, a_M_vec, label = 'Analytical Solution')
    ax.semilogx(plot_len, M_MC, label = 'Computed Solution')
    ax.set_ylabel(r'Mean Magnetization $\langle |M(T)| \rangle$')
    ax.legend()
    ax.set_xlabel('Monte Carlo Cycles')
    plt.savefig('partb_mag')


    fig, ax= plt.subplots()
    ax.set_title('T = 1.0 kT/J')
    ax.semilogx(plot_len, a_C_V_vec, label = 'Analytical Solution')
    ax.semilogx(plot_len, C_V_MC, label = 'Computed Solution')
    ax.set_ylabel(r'Heat Capacity $C_V$')
    ax.legend()
    ax.set_xlabel('Monte Carlo Cycles')
    plt.savefig('partb_cv')


    fig, ax= plt.subplots()
    ax.set_title('T = 1.0 kT/J')
    ax.semilogx(plot_len, a_Chi_vec, label = 'Analytical Solution')
    ax.semilogx(plot_len, Chi_MC, label = 'Computed Solution')
    ax.set_ylabel(r'Susceptibility $\chi$')
    ax.legend()
    ax.set_xlabel('Monte Carlo Cycles')
    plt.savefig('partb_chi')

    plt.show()

plot_c = False
if(plot_c == True):

    mat_random_1 = np.loadtxt("1.000_MC_results_random_L20.txt", skiprows = 1)
    mat_ordered_1 = np.loadtxt("1.000_MC_results_ordered_L20.txt", skiprows = 1)
    mat_random_24 = np.loadtxt("2.400_MC_results_random_L20.txt", skiprows = 1)
    mat_ordered_24 = np.loadtxt("2.400_MC_results_ordered_L20.txt", skiprows = 1)


    E_r1 = mat_random_1[:,2]
    M_r1 = mat_random_1[:,3]
    flips_r1 = mat_random_1[:,4]

    E_o1 = mat_ordered_1[:,2]
    M_o1 = mat_ordered_1[:,3]
    flips_o1 = mat_ordered_1[:,4]

    E_r24 = mat_random_24[:,2]
    M_r24 = mat_random_24[:,3]
    flips_r24 = mat_random_24[:,4]

    E_o24 = mat_ordered_24[:,2]
    M_o24 = mat_ordered_24[:,3]
    flips_o24 = mat_ordered_24[:,4]


    N = len(mat_random_1[:,0])
    plot_vec = np.linspace(0,N,N)

    sns.set_style('dark', {'axes.grid':True,'axes.edgecolor':'black', 'font.family':['serif'], 'font.serif':['Roman'],
                           'xtick.bottom':True, 'ytick.left':True})

    # T = 1.0
    fig, (ax1, ax2) = plt.subplots(2,)
    ax1.set_title(r'T = 1.0 kT/J', x=0.52)
    plt.xlabel('Monte Carlo Cycles')
    ax1.semilogx(plot_vec,E_r1, label = 'Random Initial State', linewidth = 0.8)
    ax1.legend(loc = 'upper right')
    ax1.set_ylabel(r'$\langle |E| \rangle$')
    ax2.semilogx(plot_vec, E_o1, label = 'Ordered Initial State', linewidth = 0.8)
    ax2.legend(loc = 'upper right')
    ax2.set_ylabel(r'$\langle |E| \rangle$')
    plt.savefig('e_t1_l20')
    plt.show()

    fig, (ax1, ax2) = plt.subplots(2,)
    ax1.set_title(r'T = 1.0 kT/J', x=0.52)
    plt.xlabel('Monte Carlo Cycles')
    ax1.semilogx(plot_vec, M_r1, label = 'Random Initial State', linewidth = 0.8)
    ax1.legend(loc = 'upper left')
    ax1.set_ylabel(r'$\langle |M| \rangle$')
    ax2.semilogx(plot_vec, M_o1, label = 'Ordered Initial State', linewidth = 0.6)
    ax2.legend(loc = 'upper right')
    ax2.set_ylabel(r'$\langle |M| \rangle$')
    plt.savefig('m_t1_l20')
    plt.show()

    fig, (ax1, ax2) = plt.subplots(2,)
    ax1.set_title(r'T = 1.0 kT/J', x=0.52)
    plt.xlabel('Monte Carlo Cycles')
    ax1.semilogx(plot_vec, flips_r1, label = 'Random Initial State', linewidth = 0.8)
    ax1.legend(loc = 'upper right')
    ax1.set_ylabel(r'Accepted Spin Flips')
    ax2.plot(plot_vec, flips_o1, label = 'Ordered Initial State', linewidth = 0.8)
    ax2.legend(loc = 'upper right')
    ax2.set_ylabel(r'Accepted Spin Flips')
    plt.savefig('flips_t1_l20')
    plt.show()

    # T = 2.4
    fig, ax1 = plt.subplots()
    ax1.set_title(r'T = 2.4 kT/J', x=0.52)
    plt.xlabel('Monte Carlo Cycles')
    ax1.semilogx(plot_vec,E_r24, label = 'Random Initial State', linewidth = 0.8)
    ax1.semilogx(plot_vec, E_o24, label = 'Ordered Initial State', linewidth = 0.8)
    ax1.legend(loc = 'upper right')
    ax1.set_ylabel(r'$\langle |E| \rangle$')
    ax1.set_xlabel(r'Monte Carlo Cycles')
    plt.savefig('e_t24_l20')
    plt.show()

    fig, ax1 = plt.subplots()
    ax1.set_title(r'T = 2.4 kT/J', x=0.52)
    plt.xlabel('Monte Carlo Cycles')
    ax1.semilogx(plot_vec, M_r24, label = 'Random Initial State', linewidth = 0.8)
    ax1.semilogx(plot_vec, M_o24, label = 'Ordered Initial State', linewidth = 0.8)
    ax1.legend(loc = 'upper left')
    ax1.set_ylabel(r'$\langle |M| \rangle$')
    ax1.set_xlabel(r'Monte Carlo Cycles')
    plt.savefig('m_t24_l20')
    plt.show()

    fig, (ax1, ax2) = plt.subplots(2,)
    ax1.set_title(r'T = 2.4 kT/J', x=0.52)
    plt.xlabel('Monte Carlo Cycles')
    ax1.semilogx(plot_vec, flips_r24, label = 'Random Initial State', linewidth = 0.8)
    ax1.legend(loc = 'upper right')
    ax1.set_ylabel(r'Accepted Spin Flips')
    ax2.plot(plot_vec, flips_o24, label = 'Ordered Initial State', linewidth = 0.8)
    ax2.legend(loc = 'upper right')
    ax2.set_ylabel(r'Accepted Spin Flips')
    plt.savefig('flips_t24_l20')
    plt.show()




histogram = True
if histogram ==True:

    mat_random_1 = np.loadtxt("1.000_MC_results_random_L20.txt", skiprows = 1)
    mat_random_24 = np.loadtxt("2.400_MC_results_random_L20.txt", skiprows = 1)

    eq_index = 10000 #Equilibrium index, assuming equilibrium at N=10^4 (see from graph)
    L = 20 #Number of spins
    E_norm_1 = mat_random_1[:,1] / L**2
    E_norm_24 = mat_random_24[:,1]/L**2

    bins1 = np.arange(np.min(E_norm_1), np.max(E_norm_1), .01)
    bins24 = np.arange(np.min(E_norm_24), np.max(E_norm_24), 4 / 400)

    fig, ax = plt.subplots(2, 1)
    ax[0].hist(E_norm_1, bins=bins1[:10], density=True)
    ax[1].hist(E_norm_24, bins=bins24, density=True)
    ax[0].set_title(r"$kT/J = 1$")
    ax[1].set_title(r"$kT/J = 2.4$")
    ax[1].set_xlabel(r"$E / L^2$ [J]")
    ax[0].set_ylabel(r"% of occurences")
    ax[1].set_ylabel(r"% of ocurrences")
    fig.tight_layout(w_pad=1)
    plt.savefig('probabilities')
    plt.show()

    var1 = np.var(E_norm_1)
    var24 = np.var(E_norm_24)
    print('T = 1 Variance: ', var1)
    print('T = 2.4 variance: ', var24)




    #
