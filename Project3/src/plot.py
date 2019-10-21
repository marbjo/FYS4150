import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker
'''
Set up data in easy to read data frames for plotting and analysis

Plotting code follows below dataframes

'''
pd.set_option('precision', 8)

df_parallel_bf_1 = pd.read_csv(
    '1_thread_monte_brute_force.txt',
    delim_whitespace=True,
    header=None,
)
df_parallel_bf_1.rename(
    columns = {
        0:'int points',
        1:'BF int',
        2:'BF error',
        3:'BF variance',
        4:'BF time',
    },
    inplace=True,
)

df_parallel_bf_8 = pd.read_csv(
    '8_thread_monte_brute_force.txt',
    delim_whitespace=True,
    header=None,
)
df_parallel_bf_8.rename(
    columns = {
        0:'int points',
        1:'BF int',
        2:'BF error',
        3:'BF variance',
        4:'BF time',
    },
    inplace=True,
)

df_parallel_imp_1 = pd.read_csv(
    '1_thread_monte_improved.txt',
    delim_whitespace=True,
    header=None,
)
df_parallel_imp_1.rename(
    columns = {
        0:'int points',
        1:'imp int',
        2:'imp error',
        3:'imp variance',
        4:'imp time',
    },
    inplace=True,
)

df_parallel_imp_8 = pd.read_csv(
    '8_thread_monte_improved.txt',
    delim_whitespace=True,
    header=None,
)
df_parallel_imp_8.rename(
    columns = {
        0:'int points',
        1:'imp int',
        2:'imp error',
        3:'imp variance',
        4:'imp time',
    },
    inplace=True,
)

df_quadrature = pd.read_csv(
    'brute_force_results.txt',
    delim_whitespace=True,
    header = None,
)
df_quadrature.rename(
    columns = {
        0 : 'int points',
        1 : 'Leg int',
        2 : 'Leg error',
        3 : 'Leg time',
        4 : 'Lag int',
        5 : 'Lag error',
        6 : 'Lag time'
    },
    inplace = True,
)

df_montecarlo = pd.read_csv(
    'montecarlo_results.txt',
    delim_whitespace = True,
    header = None,

)

df_montecarlo.rename(
    columns = {
        0 : 'int points',
        1 : 'BF int',
        2 : 'BF error',
        3 : 'BF variance',
        4 : 'BF time',
        5 : 'improved int',
        6 : 'improved error',
        7 : 'improved variance',
        8 : 'improved time',
    },
    inplace = True,
)
df_o1flag = pd.read_csv(
    '8_thread_monte_improved_O.txt',
    delim_whitespace=True,
    header=None,
)
df_o2flag = pd.read_csv(
    '8_thread_monte_improved_O2.txt',
    delim_whitespace=True,
    header=None,
)
df_o3flag = pd.read_csv(
    '8_thread_monte_improved_O3.txt',
    delim_whitespace=True,
    header=None,
)

sns.set_style(
    'dark',
    {
        'axes.edgecolor': '.1',
        'xtick.bottom': True,
        'ytick.left': True,
    }
)
print(sns.axes_style())
fig, ax = plt.subplots()
ax.plot(df_parallel_bf_1['int points'],df_parallel_bf_1['BF time'], label = "BF 1 thread")
ax.plot(df_parallel_bf_8['int points'],df_parallel_bf_8['BF time'], label = "BF 8 threads")
ax.plot(df_parallel_imp_1['int points'],df_parallel_imp_1['imp time'], label = "Imp 1 thread")
ax.plot(df_parallel_imp_8['int points'],df_parallel_imp_8['imp time'], label = "Imp 8 threads")
ax.legend()
ax.grid(axis = 'x', linestyle = '--')
ax.grid(axis = 'y', linestyle = '--')
ax.set_xlabel('Integration Points')
ax.set_ylabel('CPU Time (s)')
fig.savefig('Parallel.png')
# plt.show()

fig, ax = plt.subplots()
ax.semilogy(df_quadrature['int points'], df_quadrature['Leg error'], label = 'Brute Force')
ax.semilogy(df_quadrature['int points'], df_quadrature['Lag error'], label = 'Improved')
ax.grid(axis = 'y', linestyle = '--')
ax.grid(axis = 'x', linestyle = '--')
ax.legend()
ax.set_xlabel('Integration Points')
ax.set_ylabel('Absolute Error')
fig.savefig('quadrature.png')
# plt.show()

fig, ax = plt.subplots()
ax.loglog(df_montecarlo['int points'], df_montecarlo['BF error'], label = 'Brute Force')
ax.loglog(df_montecarlo['int points'], df_montecarlo['improved error'], label = 'Improved')
ax.grid(axis = 'y', linestyle = '--')
ax.grid(axis = 'x', linestyle = '--')
ax.legend()
ax.set_xlabel('Integration Points')
ax.set_ylabel('Absolute Error')
fig.savefig('monteerror.png')

fig, ax = plt.subplots()
ax.loglog(df_montecarlo['int points'], df_montecarlo['BF variance'], label = 'Brute Force')
ax.loglog(df_montecarlo['int points'], df_montecarlo['improved variance'], label = 'Improved')
ax.grid(axis = 'y', linestyle = '--')
ax.grid(axis = 'x', linestyle = '--')
ax.legend()
ax.set_xlabel('Integration Points')
ax.set_ylabel('Variance')
fig.savefig('montevariance.png')
plt.show()

fig, ax = plt.subplots()
ax.loglog(df_o1flag[0], df_o1flag[4], label = 'O1')
ax.loglog(df_o2flag[0], df_o2flag[4], label = 'O2')
ax.loglog(df_o3flag[0], df_o3flag[4], label = 'O3')
ax.loglog(df_parallel_imp_8['int points'],df_parallel_imp_8['imp time'], label = "No Flag")
ax.grid(axis = 'y', linestyle = '--')
ax.grid(axis = 'x', linestyle = '--')
ax.legend()
ax.set_xlabel('Integration Points')
ax.set_ylabel('Runtime (s)')
fig.savefig('flags.png')
plt.show()
