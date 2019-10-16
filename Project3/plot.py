import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
'''
Set up data in easy to read data frames for plotting and analysis

'''
pd.set_option('precision', 10)

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
        0 : 'int pts',
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
print(df_quadrature)
print(df_montecarlo)
# fig = plt.figure()
# sns.set_style('darkgrid')
# plt.plot(df_)
