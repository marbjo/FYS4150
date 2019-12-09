# howdy
# ######### IMPORTS #############
import time
import numpy as np
from numba import jit, prange
from random import random, seed
import matplotlib.pyplot as plt
# ######### IMPORTS #############

# ########## GENERAL SHAPE PLOT ############################
shape_plot = False
if shape_plot == True:
    steps = np.linspace(0,99,101)
    vals = np.random.randint(0,99, size = steps.shape[0])
    beta = 1 / np.mean(vals)
    vals = vals[np.argsort(vals)]
    omega = beta*np.exp(-beta*steps)
    norm_steps = steps/ np.mean(vals)

    fig = plt.figure()
    plt.plot(norm_steps, omega)
    plt.show()
# ########## GENERAL SHAPE PLOT ############################

class FinanceExperiment:
    def __init__(
            self,
            agents = 500, starting_amt = 1000,
            MCsteps = 1e3, transactions = 1e7,
            lam = 0, alp = 0, gam = 0,
            bin_max = 'BRUTE',
            filename = None
        ):

        if (type(filename) is not str):
            raise TypeError('Put in a file name or else...')
        if (bin_max not in ['BRUTE', 'ELEGANT']):
            raise TypeError("bin_max must be in ['BRUTE','ELEGANT']")
        # INIT ARGUEMENTS
        self._agents = agents
        self._starting_amt = starting_amt
        self._MCsteps = int(MCsteps)
        self._transactions = int(transactions)
        self._lam = lam
        self._alf = alp
        self._gam = gam
        self._filename = filename
        self._equity = np.zeros(agents)
        self._diff_vec = np.zeros(int(MCsteps)-1)

        # BIN ARGS
        self._bin_size = 0.01 * starting_amt
        if (bin_max == 'BRUTE'):
            self._bin_max = 10 * starting_amt
        elif (bin_max == 'ELEGANT'):
            self._bin_max = 2*starting_amt/np.sqrt(gam + 0.01)
        self._bin_num = int(self._bin_max/self._bin_size)
        self._bin_steps = np.linspace(0, self._bin_max, self._bin_num)
        self._bin_vals = np.zeros(self._bin_num)

        self._val_holder = np.array([np.zeros(self._bin_num) for step in range(self._MCsteps)])
        
        if (self._MCsteps == 1):
            self._trans_holder = np.array([np.zeros(self._agents) for trn in range(int(self._transactions/100))])

    # wrap jitted function
    def montecarlo(self):
        _montecarlo(self._MCsteps, self._starting_amt, self._transactions, self._agents, self._equity,
                    self._bin_num, self._bin_size, self._bin_vals, self._val_holder, self._lam, self._alf, self._gam)

    # wrap jitted function
    def error_calc(self):
        _error_calc(self._MCsteps, self._diff_vec, self._val_holder)

    # wrap jit function
    def storeTransactions(self):
        _storeTransactions(self._starting_amt, self._transactions, self._agents, self._equity, self._trans_holder,
                           self._lam, self._alf, self._gam)
    
    def calcVar(self):
        self._variance = np.var(self._trans_holder, axis = 1)
    
    
    def saveBinParams(self):
        np.savez(self._filename + 'BinParams', binSize = self._bin_size, binMax = self._bin_max,
                 binNumber = self._bin_num, binSteps = self._bin_steps)
        
    def saveBinVals(self):
        np.savez(self._filename + 'BinVals', final_dist = self._bin_vals,
                 error_vals = self._diff_vec, all_vals = self._val_holder)
    
    def saveTransVar(self):
        np.savez(self._filename + 'TransVar', trans_variance = self._variance)

# jitted for speed yo
@jit(nopython = True)
def _error_calc(MCsteps, diff_vec, val_holder):
    for i in range(0, MCsteps-1):
        diff_vec[i]=np.linalg.norm(val_holder[i]/(i+1)-val_holder[i+1]/(i+2))

# jitted function for calculations
@jit(nopython = True)
def _montecarlo(MCsteps, starting_amt, transactions, agents, equity, bin_num, bin_size, bin_vals, val_holder, lambd, alpha
                , gamma):
    
    for mc_val in range(MCsteps):
        print(mc_val)
        equity.fill(starting_amt)
        counter = np.zeros((agents,agents))
        for deals in range(transactions):
            # TRANSACTION CUZZO
            eps = np.random.uniform(0,1)
            z = np.random.uniform(0,1)
            temp = np.random.choice(agents, 2, replace = False)
            idx_i = temp[0]
            idx_j = temp[1]
            
            if (equity[idx_i] == equity[idx_j]):
                prob = 1
            else:
                prob = abs((equity[idx_i] - equity[idx_j])/starting_amt)**(-alpha)*(counter[idx_i,idx_j] + 1)**gamma
            
            if (z < prob):
                m1 = lambd * equity[idx_i] + (1 - lambd) * eps * (equity[idx_i] + equity[idx_j])
                m2 = lambd * equity[idx_j] + (1 - lambd) * (1 - eps) * (equity[idx_i] + equity[idx_j])
                equity[idx_i] = m1
                equity[idx_j] = m2
                counter[idx_i,idx_j]+=1
                counter[idx_j,idx_i]+=1

        # UPDATE BINS YO
        for i in range(agents):
            for j in range(bin_num):
                if ((equity[i] > j*bin_size) and (equity[i] < (j+1)*bin_size)):
                    bin_vals[j] += 1
        val_holder[mc_val] = bin_vals

#jit speed up        
@jit(nopython = True)
def _storeTransactions(starting_amt, transactions, agents, equity, trans_holder, lambd, alpha, gamma):
    counter = np.zeros((agents,agents))
    equity.fill(starting_amt)
    for deals in range(transactions):
        # TRANSACTION CUZZO
        eps = np.random.uniform(0,1)
        z = np.random.uniform(0,1)
        temp = np.random.choice(agents, 2, replace = False)
        idx_i = temp[0]
        idx_j = temp[1]
        
        if (equity[idx_i] == equity[idx_j]):
            prob = 1
        else:
            prob = abs((equity[idx_i] - equity[idx_j])/starting_amt)**(-alpha)*(counter[idx_i,idx_j] + 1)**gamma
            
        if (z < prob):
            m1 = lambd * equity[idx_i] + (1 - lambd) * eps * (equity[idx_i] + equity[idx_j])
            m2 = lambd * equity[idx_j] + (1 - lambd) * (1 - eps) * (equity[idx_i] + equity[idx_j])
            equity[idx_i] = m1
            equity[idx_j] = m2
            counter[idx_i,idx_j]+=1
            counter[idx_j,idx_i]+=1
        # SAVE FOR VARIANCE OF M/<M>, SAVE ONCE EVERY 100 VALUES TO SO COMP DONT CATCH FIRE
            
        if (deals%100 == 0):
            print('hello', deals)
            it = deals//100
            trans_holder[it] = equity/starting_amt

    
    
if __name__ == '__main__':
    
    #a = FinanceExperiment(agents = 1000, starting_amt = 1000, MCsteps = 1, transactions = 1e7, lam = 0, alp = 1.0, gam = 1.0,
    #                      filename = 'parte_1000_0_10_10_')
    #a.storeTransactions()
    #a.calcVar()
    #a.saveTransVar()

    a = FinanceExperiment(agents = 1000, starting_amt = 1000, MCsteps = 1e3, transactions = 1e6, lam = 0.25, alp = 2.0, gam = 0.0,
                          filename = 'partd_1000_25_20_00_')
    a.montecarlo()
    a.error_calc()
    a.saveBinVals()
    a.saveBinParams()
    a = FinanceExperiment(agents = 1000, starting_amt = 1000, MCsteps = 1e4, transactions = 1e5, lam = 0.25, alp = 1.5, gam = 0.0,
                          filename = 'partd_1000_25_15_00_')
    a.montecarlo()
    a.error_calc()
    a.saveBinVals()
    a = FinanceExperiment(agents = 1000, starting_amt = 1000, MCsteps = 1e4, transactions = 1e5, lam = 0.25, alp = 1.0, gam = 0.0,
                          filename = 'partd_1000_25_10_00_')
    a.montecarlo()
    a.error_calc()
    a.saveBinVals()
    a = FinanceExperiment(agents = 1000, starting_amt = 1000, MCsteps = 1e4, transactions = 1e5, lam = 0.25, alp = 0.5, gam = 0.0,
                          filename = 'partd_1000_25_00_00_')
    a.montecarlo()
    a.error_calc()
    a.saveBinVals()


   
