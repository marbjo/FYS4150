import numpy as np
import matplotlib.pyplot as plt
# import numb
import time


class Rotation_Object():
    '''
    Create an object that can be rotated via jacobi or solved via numpy

    potential options:
        'no pot' = solve with no potential
        'hamiltonian' = solve with hamiltonian potential
        'repulsive' = solve with repulsive hamiltonian potential

    '''
    def __init__(self, size, potential = 'no pot', omega = 0, start = 0, stop = 1):

        self._size = size
        self._pot = potential
        self._start = start
        self._stop = stop
        self._A = np.zeros((self._size, self._size))
        self._Asize = self._A.shape[0]
        self._R = np.eye(self._size)
        self._omega = omega

    @property
    def xvec(self):
        h = (self._stop - self._start) / (self._size + 1)
        self._xvec = np.zeros(self._size + 2)
        for i in range(self._size + 2):
            self._xvec[i] = self._start + i*h
        return self._xvec

    @property
    def build(self):
        '''
        build tridiagonal matrixes based on potential type, leaving out the first and
        last elements which will be manually set later as initial conditions.

        Builds matrices for:
            No potential = 'no pot'
            3D hamitonian = 'hamiltonian'
            2 particle non interacting potential = 'non interact'
            2 particle with coulomb potential = 'repulsive'
        '''
        self._xvec = self.xvec
        h = (self._stop - self._start) / (self._size + 1)
        d = 2 / h**2
        a = -1 / h**2

        if (self._pot == 'no pot'):
            for i in range(self._size):
                self._A[i,i] = d
                if (i != self._size-1):
                    self._A[i+1,i] = a
                    self._A[i,i+1] = a
            return self._A

        elif (self._pot == 'hamiltonian'):
            for i in range(self._size):
                self._A[i,i] = d + (self._xvec[i+1]**2)
                if (i != self._size-1):
                    self._A[i+1,i] = a
                    self._A[i,i+1] = a
            return self._A

        elif (self._pot == 'non interact'):
            for i in range(self._size):
                self._A[i,i] = d + (self._omega**2)*(self._xvec[i+1]**2)
                if (i != self._size-1):
                    self._A[i+1,i] = a
                    self._A[i,i+1] = a
            return self._A

        elif(self._pot == 'repulsive'):
            for i in range(self._size):
                self._A[i,i] = d + (self._omega**2)*(self._xvec[i+1]**2) + 1/self._xvec[i+1]
                if (i != self._size-1):
                    self._A[i+1,i] = a
                    self._A[i,i+1] = a
            return self._A


    @property
    def offdiagmax(self):
        self._idx = np.zeros(2)
        max = 0
        for i in range(self._size):
            for j in range(i+1, self._size):
                aij = np.absolute(self._A[i,j])
                if (aij > max):
                    max = aij
                    self._idx[0] = i
                    self._idx[1] = j
        return self._idx


    def jacobi(self, l, k):
        '''
        jacobi rotation
        '''
        if (self._A[k,l] != 0.0):
            tau = ( self._A[l,l] - self._A[k,k] ) / (2*self._A[k,l])
            if (tau >= 0):
                t = 1.0/( tau + np.sqrt(1.0 + tau*tau) )
            else:
                t = -1.0/( -tau + np.sqrt(1.0 + tau*tau) )
            c = 1/np.sqrt(1+t*t)
            s = c*t
        else:
            c = 1.0
            s = 0.0

        a_kk = self._A[k,k]
        a_ll = self._A[l,l]
        self._A[k,k] = c*c*a_kk - 2.0*c*s*self._A[k,l] + s*s*a_ll;
        self._A[l,l] = s*s*a_kk + 2.0*c*s*self._A[k,l] + c*c*a_ll;
        self._A[k,l] = 0.0; # hard-coding non-diagonal elements by hand
        self._A[l,k] = 0.0; # ------------""--------------
        for i in range(self._size):
            if ((i != k) and (i != l)):
                a_ik = self._A[i,k]
                a_il = self._A[i,l]
                self._A[i,k] = c*a_ik - s*a_il
                self._A[k,i] = self._A[i,k]
                self._A[i,l] = c*a_il + s*a_ik
                self._A[l,i] = self._A[i,l]

            r_ik = self._R[i,k]
            r_il = self._R[i,l]

            self._R[i,k] = c*r_ik - s*r_il
            self._R[i,l] = c*r_il + s*r_ik

        return self._A, self._R

    def jacobi_method(self, maxnondiag = 1.0E8, tol = 1.0E-10, maxiter = 1.0E5):
        '''
        implments jaboci rotation to find eigen value and eigen vector pairs
        '''
        iterations = 0
        while (np.absolute(maxnondiag) > tol and iterations <= maxiter):

            maxind = self.offdiagmax
            p = int(maxind[0])
            q = int(maxind[1])

            maxnondiag = self._A[p,q]

            a, b = self.jacobi(p,q)
            iterations += 1
        return a, b

    def numpy_solve(self):
        eval, evec = np.linalg.eig(obj._A)
        return eval, evec

    # @property
    # def lambda_analytic(self):
    #     h = (self._stop - self._start) / self._size
    #     a = -1 / h**2
    #     d = np.zeros(self._size)
    #     lamb = np.zeros(self._size)
    #     d = np.diag(self._A)
    #
    #     for i in range(self._size):
    #         lamb[i] = d[i] + 2*a*np.cos((i*np.pi)/(self._size+1))
    #     return lamb

# sort results and add end points to eigen vectors
def sort_results(Evalues, Evectors):
    '''
    sorts resulting eigenvalues from smallest to biggest and the eigenvalue matrix to corespond
    '''
    Evalues = np.diag(Evalues)
    permute = Evalues.argsort()
    Evalues = Evalues[permute]
    Evectors = Evectors[:,permute]
    newvec = np.zeros((Evectors.shape[0]+2,Evectors.shape[1]))
    newvec[1:-1,:] = Evectors

    return Evalues, newvec

def time_algo(n, algo = 'jacobi', potential = 'no pot', start = 0, stop = 0, maxnondiag = 1.0E8, tol = 1.0E-10, maxiter = 1.0E5):
    '''
    times the alogrithm as it solves the eigen values for matrixes of size 2x2 to nxn
    we have skipped the trivial 1x1 matrix case
    takes algo arguments of 'jacobi' and 'numpy'
    '''
    timevec = np.zeros(n-1)
    if (algo == 'jacobi'):
        for i in range(2, n+1):
            obj = Rotation_Object(i, potential, start, stop)
            obj.build
            t0 = time.time()
            obj.jacobi_method(maxnondiag, tol, maxiter)
            t1 = time.time()
            timevec[i-2] = t1-t0
    elif (algo == 'numpy'):
        for i in range(2, n+1):
            obj = Rotation_Object(i, potential, start, stop)
            obj.build
            t0 = time.time()
            np.linalg.eig(obj._A)
            t1 = time.time()
            timevec[i-2] = t1-t0
    return timevec



if __name__ == '__main__':


    #time algorithms
    print('Time for Jacobi')
    a = time_algo(15, algo = 'jacobi')
    print(a, '\n')
    print('Time for numpy')
    b = time_algo(15, algo = 'numpy')
    print(b, '\n')

    #wave functions for non interacting potnential for two particle
    noninteract = Rotation_Object(50, potential = 'non interact', omega = .1, start = 0, stop = 20)
    noninteract.build
    xpoints = noninteract.xvec
    c, d = noninteract.jacobi_method()
    eval, evec = sort_results(c, d)

    #wave functions for potential including coulomb potential
    interact = Rotation_Object(50, potential = 'repulsive', omega = .1, start = 0, stop = 20)
    interact.build
    e, f = interact.jacobi_method()
    eval1, evec1 = sort_results(e, f)


    fig = plt.figure()
    plt.plot(xpoints, evec[:,0], label = 'non-interacting particles' )
    plt.plot(xpoints, evec1[:,0], label = 'interacting particles')
    plt.legend()
    plt.show()



        # np.set_printoptions(precision=2)
        # print('Evalues are: \n', myval, '\n')
        # print('Evectors are: \n', myvec, '\n')
        # print('np Evalues are: \n', npval, '\n')
        # print('np Evectors are: \n', npvec, '\n')
        #print('True Evalues: \n', analytical, '\n')
        # print('Iterations: \n', iterations, '\n')
        # print('size to iteration ratio: \n', iterations/obj._size, '\n')










    #
