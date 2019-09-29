import numpy as np
import matplotlib.pyplot as plt
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
        return a, b, iterations

    def numpy_solve(self):
        eval, evec = np.linalg.eig(self._A)
        return eval, evec

class Test_object():

    def __init__(self):
        self._size = 4
        self._R = np.eye(4)

    @property
    def build_test(self):
        '''
        build a matrix to be used for unit tests
        Max off diagonal is [0,3]
        '''
        self._A = np.zeros((4,4))
        for i in range(4):
            self._A[i,i]=2
        self._A[0,3] = self._A[3,0] = 10
        self._A[2,1] = self._A[1,2] = 1
        self._A[3,2] = self._A[2,3] = 3
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
        return a, b, iterations

    def numpy_solve(self):
        eval, evec = np.linalg.eig(self._A)
        return eval, evec

def sort_results(Evalues, Evectors, algo):
    '''
    sorts resulting eigenvalues from smallest to biggest and the eigenvalue matrix to corespond
    '''
    if (algo == 'jacobi'):
        Evalues = np.diag(Evalues)
        permute = np.argsort(Evalues)
        Evalues = Evalues[permute]
        Evectors = Evectors[:,permute]
        newvec = np.zeros((Evectors.shape[0]+2,Evectors.shape[1]))
        newvec[1:-1,:] = Evectors

    if (algo == 'numpy'):
        permute = np.argsort(Evalues)
        Evalues = Evalues[permute]
        Evectors = Evectors[:,permute]
        newvec = np.zeros((Evectors.shape[0]+2,Evectors.shape[1]))
        newvec[1:-1,:] = Evectors

    return Evalues, newvec

def time_algo(n, rho_min, rho_max, algo = 'jacobi', potential = 'no pot', maxnondiag = 1.0E8, tol = 1.0E-10, maxiter = 1.0E5):
    '''
    times the alogrithm as it solves the eigen values for matrixes of size 2x2 to nxn
    we have skipped the trivial 1x1 matrix case
    takes algo arguments of 'jacobi' and 'numpy'
    '''
    timevec = np.zeros(n-1)
    nvec = np.zeros(n-1)
    if (algo == 'jacobi'):
        for i in range(2, n+1):
            obj = Rotation_Object(i, potential, start = rho_min, stop = rho_max)
            obj.build
            t0 = time.time()
            obj.jacobi_method(maxnondiag, tol, maxiter)
            t1 = time.time()
            timevec[i-2] = t1-t0
            nvec[i-2] = i
    elif (algo == 'numpy'):
        for i in range(2, n+1):
            obj = Rotation_Object(i, potential, start = rho_min, stop = rho_max)
            obj.build
            t0 = time.time()
            np.linalg.eig(obj._A)
            t1 = time.time()
            timevec[i-2] = t1-t0
            nvec[i-2] = i
    return timevec, nvec

def one_particle_accuracy(n, rho_min, rho_max, algo = 'jacobi', maxnondiag = 1.0E8, tol = 1.0E-10, maxiter = 1.0E5):

    if (algo == 'jacobi'):
        obj = Rotation_Object(n, 'hamiltonian', start = rho_min, stop = rho_max)
        obj.build
        xpoints = obj.xvec
        eval, evec, _ = obj.jacobi_method(maxnondiag, tol, maxiter)
        sorted_val, sorted_vec = sort_results(eval, evec, algo = 'jacobi')
        return sorted_val, sorted_vec, xpoints

    if (algo == 'numpy'):
        obj = Rotation_Object(n, 'hamiltonian', start = rho_min, stop = rho_max)
        obj.build
        xpoints =obj.xvec
        eval, evec = obj.numpy_solve()
        sorted_val, sorted_vec = sort_results(eval, evec, algo = 'numpy')
        return sorted_val, sorted_vec, xpoints

def diagonal_unit_test():
    A = Test_object()
    A.build_test
    max_index = A.offdiagmax
    if (max_index[0] != 0 and max_index[1] != 3):
        print('Max index finder not function, test failed')
        exit()
    else:
        print('Max index finder working')

def jacobi_unit_test():

    eps = 1e-8

    A = Test_object()
    A.build_test
    a, b, _ = A.jacobi_method()
    eval, evec = sort_results(a, b, 'jacobi')

    B = Test_object()
    B.build_test
    c, d = A.numpy_solve()
    eval1, evec1 = sort_results(c, d, 'numpy')

    t1 = evec[0]
    t2 = evec1[0]

    for i in range(4):
        if (np.absolute(t1[i]-t2[i]) > eps):
            print('Jacobi rotation algorithm not functioning')
            exit()
    print('Jacobi algorithm is finding correct Eigenvectors')

    if (np.dot(evec[0], evec[1]) > eps):
        print('Jacobi rotation algorithm not functioning')
        exit()
    else:
        print('Jacobi algorithm Eigenvectors are orthogonal')


if __name__ == '__main__':

    unit_test = False
    do_interact = False
    do_noninteract = False
    do_interact_numpy = False
    do_time = False
    plot_compare = False
    eval_accuracy = False

    if (unit_test == True):

        diagonal_unit_test()
        jacobi_unit_test()

    if (do_time == True):
        #time algorithms
        print('Time for Jacobi')
        a, b = time_algo(30, 0, 1, algo = 'jacobi')
        print(a, '\n')
        print('Time for numpy')
        c, _ = time_algo(30, 0, 1, algo = 'numpy')
        print(c, '\n')
        fig = plt.figure()
        plt.plot(b**2, a, label = 'Jacobi Time')
        plt.plot(b**2, c, label = 'Numpy.linalg.eig Time')
        plt.xlabel('Grid points N^2')
        plt.ylabel('Time in Seconds')
        plt.legend()
        plt.show()

    if (do_noninteract == True):
        #wave functions for non interacting potnential for two particle
        noninteract = Rotation_Object(100, potential = 'non interact', omega = .01, start = 0, stop = 50)
        noninteract.build
        xpoints = noninteract.xvec
        c, d, _ = noninteract.jacobi_method()
        eval, evec = sort_results(c, d, 'jacobi')

    if (do_interact == True):
        #wave functions for potential including coulomb potential
        interact = Rotation_Object(60, potential = 'repulsive', omega = .01, start = 0, stop = 35)
        interact.build
        xpoints = interact.xvec
        e, f, _ = interact.jacobi_method()
        oval, ovec = sort_results(e, f, 'jacobi')

        interact1 = Rotation_Object(60, potential = 'repulsive', omega = .5, start = 0, stop = 35)
        interact1.build
        e, f, _ = interact1.jacobi_method()
        oval1, ovec1 = sort_results(e, f, 'jacobi')

        interact2 = Rotation_Object(60, potential = 'repulsive', omega = 1, start = 0, stop = 35)
        interact2.build
        e, f, _ = interact2.jacobi_method()
        oval2, ovec2 = sort_results(e, f, 'jacobi')

        interact3 = Rotation_Object(60, potential = 'repulsive', omega = 5, start = 0, stop = 35)
        interact3.build
        e, f, _ = interact3.jacobi_method()
        oval3, ovec3 = sort_results(e, f, 'jacobi')

    if (do_interact_numpy == True):
        obj = Rotation_Object(50, potential = 'repulsive', omega = .01, start = 0, stop = 50)
        obj.build
        xpoints = obj.xvec
        g, h = np.linalg.eig(obj._A)
        eval2, evec2 = sort_results(g, h, 'numpy')
        print('Eval jacobi= ', eval2)
        print('Evec jacobi= ', evec2[:,0])

    if (plot_compare == True):
        fig = plt.figure()
        # plt.plot(xpoints,evec[:,0], label = 'non-interacting particles' )
        plt.plot(xpoints,ovec[:,0]**2, label = 'omega = 0.01')
        plt.plot(xpoints,ovec1[:,0]**2, label = 'omega = 0.5')
        plt.plot(xpoints,ovec2[:,0]**2, label = 'omega = 1')
        plt.plot(xpoints,ovec3[:,0]**2, label = 'omega = 5')

        # plt.plot(xpoints,evec2[:,0]**2, label = 'numpy interacting particles')
        plt.legend()
        plt.show()

    if (eval_accuracy == True):
        t0 = time.time()
        a, b, c = one_particle_accuracy(75, 0, 5, algo = 'jacobi', maxnondiag = 1.0E8, tol = 1.0E-10, maxiter = 1.0E5)
        t1 = time.time()
        np.set_printoptions(precision = 5)
        print(a[0:5])
        print('time= ', t1-t0)
        fig = plt.figure()
        plt.plot(c, b[:,0])
        plt.show()











    #
