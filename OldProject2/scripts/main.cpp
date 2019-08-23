#include <iostream>
#include <cmath>
#include <armadillo>


using namespace std;
using namespace arma;


// Constructing tridiagonal Toeplitz matrix A
arma::mat makeTri(double rho_0, double rho_n, int n){

	// Initializing empty matrix A
	arma::mat A = arma::mat(n,n);
	A.zeros();

	// Steplength calculated from given rho values and number of gridpoints
	double h = (rho_n - rho_0)/(double)n;

	// Second derivative diagonal constants
	double d = 2/(h*h);
	double a = -1/(h*h);

	// Filling the tridiagonal matrix
	for (int i = 0; i < n; i++){

		// Main diagonal
		A(i,i) = d;

		// Secondary diagonals
		if (i < n-1){
			A(i,i+1) = a;
			A(i+1,i) = a;
		}
	}

	return A;
}
