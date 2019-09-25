//This is the start of project 2

//-O3 compiler flag for vectorization
//c++ -O3 -o myprogram.exe  myprogram.cpp -larmadillo
// ./myprogram.exe 10

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <string>
#include <ctime>
#include <armadillo>

using namespace std;
///using namespace arma;

int* offdiagmax(arma::mat A, int n){
    int* pos = new int[2];
    double max = 0;
    for (int i = 0; i < n; ++i){
        for (int j = i+1; j < n; ++j){
            double aij = fabs(A(i,j));
            if (aij > max){
                max = aij;
                pos[0] = i;
                pos[1] = j;
            }
        }
    }
    return pos;
}

arma::mat CreateMatrix(int n){
    //Function for setting up and filling tridiagonal matrix, with command line
    //argument n as dimensionality

    int mat_size = n;
    arma::mat A(mat_size,mat_size, arma::fill::zeros);

    double h = 1. / n;
    double hh = h*h;
    double d = 2/hh;
    double a = -1/hh;

    //Filling first row
    A(0,0) = d;
    //A(0,0) = d + h;
    A(0,1) = a;

    //Filling the 3 diagonals
    for(int i=1; i<=n-2; i++){
       A(i,i) = d;
       //A(i,i) = d + (i+1)*h;

       A(i,i-1) = a;
       A(i,i+1) = a;
    }

    //Filling the last row
    A(n-1,n-1) = d;
    //A(n-1,n-1) = d + n*h;
    A(n-1,n-2) = a;

    return A;
}

arma::mat CreateMatrixQuantum(int n,double rho_max){
    //Function for setting up and filling tridiagonal matrix, with command line
    //argument n as dimensionality

    int mat_size = n;
    arma::mat A(mat_size,mat_size, arma::fill::zeros);

    //double rho_max = 5;

    double h = rho_max / n;
    double hh = h*h;
    double d = 2/hh;
    double a = -1/hh;

    //Filling first row
    A(0,0) = d + h*h;
    A(0,1) = a;

    //Filling the 3 diagonals
    for(int i=1; i<=n-2; i++){
       A(i,i) = d + (i+1)*h * (i+1)*h;

       A(i,i-1) = a;
       A(i,i+1) = a;
    }

    //Filling the last row
    A(n-1,n-1) = d + n*h * n*h;
    A(n-1,n-2) = a;

    return A;
}

std::tuple<arma::mat, arma::mat> Jacobi_Rotation(arma::mat A, arma::mat R, int k, int l, int n){
    double s;
    double c;
    if ( A(k,l) != 0.0 ){
        double t;
        double tau = ( A(l,l) - A(k,k) ) / (2*A(k,l));

        if ( tau >= 0 ){
            t = 1.0/( tau + sqrt(1.0 + tau*tau) );
        }
        else {
            t = -1.0/( -tau + sqrt(1.0 + tau*tau) );
        }

        c = 1/sqrt(1+t*t);
        s = c*t;

    }
    else{
        c = 1.0;
        s = 0.0;
    }

    double a_kk, a_ll, a_ik, a_il, r_ik, r_il;
    a_kk = A(k,k);
    a_ll = A(l,l);
    A(k,k) = c*c*a_kk - 2.0*c*s*A(k,l) + s*s*a_ll;
    A(l,l) = s*s*a_kk + 2.0*c*s*A(k,l) + c*c*a_ll;
    A(k,l) = 0.0; // hard-coding non-diagonal elements by hand
    A(l,k) = 0.0; // ------------""--------------

    for(int i = 0; i < n; i++ ){
        if( i != k && i != l ) {
            a_ik = A(i,k);
            a_il = A(i,l);
            A(i,k) = c*a_ik - s*a_il;
            A(k,i) = A(i,k);
            A(i,l) = c*a_il + s*a_ik;
            A(l,i) = A(i,l);
        }
        r_ik = R(i,k);
        r_il = R(i,l);

        R(i,k) = c*r_ik - s*r_il;
        R(i,l) = c*r_il + s*r_ik;
    }
    return std::make_tuple(A, R);
}

int main(int argc, char const *argv[]){
    //Reading dimensionality and rho_max as command line argument
    int n = atoi(argv[1]);
    //int rho_max = atoi(argv[2]);

    int rho_max = 10;
    //Creating matrices A and R, A is filled through the create matrix function

    //Creating matrix for Buckling Beam
    //arma::mat A = CreateMatrix(n);

    //Creating matrix for quantum dots
    arma::mat A = CreateMatrixQuantum(n,rho_max);

    arma::mat R = arma::mat(n,n,arma::fill::eye);

    //Tolerance for accepting an element as zero
    double tolerance = 1.0E-10;

    //Initializing counting variables for while loop
    int iterations = 0;
    double maxiter = 1.0E5;
    double maxnondiag = 1.0E8;

    while (fabs(maxnondiag) > tolerance && iterations <= maxiter){

        //Getting indices of the largest offdiagonal element
        int* max_ind = offdiagmax(A,n);
        int p = max_ind[0];
        int q = max_ind[1];
        //Assigning the value of the largest offdiagonal element
        maxnondiag = A(p,q);

        //Extracting the tuple returned from Jacobi_Rotate function
        std::tie(A, R) = Jacobi_Rotation(A, R, p, q, n);

        iterations++;
    }

    //Computing Armadillo eigenpairs for comparison
    arma::vec eigval;
    arma::mat eigvec;

    arma::eig_sym(eigval, eigvec, A);

    cout << "These are the first computed eigenvalues: "<< "\n" << endl;
    cout << A(0,0) << "\n" << A(1,1) << "\n" << A(2,2) << "\n" << A(3,3) << "\n" << A(4,4) << "\n" << A(5,5) << endl;

    //cout << "\n" <<"Dimension n: "<< n << endl;
    cout <<"Number of iterations in Jacobi rotation: "<< iterations << endl;
    //cout <<"Ratio iterations/n: "<< iterations/float(n) << endl;

    return 0;
}
