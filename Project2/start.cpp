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
    A(0,1) = a;

    //Filling the 3 diagonals
    for(int i=1; i<=n-2; i++){
       A(i,i) = d;
       A(i,i-1) = a;
       A(i,i+1) = a;
    }

    //Filling the last row
    A(n-1,n-1) = d;
    A(n-1,n-2) = a;
    //cout << "Columns: " << A.n_cols << " Rows: " << A.n_rows << endl;
    //cout << A << endl;
    return A;
}


std::tuple<arma::mat, arma::mat> Jacobi_Rotate(arma::mat A, arma::mat R, int k, int l, int n){
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

    //     //c = cos(theta), s = sin(theta), t = tan(theta),
    //     B(i,i) = A(i,i); // i =/= k, i =/= l
    //     B(i,k) = A(i,k)*c - A(i,l)*s; //i =/= k, i =/= l
    //     B(i,l) = A(i,l)*c + A(i,k)*s; //i =/= k, i =/= l
    //     B(k,k) = A(k,k)*c*c - 2*A(k,l)*c*s + A(l,l)*s*s;
    //     B(l,l) = A(l,l)*c*c + 2*A(k,l)*c*s + A(k,k)*s*s;
    //     B(k,l) = ( A(k,k) - A(l,l) )*c*s + A(k,l)*(c*c - s*s);

    double a_kk, a_ll, a_ik, a_il, r_ik, r_il;
    a_kk = A(k,k);
    a_ll = A(l,l);
    A(k,k) = c*c*a_kk - 2.0*c*s*A(k,l) + s*s*a_ll;
    A(l,l) = s*s*a_kk + 2.0*c*s*A(k,l) + c*c*a_ll;
    A(k,l) = 0.0; // hard-coding non-diagonal elements by hand
    A(l,k) = 0.0; // same here
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


int main(int argc, char const *argv[])
{
    int n = atoi(argv[1]);
    //arma::mat R(n,n, arma::fill::zeros);
    arma::mat A = CreateMatrix(n);
    arma::mat R = arma::mat(n,n,arma::fill::eye);

    //A.print();

    double tolerance = 1.0E-10;

    cout << tolerance << endl;
    int iterations = 0;
    double maxiter = 1.0E5;
    double maxnondiag = 1.0E8;
    while (fabs(maxnondiag) > tolerance && iterations <= maxiter){

        int* max_ind = offdiagmax(A,n);
        int p = max_ind[0];
        int q = max_ind[1];
        maxnondiag = A(p,q);

        //Jacobi_Rotate(A, R, p, q, n);
        std::tie(A, R) = Jacobi_Rotate(A, R, p, q, n);
        //cout << "One iteration!" << endl;

        iterations++;
    }

    arma::vec eigval;
    arma::mat eigvec;

    arma::eig_sym(eigval, eigvec, A);

    cout << "This is A: "<< "\n" << endl;
    A.print();

    cout << "\n" <<"This is armadillo's eigenvalues: "<< "\n" << endl;
    eigval.print();

    cout << "This is R: "<< "\n" << endl;
    R.print();

    cout << "\n" <<"This is armadillo's eigenvectors: "<< "\n" << endl;
    eigvec.print();

    return 0;
}
