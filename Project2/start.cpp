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

void offdiagmax(arma::mat A, int n, int& k, int& l){
    //int* pos = new int[2];
    double max = 0;
    for (int i = 0; i < n; ++i){
        for (int j = i+1; j < n; ++j){
            double aij = fabs(A(i,j));
            if (aij > max){
                max = aij;
                k = i;
                l = j;
            }
        }
    }
    //return pos;
}

arma::mat CreateMatrix(int n, arma::mat A){
    //Function for setting up and filling tridiagonal matrix, with command line
    //argument n as dimensionality

    //int mat_size = n;
    //arma::mat A(mat_size,mat_size, arma::fill::zeros);

    double h = 1. / double(n);
    double hh = h*h;
    double d = 2./hh;
    double a = -1./hh;

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

arma::mat CreateMatrixQuantum(int n,double rho_max, arma::mat A){
    //Function for setting up and filling tridiagonal matrix, with command line
    //argument n as dimensionality

    //int mat_size = n;
    //arma::mat A(mat_size,mat_size, arma::fill::zeros);

    //double rho_max = 5;

    double h = rho_max / double(n+1);
    double hh = h*h;
    double d = 2./hh;
    double a = -1./hh;

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

arma::mat CreateMatrixQuantum2(int n,double rho_max, double omega, arma::mat A){
    //Function for setting up and filling tridiagonal matrix, with command line
    //argument n as dimensionality

    //int mat_size = n;
    //arma::mat A(mat_size,mat_size, arma::fill::zeros);

    double h = rho_max / double(n);
    double hh = h*h;
    double d = 2./hh;
    double a = -1./hh;

    //Filling first row

    //rho_i = i*h
    A(0,0) = d + omega*omega*h*h + 1./(h);
    A(0,1) = a;

    //Filling the 3 diagonals
    for(int i=1; i<=n-2; i++){
       A(i,i) = d + omega * omega * (i+1)*h * (i+1)*h + 1./((i+1)*h);

       A(i,i-1) = a;
       A(i,i+1) = a;
    }

    //Filling the last row
    A(n-1,n-1) = d + omega * omega * n*h * n*h + 1./(n*h);
    A(n-1,n-2) = a;

    return A;
}

std::tuple<arma::mat, arma::mat> Jacobi_Rotation(arma::mat A, arma::mat R, int k, int l, int n){
//arma::mat Jacobi_Rotation(arma::mat A, int k, int l, int n){
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
    //return A;
}

std::tuple<arma::mat, arma::mat> Multiple_Jacobi_Rotation(arma::mat A, arma::mat R, int n){
    //Tolerance for accepting an element as zero
    double tolerance = 1.0E-10;

    //Initializing counting variables for while loop
    int iterations = 0;
    double maxiter = 1.0E5;
    double maxnondiag = 1.0E8;

    while (fabs(maxnondiag) > tolerance && iterations <= maxiter){

        //Getting indices of the largest offdiagonal element
        int k  = 0;
        int l = 0;
        offdiagmax(A,n,k,l);
        //int p = max_ind[0];
        //int q = max_ind[1];
        //Assigning the value of the largest offdiagonal element
        maxnondiag = A(k,l);

        //Extracting the tuple returned from Jacobi_Rotate function
        std::tie(A, R) = Jacobi_Rotation(A, R, k, l, n);
        //A = Jacobi_Rotation(A,p,q,n);
        iterations++;
    }
    return std::make_tuple(A, R);
}

void MaxTest(arma::mat test_mat , int n){
    //Unit test to test if the "find largest offdiagonal element"-function is working
    int k = 0;
    int l = 0;
    offdiagmax(test_mat,3,k,l);
    //int p = max_ind[0];
    //int q = max_ind[1];
    double maxnondiag_test = test_mat(k,l);
    if( maxnondiag_test == test_mat.max() ){
        cout << "Max offdiagonal unit test passed!" << endl;
    }
    else{
        cout << "Max offdiagonal unit test NOT passed!" << endl;
        exit(1);
    }

}

//void OrthogonalTest(arma::mat A_test, arma::vec R_test, int n_test){}

int main(int argc, char const *argv[]){
    //Reading dimensionality,omega and 0/1/2, whether you want buckling beam, one or two electrons respectively, as command line argument
    if (argc < 4) {
        cout << "Error: missing command line arguments. Must provide dimensionality n, omega and 0,1,2 for running Buckling Beam, one or two quantum dots respectively" << endl;
        return 1;
    }

    int n = atoi(argv[1]);
    double omega = stod(argv[2]);
    int pot_check = atoi(argv[3]);
    //double rho_max = 8.0;//4.78;
    double rho_max = stod(argv[4]);

    //Creating matrices A and R. A is filled through the create matrix function
    arma::mat A(n,n, arma::fill::zeros);
    arma::mat R = arma::mat(n,n,arma::fill::eye);

    //Basically just for testing
    arma::mat R_test = arma::mat(n,n,arma::fill::eye);

    if(pot_check == 0){
        //Creating matrix for Buckling Beam
        A = CreateMatrix(n,A);
    }
    else if(pot_check == 1){
        //Creating matrix for quantum dots
        A = CreateMatrixQuantum(n,rho_max,A);
    }
    else if(pot_check == 2){
        //Creating matrix for two quantum dots
        A = CreateMatrixQuantum2(n,rho_max,omega,A);
    }
    else{
        cout << "Must give 0,1,2 to choose Buckling beam, one or two quantum dots" << endl;
        return 0;
    }

    //Unit tests
    arma::mat test_mat = {{1,2,3},{4,5,6.},{1,2,3}};
    MaxTest(test_mat, test_mat.n_cols);


    //Computing Armadillo eigenpairs for comparison
    arma::vec eigval;
    arma::mat eigvec;
    arma::eig_sym(eigval, eigvec, A);

    //Performing Jacobi rotations on the matrix until diagonal
    std::tie(A, R) = Multiple_Jacobi_Rotation(A, R, n);

    //Saving eigenvector matrix to file with appropriate name
    //Coulomb interaction case will contain value for omega in name
    string name;
    if(pot_check == 0){
        name = "_buckling";
    }
    else if(pot_check == 1){
        name = "_quantum_dot";
    }
    else if(pot_check == 2){
        name = "_Coulomb_" + to_string(omega);
    }
    string Rname = "R" + name;
    R.save(Rname, arma::raw_ascii);

    //Extracting eigenvalues to a vector and sorting it
    arma::vec eig_comp(n);
    for(int l=0; l<n; l++){
        eig_comp(l) = A(l,l);
    }

    //Saving a vector of the eigenvalues (not sorted) to file
    string eigname = "Eig" + name;
    eig_comp.save(eigname,arma::raw_ascii);

    //Computing error between computed and analytical eigvalues for first 4.
    //This was basically done to find the best rho_max
    arma::vec eig_comp_sort = arma::sort(eig_comp);
    double err_tot = 0;

    //Printing first four Eigenvalues for checking.
    cout << "First four Eigenvalues: " << endl;
    cout << eig_comp_sort(0) << "\n" << eig_comp_sort(1) << "\n" << eig_comp_sort(2) << "\n"  << eig_comp_sort(3) << endl;

    for(int o=0; o<4; o++){
        err_tot = err_tot + fabs(eig_comp_sort(o) - eigval(o));
    }

    cout << "Total error: " << err_tot << endl;

    return 0;
}
