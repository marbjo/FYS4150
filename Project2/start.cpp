//This is the start of project 2

//c++ -o myprogram.exe  myprogram.cpp -larmadillo
// ./myprogram.exe

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <string>
#include <ctime>
#include <armadillo>

using namespace std;
//using namespace arma;

void CreateMatrix(int n)
{
    //Function for setting up and filling tridiagonal matrix, with command line
    //argument n as dimensionality
    
    int mat_size = n;
    arma::mat A(mat_size,mat_size, arma::fill::zeros);

    double h = 1/float(n);
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
    cout << A << endl;
}


int main(int argc, char const *argv[])
{
    int n = atoi(argv[1]);
    CreateMatrix(n);
    return 0;
}
