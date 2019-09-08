// my first program in C++

//c++ -o myprogram.exe  myprogram.cpp -larmadillo
// ./myprogram.exe

#include <iostream>
#include <fstream>
#include <armadillo>
#include <cmath>
#include <iomanip>
//#include <string>

using namespace std;
using namespace arma;

// object for output files
ofstream ofile;

inline double f(double x){return 100.0*exp(-10.0*x);
}
inline double exact(double x) {return 1.0-(1.0-exp(-10.0))*x-exp(-10.0*x);}

int main(int argc, char* argv[])
  //Function takes exponent as command line argument, i.e. n = 10^x, x given by user
  {
    double n = pow(10.0, atof(argv[1]));
    //Defining matrices and constants
    vec A = Col<double>(n-1);
    A.fill(-1);
    vec B = Col<double>(n);
    B.fill(2);
    vec C = Col<double>(n-1);
    C.fill(-1);
    vec F = Col<double>(n); //Computed function values
    vec u = Col<double>(n); //Final computed values
    vec F2 = Col<double>(n); //Analytical function values
    vec x = Col<double>(n); //X-vector for evaluation and plotting
    vec ifactor1 = Col<double>(n);
    vec ifactor2 = Col<double>(n);
    double h = 1.0/(n+1);
    double hh = h*h; //Precomputing h^2

    //Filling source function and exact solution matrices
   for(int i=0; i<=n-1; i++){
      x(i) = (i+1)*h;
      F(i) = hh*f( x(i) );
      F2(i) = exact( x(i) );
   }
   for(double l=0; l<=(n-1); l++){
       ifactor1(l) = (l+1)/(l+2);
       ifactor2(l) = (l+2)/(l+1);
       //cout << ifactor(l) << endl;
   }
   //cout << ifactor <<endl;
   //Forward substitution
  for(int j=1; j<=(n-1); j++){
      F(j) = F(j) + ifactor1(j-1)*F(j-1);
   }

   //Normalizing diagonal
   for(int k=0; k<=n-1; k++){
        F(k) = F(k)/ifactor2(k);
        u(k) = F(k);
     }
   //u(n-1) = F(n-1)*n/(n+1);

    //Backwards substitution
  for(int k=(n-1); k>=1; k--){
       u(k-1) = ifactor1(k) * (F(k-1) + u(k));
    }
    //Writing to text file and giving n as title
    ofstream myfile;
    string name = "Test"; //int(h);
    myfile.open(name);

    //myfile << setw(15) << "x" ;
    //myfile << setw(15) << "Computed values" ;
    //myfile << setw(15) << "Exact values" ;
    //myfile << setw(15) << "Relative error" << "\n";

      for (int i = 0; i <= n-1; i++) {
	  //double xval = x[i];
 	  double RelativeError = fabs((F2(i)-u(i))/F2(i));
      myfile << setw(15) << setprecision(8) << x(i);
      myfile << setw(15) << setprecision(8) << u(i);
      myfile << setw(15) << setprecision(8) << F2(i);
      myfile << setw(15) << setprecision(8) << RelativeError << "\n";
      }
      myfile.close();
      //delete [] x; delete [] A; delete [] B; delete [] C; delete [] F; delete [] F;
  return 0;
}
//Dynamic memory allocation, pointers?
