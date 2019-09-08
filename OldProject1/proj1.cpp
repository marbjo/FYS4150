//c++ -o myprogram.exe  myprogram.cpp -larmadillo
// ./myprogram.exe

#include <iostream>
#include <fstream>
#include <armadillo>
#include <cmath>
#include <iomanip>
#include <string>

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
    vec F2 = Col<double>(n); //Analytical function values
    vec x = Col<double>(n); //X-vector for evaluation and plotting
    double h = 1.0/(n+1);
    double hh = h*h; //Precomputing h^2

    //Filling source function and exact solution matrices
   for(int i=0; i<=n-1; i++){
      x(i) = (i+1)*h;
      F(i) = hh*f( x(i) );
      F2(i) = exact( x(i) );
   }

   //Forward substitution
  for(int j=1; j<=(n-1); j++){
      B(j) = B(j) - A(j-1)*C(j-1)/B(j-1);
      F(j) = F(j) - A(j-1)*F(j-1)/B(j-1);
   }

   //Normalizing diagonal
  for(int k=0; k<=n-2; k++){
       F(k) = F(k)/B(k);
       C(k) = C(k)/B(k);
    }
    //Normalizing last row
    F(n-1) = F(n-1)/B(n-1);
    B.fill(1.0);

    //Backwards substitution
  for(int k=(n-1); k>=1; k--){
       F(k-1) = ( F(k-1) - C(k-1)*F(k) )/B(k-1);
    }
    //Writing to text file and giving n as title
    ofstream myfile;
    int name = int(n);
    myfile.open(to_string(name));

    //myfile << setw(15) << "x" ;
    //myfile << setw(15) << "Computed values" ;
    //myfile << setw(15) << "Exact values" ;
    //myfile << setw(15) << "Relative error" << "\n";

      for (int i = 0; i <= n-1; i++) {
	  //double xval = x[i];
 	  double RelativeError = fabs((F2(i)-F(i))/F2(i));
      //myfile << setw(15) << setprecision(8) << x(i);
      //myfile << setw(15) << setprecision(8) << F(i);
      //myfile << setw(15) << setprecision(8) << F2(i);
      //myfile << setw(15) << setprecision(8) << RelativeError << "\n";
      myfile << setprecision(8) << x(i) << ",";
      myfile << setprecision(8) << F(i) << ",";
      myfile << setprecision(8) << F2(i) << ",";
      myfile << setprecision(8) << RelativeError << "\n";
      }
      myfile.close();
      //delete [] x; delete [] A; delete [] B; delete [] C; delete [] F; delete [] F;
  return 0;
}
//Dynamic memory allocation, pointers?
