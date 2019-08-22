// my first program in C++

//c++ -o myprogram.exe  myprogram.cpp -larmadillo
// ./myprogram.exe

#include <iostream>
#include <armadillo>
#include <cmath>

using namespace std;
using namespace arma;

inline double f(double x){return 100.0*exp(-10.0*x);
}
inline double exact(double x) {return 1.0-(1-exp(-10))*x-exp(-10*x);}

int main(int argc, char* argv[])
  //int argc, char** argv
  {
    //Defining matrices and constants
    double n = atof(argv[1]);
    //double n = 10;
    //mat A(n-1,1);
    //A.fill(-1);
    //mat B(n,1);
    //B.fill(2);
    //mat C(n-1,1);
    //C.fill(-1);
    //mat F((n),1); //Array for computed values
    //mat F2((n),1); //Array for exact values
    vec A = Col<double>(n-1);
    A.fill(-1);
    vec B = Col<double>(n);
    B.fill(2);
    vec C = Col<double>(n-1);
    C.fill(-1);
    vec F = Col<double>(n);
    //F.print();
    vec F2 = Col<double>(n);
    double h = 1/(n);
    double hh = h*h;
    //cout<<hh<<endl;
    //return 1;
    //Filling function value and exact solution matrices
   for(int i=0; i<=n-1; i++){
      double x = (i+1)*h;
      //cout<<"X is: "<<x<<endl;
      F(i) = hh*f(x);
      F2(i) = exact(x);
   }
   //F.print();
   //F2.print();
   //return 1;
   //Boundary conditions
   //F(0) = 0
   //F(n+2) = 0
   //F2(0) = exact(0)
   //F2(n+2) = exact(1)
   //Forward substitution
  for(int j=1; j<=n-1; j++){
      B(j) = B(j) - A(j-1)*C(j-1)/B(j-1);
      F(j) = F(j) - A(j-1)*F(j-1)/B(j-1);
   }
   //cout << "This is ok 1."<<endl;
   //Normalizing diagonal
  for(int k=0; k<=n-2; k++){
       F(k) = F(k)/B(k);
       C(k) = C(k)/B(k);
    }
    //Normalizing last row
    F(n-1) = F(n-1)/B(n-1);

    //cout << "This is ok 2."<<endl;
    //Backwards substitution
  for(int k=(n-1); k>=1; k--){
       F(k-1) = ( F(k-1) - C(k-1)*F(k) ) / B(k-1);
    }
    //cout << "This is ok 3."<<endl;
    //Printing matrices for comparison
    //cout<<"Computed: "<<F<<endl;
    //cout<<"Exact: "<<F2<<endl;
    double mean_error = mean(abs(F2-F)); // /n;
    cout<<"Error: " << mean_error <<endl;
  return 0;
}
