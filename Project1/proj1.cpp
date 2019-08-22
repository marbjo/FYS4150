// my first program in C++

//c++ -o myprogram.exe  myprogram.cpp
// ./myprogram.exe

#include <iostream>
#include <armadillo>
#include <cmath>

using namespace std;
using namespace arma;

inline double f(double x){return 100.0*exp(-10.0*x);
}
inline double exact(double x) {return 1.0-(1-exp(-10))*x-exp(-10*x);}

int main()
  //int argc, char** argv
  {
    //Defining matrices and constants
    double n = 10;
    mat A(n-1,1);
    A.fill(-1);
    mat B(n,1);
    B.fill(2);
    mat C(n-1,1);
    C.fill(-1);
    mat F(n,1);
    double h = 1/(n+1);
    double hh = h*h;
    mat F2(n,1);


    //Filling function value and exact solution matrices
   for(int i=1; i==n-1; i++){
      F(i) = hh*f(i*h);
      double x = h*i;
      F2(i) = exact(x);
   }
//Forward substitution
  for(int j=1; j==(n-1); j++){
      B(j) = B(j) - A(j-1)*C(j-1)/B(j-1);
      F(j) = F(j) - A(j-1)*F(j-1)/B(j-1);
   }
   //Normalizing diagonal
  for(int j=0; j==n; j++){
       F(j) = F(j)/B(j);
       C(j) = C(j)/B(j);
    }
    //Backwards substitution
  for(int k=n; k==1; k--){
       F(k-1) = ( F(k-1) - C(k-1)*F(k) ) / B(k-1);
    }
    //Printing matrices for comparison
    cout<<"Computed: "<<F<<endl;
    cout<<"Exact: "<<F2<<endl;
  return 0;
}
