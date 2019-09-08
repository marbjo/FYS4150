#include "armadillo"
#include <iostream>
#include <fstream>
#include <ctime>
#include <iomanip>
#include <string>
//


//must link when compiling: ex g++ file.cpp -o file -larmdaillo

using namespace std;
using namespace arma;

double exact(double x) {
  return 1.0-(1.0-exp(-10.0))*x-exp(-10.0*x);
}
double relative_error (double computed, double exact) {
  return fabs((computed-exact)/exact);
}
double fsource(double x) {
  return 100.0*exp(-10.0*x);
}
void armadillo_solve(int exponent) {

  ofstream newfile_2;
  ofstream timefile_2;
  vec time_vector = vec(exponent);

  for (int j = 1; j < exponent+1; j++){
    //makes file out name coresspond to size n
    string name2("armadillo_solver");
    string size = to_string(j);
    name2.append("_" + size + "_.txt");

    //create step size parameters
    int n = (int) pow(10.0,j);
    int mat_size = n-1;
    vec x = vec(n-1); vec b = vec(n-1); vec solution = vec(n-1); vec error = vec(n-1);

    //fill in points vector x[i] and B_hat vector, throwing away the first and last element
    //since those are given conditions
    //int n = (int) pow(10.0,i);
    double h = 1.0/(n);
    double h_square = h*h;

    for (int i = 1; i < n; i++) {
      x(i-1) = i*h;
      b(i-1) = h_square*fsource(i*h);
    }

    //setup vectors and matrices
    mat A = zeros<mat>(mat_size,mat_size);
    A(0,0)=2;
    A(0,1)=-1;
    A(mat_size-1,mat_size-1)=2;
    A(mat_size-1,mat_size-2)=-1;

    for (int i=1; i<mat_size-1; i++){
      A(i,i)=2;
      A(i,i+1)=-1;
      A(i,i-1)=-1;
    }
    // cout << A.n_rows<<"-";
    // cout << A.n_cols<<endl;

    clock_t start, finish;
    start = clock();
    mat L, U;
    lu(L, U, A);
    vec q = solve(L, b);
    solution = solve(U, q);
    finish = clock();
    time_vector(j-1)=((double) (finish-start))/CLOCKS_PER_SEC;
    // cout<<time_vector.n_rows;
    // cout << time_vector;
    for (int i=0; i < n-1; i++) {
      double xval= x(i);
      error(i)= relative_error(solution(i),exact(xval));
    }

    //write important vectors to a file
    newfile_2.open(name2);
    newfile_2 << setiosflags(ios::showpoint | ios::uppercase);
    for (int i = 0; i < n-1; i++) {
      double xval= x(i);
      newfile_2 << setw(15) << setprecision(8) << xval; //parameterized x values
      newfile_2 << setw(15) << setprecision(8) << solution(i);
      newfile_2 << setw(15) << setprecision(8) << exact(xval);
      newfile_2 << setw(15) << setprecision(8) << error(i) << endl;
    }
    newfile_2.close();
  }
  //send time values to a file
  string s1("armadillo_solver_time.txt");
  timefile_2.open(s1);
  timefile_2 << setiosflags(ios::showpoint | ios::uppercase);
  for (int i=0; i<exponent; i++) {
    timefile_2 << setw(15) << setprecision(8) << time_vector(i) << endl;
  }
  timefile_2.close();
}

int main(int argc, char const *argv[]) {
  int exponent_val;
  if( argc != 2 ){
    cout << "ERROR: \nEnter command line as ./my_solver n \nfor a matrix of 10^n gridpoints" << endl;
    exit(1);
  }
  else{
    exponent_val= atoi(argv[1]);
  }
  armadillo_solve(exponent_val);
return 0;
}
