#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <string>
#include <ctime>

using namespace std;

//Functions for test and exact solution
double fsource(double x) {
  return 100.0*exp(-10.0*x);
}

double exact(double x) {
  return 1.0-(1.0-exp(-10.0))*x-exp(-10.0*x);
}

double relative_error (double computed, double exact) {
  return fabs((computed-exact)/exact);
}

void gaussian(int exponent) {

  ofstream newfile;
  ofstream timefile;
  ofstream errorfile;

  double *time = new double[exponent];
  double *mean_error = new double[exponent];

  //loop over all 10^n values
  for (int i = 1; i < exponent+1; i++){

    //makes file out name coresspond to size n
    string name("gaussian");
    string size = to_string(i);
    name.append("_" + size + ".txt");

    //create step size parameters
    int n = (int) pow(10.0,i);
    double h = 1.0/(n);
    double h_square = h*h;

    //solving A * X_hat = B_hat
    //create vectors to be used in forward and back sub as well as x[i] points vector
    //c is bottom diagonal row, d is middle, e is top, b is B_hat, solution is X_hat
    double *x = new double[n+1];
    double *c = new double[n];
    double *d = new double[n+1];
    double *e = new double[n];
    double *b = new double[n+1];
    double *solution = new double[n+1];
    double *rel_error = new double[n];

    //fill in points vector x[i] and B_hat vector
    for ( int i = 0; i <= n; i++) {
      x[i] = i*h;
      b[i] = h_square*fsource(i*h);
    }

    //create diagonal vectors, making c/e 1 element shorter than d. Add initial conditions
    for ( int i=0; i < n+1; i++) {d[i]=2.0;}
    for ( int i=0; i < n; i++){
      e[i]=-1.0;
      c[i]=-1.0;
    }
    solution[0]=solution[n]=0;

    //start clock
    clock_t time_req;
    time_req = clock();

    /*forward substitution*/
    for (int i = 1; i < n-1; i++) {
      d[i+1] = d[i+1] - (c[i]/d[i])*e[i];
      b[i+1] = b[i+1] - (c[i]/d[i])*b[i];
    }
    /*backward substitution*/
    solution[n-1]=b[n-1]/d[n-1];
    for (int i = n-2; i>0; i--) {
      solution[i]=(b[i]-e[i]*solution[i+1])/d[i];
    }

    //stop clock
    time_req = clock()-time_req;
    time[i]=((double) time_req)/CLOCKS_PER_SEC;

    //relative error
    for (int i=1; i < n; i++) {
      double xval= x[i];
      rel_error[i]= relative_error(solution[i],exact(xval));
    }

    //put mean error into vector for corresponding n
    mean_error[i] = (rel_error[1]+rel_error[n-1])/2.0;

    //write important vectors to a file
    newfile.open(name);
    newfile << setiosflags(ios::showpoint | ios::uppercase);
    for (int i = 1; i < n; i++) {
      newfile << setw(15) << setprecision(8) << x[i]; //parameterized x values
      newfile << setw(15) << setprecision(8) << solution[i];
      newfile << setw(15) << setprecision(8) << exact(x[i]);
      newfile << setw(15) << setprecision(8) << rel_error[i] << endl;
    }
    newfile.close();
  }

  //send time values to a file
  string s1("standard_gaussian_time.txt");
  timefile.open(s1);
  timefile << setiosflags(ios::showpoint | ios::uppercase);
  for (int i=1; i<exponent+1; i++) {
    timefile << setw(15) << setprecision(8) << time[i] << endl;
  }
  timefile.close();

  //send mean error values to a file
  string s2("standard_gaussian_error.txt");
  errorfile.open(s2);
  errorfile << setiosflags(ios::showpoint | ios::uppercase);
  for (int i=1; i<exponent+1; i++) {
    errorfile << setw(15) << setprecision(8) << mean_error[i] << endl;
  }
  errorfile.close();
}

void gaussian_special(int exponent) {

  ofstream newfile_1;
  ofstream timefile_1;
  ofstream errorfile_1;

  double *time = new double[exponent];
  double *mean_error = new double[exponent];

  for (int i = 1; i < exponent+1; i++){

    //makes file out name coresspond to size n
    string name1("special_gaussian");
    string size = to_string(i);
    name1.append("_" + size + "_fast.txt");

    //create step size parameters
    int n = (int) pow(10.0,i);
    double h = 1.0/(n);
    double h_square = h*h;

    //solving A * X_hat = B_hat
    //create vectors to be used in forward and back sub as well as x[i] points vector
    //d is middle, b is B_hat, solution is X_hat
    double *x = new double[n+1]; double *d = new double[n+1]; double *b = new double[n+1];
    double *solution = new double[n+1]; double *rel_error = new double[n];

    //fill in points vector x[i] and B_hat vector
    for (int i = 0; i <= n; i++) {
      x[i] = i*h;
      b[i] = h_square*fsource(i*h);
    }

    //define initial conditioins and setup d vector
    solution[0]=solution[n]=0; d[0]=d[n]=0;
    for (int i = 1; i < n; i++) {
      d[i] = (i+1.0)/( (double) i);
    }

    //start clock
    clock_t start, finish;
    start = clock();

    //forward substitution
    for (int i = 1; i < n-1; i++) {
      b[i+1] = b[i+1] + b[i]/d[i];
    }

    //back substitution
    solution[n-1] = b[n-1]/d[n-1];
    for (int i = n-2; i > 0; i--) {
      solution[i] = (b[i]+solution[i+1])/d[i];
    }

    //stop clock
    finish = clock();
    time[i]=((double) (finish-start))/CLOCKS_PER_SEC;

    //relative error
    for (int i = 1; i < n; i++) {
      double xval = x[i];
      rel_error[i] = relative_error(solution[i],exact(xval));
    }

    //put mean error in vector for corresponding n
    mean_error[i] = (rel_error[1]+rel_error[n-1])/2.0;

    //write x values, computed solution, exact solution, and relative error to file
    newfile_1.open(name1);
    newfile_1 << setiosflags(ios::showpoint | ios::uppercase);
    for (int i = 1; i < n; i++) {
      newfile_1 << setw(15) << setprecision(8) << x[i]; //parameterized x values
      newfile_1 << setw(15) << setprecision(8) << solution[i];
      newfile_1 << setw(15) << setprecision(8) << exact(x[i]);
      newfile_1 << setw(15) << setprecision(8) << rel_error[i] << endl;
    }
    newfile_1.close();
  }

  //write time values to file
  string s1("specialized_gaussian_time.txt");
  timefile_1.open(s1);
  timefile_1 << setiosflags(ios::showpoint | ios::uppercase);
  for (int i=1; i<exponent+1; i++) {
    timefile_1 << setw(15) << setprecision(8) << time[i] << endl;
  }
  timefile_1.close();

  //write error values to file
  string s2("specialized_gaussian_error.txt");
  errorfile_1.open(s2);
  errorfile_1 << setiosflags(ios::showpoint | ios::uppercase);
  for (int i=1; i<exponent+1; i++) {
    errorfile_1 << setw(15) << setprecision(8) << mean_error[i] << endl;
  }
  errorfile_1.close();
}

//command line format ./filename n where is 10^n grid points
int main(int argc, char const *argv[]) {
  int exponent_val;
  if( argc != 2 ){
    cout << "ERROR: \nEnter command line as ./my_solver n \nfor a matrix of 10^n gridpoints" << endl;
    exit(1);
  }
  else{
    exponent_val= atoi(argv[1]);
  }
  gaussian_special(exponent_val);
  gaussian(exponent_val);

  return 0;
}
