#include <iostream>
#include <ctime>
#include <cmath>
#include <tuple>
#include <random>
#include <string>
#include <fstream>
#include <omp.h>
#define ZERO 1.0E-10

using namespace std;

//wave function in cartesian coordinates
double cartesian_f(double x1, double y1, double z1, double x2, double y2, double z2){
  double alpha = 2.;

  double exp1 = -2*alpha*sqrt(x1*x1 + y1*y1 + z1*z1);
  double exp2 = -2*alpha*sqrt(x2*x2 + y2*y2 + z2*z2);
  double deno = sqrt( (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2) );
  if(deno < ZERO){
      return 0;
  }
  return exp(exp1+exp2) / deno ;
}

//wave function remainder spherical coordinates
double spherical_f(double u1, double u2, double theta1, double theta2, double phi1, double phi2){
  double alpha = 2.;
  double cos_beta = cos(theta1)*cos(theta2) + sin(theta1)*sin(theta2)*cos(phi1 - phi2);
  double deno = u1*u1 + u2*u2 - 2*u1*u2*cos_beta;
  if (deno < ZERO){
    return 0;
  }
  return (1/sqrt(deno))*(1/pow(2*alpha,5))*sin(theta1)*sin(theta2);
}

tuple<double,double> p_improved_monte(int N, int threads){
  /*
  use exp_dist of form e^-(a*x) for exponential piece and
  uniform dist for angular pieces
  */

  mt19937 generator;
  exponential_distribution<double> u_dist(1);
  uniform_real_distribution<double> theta_dist(0,M_PI);
  uniform_real_distribution<double> phi_dist(0,2*M_PI);

  double u1;
  double u2;
  double theta1;
  double theta2;
  double phi1;
  double phi2;
  double MCint;
  double MCintsqr;
  double fx;
  double scale = (4*M_PI*M_PI*M_PI*M_PI); // same theory as previously. rescaling uniform  dist for angles
                                          // with (b-a) contirubtion from  each angle integral
  MCint = MCintsqr = 0;

  #pragma omp parallel reduction(+:MCint, MCintsqr) num_threads(threads) private(u1, u2, theta1, theta2, phi1, phi2, fx, generator);
  generator.seed(omp_get_thread_num());
  #pragma omp parallel for
  for (int i = 1; i <= N; i++){
    u1 = u_dist(generator);
    u2 = u_dist(generator);
    theta1 = theta_dist(generator);
    theta2 = theta_dist(generator);
    phi1 = phi_dist(generator);
    phi2 = phi_dist(generator);
    fx = u1*u1*u2*u2*spherical_f(u1,u2,theta1,theta2,phi1,phi2);
    MCint += fx;
    MCintsqr += fx*fx;
  }
  MCint = MCint*scale / ((double) N);
  MCintsqr = MCintsqr*scale*scale / ((double) N);
  double var = (MCintsqr - MCint*MCint) / ((double) N);
  tuple<double,double> result = make_tuple(MCint,var);
  return result;
}

int main(int argc, char const *argv[]) {
  /*
  Get the number of threads for parallization as a command line arguement,
  then run program.

  use -1 for max threads, specify positive numbers otherwise

  will iterate over 10^M points with N specified in the command line

  c++ <-O, -O2, -O3> -std=c++11 -o <name.exe> <name.cpp> -Xpreprocessor -fopenmp -lomp
  ./<name.exe> <num_threads> <M>

  this function is not elegant and when you change compiler flags, you must also
  adjust string name manually for the output function

  */

  int threads = atoi(argv[1]);
  int M = atoi(argv[2]);
  double analytic = (5*M_PI*M_PI)/(16*16);

  if (threads > omp_get_max_threads()){
    threads = omp_get_max_threads();
    cout << "Specified threads exceeds max available, we will use max availbe, " << threads <<endl;
  }
  else if (threads == -1){
    threads = omp_get_max_threads();
  }

  // SPHERICAL MONTE CARLO
  ofstream monte2;
  string s1("thread_monte_improved_O3.txt");
  string thd1 = to_string(threads);
  thd1.append("_" + s1);
  monte2.open(thd1);

  for (int i = 1; i <= M; i++){
    double pts = pow(10,i);

    double start = omp_get_wtime();
    tuple <double, double> a = p_improved_monte(pts, threads);
    double finish = omp_get_wtime();
    double time = (finish-start);

    monte2 << setw(15) << setprecision(8) << pts;
    monte2 << setw(15) << setprecision(8) << get<0>(a);
    monte2 << setw(15) << setprecision(8) << abs((get<0>(a)-analytic)/analytic);
    monte2 << setw(15) << setprecision(8) << get<1>(a);
    monte2 << setw(15) << setprecision(8) << time << endl;
  }
  monte2.close();



  return 0;
}
