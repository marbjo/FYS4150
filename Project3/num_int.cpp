#include <iostream>
#include <ctime>
#include <cmath>
#include <tuple>
#include <random>
#include <string>
#include <fstream>
// #include <omp.h>
#define ZERO 1.0E-10
#define EPS 3.0e-14
#define MAXIT 10
using namespace std;

// g++ -std=c++11 name.cpp -o name.exe

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

//get integration point limits
double lambda_limit(double alpha, double tol){
  double lam = .1;
  double val = 1E8;
  while (val > tol){
    val = exp(-2*alpha*lam);
    lam = lam + .01;
  }
  return lam;
}

// from morten
void gauss_legendre(double x1, double x2, double x[], double w[], int n){

  int         m,j,i;
  double      z1,z,xm,xl,pp,p3,p2,p1;
  double      const  pi = 3.14159265359;
  double      *x_low, *x_high, *w_low, *w_high;

  m  = (n + 1)/2;                             // roots are symmetric in the interval
  xm = 0.5 * (x2 + x1);
  xl = 0.5 * (x2 - x1);

  x_low  = x;                                       // pointer initialization
  x_high = x + n - 1;
  w_low  = w;
  w_high = w + n - 1;

  for(i = 1; i <= m; i++) {                             // loops over desired roots
    z = cos(pi * (i - 0.25)/(n + 0.5));

         /*
   ** Starting with the above approximation to the ith root
         ** we enter the mani loop of refinement bt Newtons method.
         */

    do {
       p1 =1.0;
  p2 =0.0;

  	   /*
   ** loop up recurrence relation to get the
         ** Legendre polynomial evaluated at x
         */

  for(j = 1; j <= n; j++) {
    p3 = p2;
    p2 = p1;
    p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3)/j;
  }

   /*
   ** p1 is now the desired Legrendre polynomial. Next compute
         ** ppp its derivative by standard relation involving also p2,
         ** polynomial of one lower order.
         */

  pp = n * (z * p1 - p2)/(z * z - 1.0);
  z1 = z;
  z  = z1 - p1/pp;                   // Newton's method
    } while(fabs(z - z1) > ZERO);

        /*
  ** Scale the root to the desired interval and put in its symmetric
        ** counterpart. Compute the weight and its symmetric counterpart
        */

    *(x_low++)  = xm - xl * z;
    *(x_high--) = xm + xl * z;
    *w_low      = 2.0 * xl/((1.0 - z * z) * pp * pp);
    *(w_high--) = *(w_low++);
  }
}
// from morten
double gammln( double xx){
	double x,y,tmp,ser;
	static double cof[6]={76.18009172947146,-86.50532032941677,
		24.01409824083091,-1.231739572450155,
		0.1208650973866179e-2,-0.5395239384953e-5};
	int j;

	y=x=xx;
	tmp=x+5.5;
	tmp -= (x+0.5)*log(tmp);
	ser=1.000000000190015;
	for (j=0;j<=5;j++) ser += cof[j]/++y;
	return -tmp+log(2.5066282746310005*ser/x);
}
//  Note that you need to call it with a given value of alpha,
// called alf here. This comes from x^{alpha} exp(-x)
// from morten
void gauss_laguerre(double *x, double *w, int n, double alf){
	int i,its,j;
	double ai;
	double p1,p2,p3,pp,z,z1;

	for (i=1;i<=n;i++) {
		if (i == 1) {
			z=(1.0+alf)*(3.0+0.92*alf)/(1.0+2.4*n+1.8*alf);
		} else if (i == 2) {
			z += (15.0+6.25*alf)/(1.0+0.9*alf+2.5*n);
		} else {
			ai=i-2;
			z += ((1.0+2.55*ai)/(1.9*ai)+1.26*ai*alf/
				(1.0+3.5*ai))*(z-x[i-2])/(1.0+0.3*alf);
		}
		for (its=1;its<=MAXIT;its++) {
			p1=1.0;
			p2=0.0;
			for (j=1;j<=n;j++) {
				p3=p2;
				p2=p1;
				p1=((2*j-1+alf-z)*p2-(j-1+alf)*p3)/j;
			}
			pp=(n*p1-(n+alf)*p2)/z;
			z1=z;
			z=z1-p1/pp;
			if (fabs(z-z1) <= EPS) break;
		}
		if (its > MAXIT) cout << "too many iterations in gaulag" << endl;
		x[i]=z;
		w[i] = -exp(gammln(alf+n)-gammln((double)n))/(pp*n*p2);
	}
}

void test_laguerre(){
  /*
  table values taken from https://keisan.casio.com/exec/system/1281279441
  */

  double *x = new double [4];
  double *w = new double [4];
  double table_val_x [] = {0,1.5173871,4.3115831,9.1710298};
  double table_val_w [] = {0,1.0374950,0.9057500,0.0567550};
  double tol = 10E-8;
  double diff_x;
  double diff_w;

  gauss_laguerre(x, w, 3, 2);

  for (int i = 0; i <= 3; i++){
    diff_x = abs(x[i] - table_val_x[i]);
    diff_w = abs(w[i] - table_val_w[i]);
    if (diff_x <= tol){
      // PASS
    }
    if (diff_w <= tol){
      // PASS
    }
    else{
      cout << "Laguerre solver not returning proper weights and integration points" << endl;
      exit(1);
    }
  }
}

void test_legendre(){
  /*
  table values taken from https://keisan.casio.com/exec/system/1280624821
  */

  double *x = new double [4];
  double *w = new double [4];
  double table_val_x [] = {-0.86113631,-0.33998104,0.33998104,0.86113631};
  double table_val_w [] = {0.34785485,0.65214515,0.65214515,0.34785485};
  double tol = 10E-8;
  double diff_x;
  double diff_w;

  gauss_legendre(-1, 1, x, w, 4);

  for (int i = 0; i < 4; i++){
    diff_x = abs(x[i] - table_val_x[i]);
    diff_w = abs(w[i] - table_val_w[i]);
    if (diff_x <= tol){
      // PASS
    }
    if (diff_w <= tol){
      // PASS
    }
    else{
      cout << "Lengendre solver not returning proper weights and integration points" << endl;
      exit(1);
    }
  }
}

// solving methods
double brute_force(int N, double a, double b){
  //N = number of integration points
  //a,b = integration boundaries

  double *x = new double [N]; //integration points
  double *w = new double [N]; //weights for each int point

  gauss_legendre(a, b, x, w, N); //find x and w for our domain with n integration points

  double integral = 0.;
  for (int i = 0; i < N; i++){
    for (int j = 0; j < N; j++){
      for (int k = 0; k < N; k++){
        for (int l = 0; l < N; l++){
          for (int m = 0; m < N; m++){
            for (int n = 0; n < N; n++){
              integral += w[i]*w[j]*w[k]*w[l]*w[m]*w[n] * cartesian_f(x[i], x[j], x[k], x[l], x[m], x[n]);
            }
          }
        }
      }
    }
  }
  return integral;

  delete [] x;
  delete [] w;
}

double improved_gauss(int N){
  // legendre method for theta / phi
  double *theta = new double [N];
  double *wtheta = new double [N];
  double *phi = new double [N];
  double *wphi = new double [N];
  // laguerre method for u because u integral has form u^alf*exp(-u)
  double *u = new double [N+1];
  double *wu = new double [N+1];

  double alf = 2.;
  double const pi = 3.14159265359;
  double integral = 0.;

  gauss_legendre(0, pi, theta, wtheta, N);
  gauss_legendre(0, 2*pi, phi, wphi, N);
  gauss_laguerre(u, wu, N, alf);

  for (int i = 1; i <= N; i++){
    for (int j = 1; j <= N; j++){
      for (int k = 0; k < N; k++){
        for (int l = 0; l < N; l++){
          for (int m = 0; m < N; m++){
            for (int n = 0; n < N; n++){
              integral += wu[i]*wu[j]*wtheta[k]*wtheta[l]*wphi[m]*wphi[n]
                          * spherical_f(u[i], u[j], theta[k], theta[l], phi[m], phi[n]);
            }
          }
        }
      }
    }
  }
  return integral;

  delete [] u;
  delete [] wu;
  delete [] wphi;
  delete [] phi;
  delete [] wtheta;
  delete [] theta;

}

tuple<double, double> brute_monte(int N, double a, double b){
  /*
  monte carlo integration just using uniform distribution
  */
  mt19937 generate(2019);
  uniform_real_distribution<double> my_dist(a,b);

  double x1;
  double y1;
  double z1;
  double x2;
  double y2;
  double z2;
  double MCint;
  double MCintsqr;
  double fx;
  double scale = pow(b-a,6); //accounts for rescaling uniform dist for each integral over (b-a)

  MCint = MCintsqr = 0;

  for (int i = 1; i <= N; i++){
    x1 = my_dist(generate);
    y1 = my_dist(generate);
    z1 = my_dist(generate);
    x2 = my_dist(generate);
    y2 = my_dist(generate);
    z2 = my_dist(generate);
    fx = cartesian_f(x1,y1,z1,x2,y2,z2);
    MCint += fx;
    MCintsqr += fx*fx;
  }
  MCint = MCint*scale / ((double) N);
  MCintsqr = MCintsqr*scale*scale / ((double) N);
  double var = (MCintsqr - MCint*MCint) / ((double) N);
  tuple<double,double> result = make_tuple(MCint,var);
  return result;
}

tuple<double,double> improved_monte(int N){
  /*
  use exp_dist of form e^-(a*x) for exponential piece and
  uniform dist for angular pieces
  */

  mt19937 generate(2019);
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

  for (int i = 1; i <= N; i++){
    u1 = u_dist(generate);
    u2 = u_dist(generate);
    theta1 = theta_dist(generate);
    theta2 = theta_dist(generate);
    phi1 = phi_dist(generate);
    phi2 = phi_dist(generate);
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

void timing_function(int N, bool quad_solve, int M, bool monte_solve){
  /*
  Timing function will do gaussian quadrature for multiples of five up to and including N.
  It will do montecarlo simulation for 10^i points in the range i = 1... i=M.
  It then prints out values from solving:

  quadrature: intergration points - legendre results -%error - legendre time - laguerre results -%error - laguerre lag_time

  montecalro: points - results - %error - variance - time - improved results - %error - improved variance - improved lag_time
              (improved means we used importance sampling)
  results are printed to two output files
  */
  double analytic = (5*M_PI*M_PI)/(16*16);

  ofstream montefile;

  if (quad_solve == true){
    ofstream quadfile;
    double *leg_time = new double [N/5];
    double *leg_results = new double [N/5];
    double *lag_time = new double [N/5];
    double *lag_results = new double [N/5];
    double *points = new double [N/5];

    // brute force methods
    double a = lambda_limit(2,10E-5);

    for (int i = 5; i <= N; i+=5){
      clock_t start, finish, start1, finish1;
      start = clock();
      leg_results[i/5-1] = brute_force(i,-a,a);
      finish = clock();
      leg_time[i/5-1] = ((double) (finish-start))/CLOCKS_PER_SEC;

      start1 = clock();
      lag_results[i/5-1] = improved_gauss(i);
      finish1 = clock();
      lag_time[i/5-1] = ((double) (finish1-start1))/CLOCKS_PER_SEC;

      points[i/5-1] = i;
    }

    string s("brute_force_results.txt");
    quadfile.open(s);
    quadfile << setiosflags(ios::showpoint | ios::uppercase);
    for (int i = 0; i < N/5; i++){
      quadfile << setw(15) << setprecision(8) << points[i];
      quadfile << setw(15) << setprecision(8) << leg_results[i];
      quadfile << setw(15) << setprecision(8) << abs((leg_results[i]-analytic)/analytic);
      quadfile << setw(15) << setprecision(8) << leg_time[i];
      quadfile << setw(15) << setprecision(8) << lag_results[i];
      quadfile << setw(15) << setprecision(8) << abs((lag_results[i]-analytic)/analytic);
      quadfile << setw(15) << setprecision(8) << lag_time[i] << endl;
    }
    quadfile.close();

    delete [] points;
    delete [] leg_results;
    delete [] leg_time;
    delete [] lag_results;
    delete [] lag_time;
  }

  if (monte_solve == true){
    ofstream montefile;

    double *bad_monte_time = new double [M];
    double *bad_monte_results = new double [M];
    double *bad_monte_var = new double [M];
    double *better_monte_time = new double [M];
    double *better_monte_results = new double [M];
    double *better_monte_var = new double [M];
    double *points = new double [M];

    double lamb = lambda_limit(2,10E-5);

    for (int i = 1; i <= M; i++){

      double pts = pow(10,i);
      points[i-1]=pts;

      clock_t start, finish, start1, finish1;
      start = clock();
      tuple<double,double> a = brute_monte(pts,-lamb,lamb);
      finish = clock();
      bad_monte_results[i-1] = get<0>(a);
      bad_monte_var[i-1] = get<1>(a);
      bad_monte_time[i-1] = ((double) (finish-start))/CLOCKS_PER_SEC;

      start1 = clock();
      tuple<double,double> b = improved_monte(pts);
      finish1 = clock();
      better_monte_results[i-1] = get<0>(b);
      better_monte_var[i-1] = get<1>(b);
      better_monte_time[i-1] = ((double) (finish1-start1))/CLOCKS_PER_SEC;
    }

    string s("montecarlo_results.txt");
    montefile.open(s);
    montefile << setiosflags(ios::showpoint | ios::uppercase);
    for (int i = 0; i < M; i++){
      montefile << setw(15) << setprecision(8) << points[i];
      montefile << setw(15) << setprecision(8) << bad_monte_results[i];
      montefile << setw(15) << setprecision(8) << abs((bad_monte_results[i]-analytic)/analytic);
      montefile << setw(15) << setprecision(8) << bad_monte_var[i];
      montefile << setw(15) << setprecision(8) << bad_monte_time[i];
      montefile << setw(15) << setprecision(8) << better_monte_results[i];
      montefile << setw(15) << setprecision(8) << abs((better_monte_results[i]-analytic)/analytic);
      montefile << setw(15) << setprecision(8) << better_monte_var[i];
      montefile << setw(15) << setprecision(8) << better_monte_time[i]<< endl;
    }
    montefile.close();

    delete [] points;
    delete [] bad_monte_results;
    delete [] bad_monte_var;
    delete [] bad_monte_time;
    delete [] better_monte_results;
    delete [] better_monte_var;
    delete [] better_monte_time;
  }


}

int main(int argc, char const *argv[]) {

  bool brute_q = false;
  bool better_q = false;
  bool brute_m = false;
  bool better_m = false;
  double analytic = (5*M_PI*M_PI)/(16*16);

  test_laguerre();
  test_legendre();

  if (brute_q == true){
    double a = lambda_limit(2,10E-5);
    cout << "int limits= " << a << endl;
    double b = brute_force(35, -a ,a);
    cout << "this is stuff= " << b << endl;
  }
  if (better_q == true){
    double a = improved_gauss(25);
    cout <<"improved cause value= " << a << endl;
    cout <<"true value= " << analytic << endl;
  }
  if (brute_m == true){
    double b = lambda_limit(2,10E-5);
    cout << "int limits= " << b << endl;
    tuple<double,double> a = brute_monte(1E7,-b,b);
    cout << "MC integration= " << get<0>(a) <<endl;
    cout << "variance= " << get<1>(a) << endl;
    cout <<"true value= " << analytic << endl;
  }
  if (better_m == true){
    tuple<double,double> a = improved_monte(1E7);
    cout << "MC integration= " << get<0>(a) <<endl;
    cout << "variance= " << get<1>(a) << endl;
    cout <<"true value= " << analytic << endl;
  }
  timing_function(35, true, 7, true);


  return 0;
}
