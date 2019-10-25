//   This is a simple program which tests the trapezoidal rule, Simpsons' rule,
//   and Gaussian quadrature using Legendre and Laguerre polynomials
//   It integrates the simple function x* exp(-x) for the interval
//   x \in [0,infty). The exact result is 1. For Legendre based quadrature a
//   tangent mapping is also used.

#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <ctime>
#include <armadillo>
#include <random>

using namespace std;

inline int periodic_index(int i, int N){
    //Returns index=0 when reaching ends
    //Otherwise returns same index sent in
    int index = (i+N) % N;
    return index;
}
inline double delta_E_func(arma::mat A, int i, int j, int L){
    //Function for calculating energy difference between new and old state
    double value = 2*A(i,j) * ( A(periodic_index(i-1,L),j) + A(periodic_index(i+1,L),j) + A(i,periodic_index(j-1,L)) + A(i,periodic_index(j+1,L)) );
    return value;
}

int main(int argc, char const *argv[]){
    if (argc < 2) {
        cout << "Error: missing command line argument. Must provide 1 if you want printed output, or anything else if not." << endl;
        return 1;
    }
    int print = atoi(argv[1]);
    //All natural units
    int L = 2; //Dimensionality
    int J = 1; //Interaction strength
    double k_b = 1.0; //1.38*pow(10,-23); //Boltzmann constant
    double T = 1.0; //Temperature
    double beta = 1/(k_b*T);
    arma::mat A(L,L, arma::fill::ones); //Initial state doesn't mattter, so just setting all ones

    //Setting
    int T_start = 1;
    int T_end = 3;
    int N_temp = 10;
    arma::vec T_vec = arma::linspace(T_start,T_end,N_temp);
    int MC_max = 1E5; //1E5;

    //Creating matrices for plotting against temperature
    arma::vec E_tot(N_temp,arma::fill::zeros); //E(T)
    arma::vec M_tot(N_temp,arma::fill::zeros); //M(T)
    arma::vec M_abs_tot(N_temp,arma::fill::zeros); //|M(T)|
    arma::vec E_tot_2(N_temp,arma::fill::zeros); // [E(T)]^2
    arma::vec M_tot_2(N_temp,arma::fill::zeros); // [M(T)]^2
    arma::vec C_v_tot(N_temp,arma::fill::zeros); // C_v(T)
    arma::vec chi_tot(N_temp,arma::fill::zeros); // chi(T)

    //Creating random distributions
    random_device rd; //Seed
    mt19937 generate(rd()); //Mersenne-Twister generator
    uniform_real_distribution<double> my_dist1(0,1);
    uniform_real_distribution<double> my_dist2(0,L);

    arma::vec acceptance_tol(5); //Precalced array of ratios for accepting a MC step.
    for(int i=0; i<5; i++){
        //Filling precalc-array for ratios
        acceptance_tol(i) = exp(-beta*(-8 + 4*i));
    }

    for(int i0=0; i0<N_temp; i0++){
        //Looping over temperatures
        double T = T_vec(i0);
        //Initizaling variables for chosen microstate, and arrays for E,M,E^2 and M^2
        double E_init = 0;
        double M_init = 0;
        arma::vec E_arr(MC_max);
        arma::vec M_arr(MC_max);
        arma::vec E_arr_2(MC_max);
        arma::vec M_arr_2(MC_max);

        //Calculating energy and magnetic momentum for intial state
        for(int i=0; i<L; i++){
            for(int j=0; j<L; j++){
                E_init += -J *( A(i,j)*A(periodic_index(i+1,L),j) + A(i,j)*A(i,periodic_index(j+1,L)) );
                M_init += A(i,j);
            }
        }

        for(int mc=0; mc < MC_max; mc++){
            //Monte Carlo loop
            for(int u=0; u<L*L; u++){
                //Loop for flipping L^2 times

                //Picking random numbers for r (acceptance rule), and two random indices for flipping a grid point
                double r = my_dist1(generate);
                int ran_index1 = int(my_dist2(generate));
                int ran_index2 = int(my_dist2(generate));

                double delta_E_val = delta_E_func(A, ran_index1, ran_index2, L);
                int corr_index = round((delta_E_val+8)/4); //Finding corresponding index for delta_E, to use precalced values

                if(r <= acceptance_tol(corr_index) ){
                    //Acceptance rule r <= e^(-beta*delta_E)
                    A(ran_index1, ran_index2) = -1*A(ran_index1, ran_index2);
                    E_init += delta_E_val;
                    M_init += 2*A(ran_index1,ran_index2);
                }//End acceptance rule

            }//End flipping loop

            //Saving values for E,M,E^2 and M^2 for one Monte Carlo iteration
            E_arr(mc) = E_init;
            M_arr(mc) = M_init;
            E_arr_2(mc) = E_arr(mc)*E_arr(mc);
            M_arr_2(mc) = M_arr(mc)*M_arr(mc);

        }//End Monte Carlo loop

        //Computing expectation values of E,M,E^2,M^2 and the values C_v and chi
        double expec_E = arma::sum(E_arr) / MC_max;  // <E>
        double expec_M = arma::sum(M_arr) / MC_max; // <M>
        double expec_E_2 = arma::sum(E_arr_2) / MC_max; // <E^2>
        double expec_M_2 = arma::sum(M_arr_2) / MC_max; // <M^2>
        double C_v = (expec_E_2 - expec_E*expec_E) / (k_b*T*T); //Heat capacity
        double chi = (expec_M_2 - expec_M*expec_M) / (k_b*T); //Susceptibility

        cout << "T = " << T << endl;

        if (print==1){
            //Send in 1 as command line argument if you want this output
            cout << "<E> : " << expec_E << endl;
            cout << "<M> : " << expec_M << endl;
            cout << "<|M|> : " << sqrt(expec_M_2) << endl;
            cout << "<E^2> : " << expec_E_2 << endl;
            cout << "<M^2> : " << expec_M_2 << endl;
            cout << "C_v : " << C_v << endl;
            cout << "chi : " << chi << endl;

            if(L==2 && (T-1) < 1E-10){
                //Analytically derived values for 2x2 case, only for benchmarking.
                //Only calculates if L=2 and for T=1.

                double arg = 8*beta*J;
                double analytic_E = -8*J*sinh(arg)/(cosh(arg)+3);
                double analytic_M = (2*J*exp(arg) + 4 )/ (cosh(arg) + 3);
                double analytic_Cv = (64*J*J*(3*cosh(arg)+1) / ((cosh(arg) + 3)*(cosh(arg) + 3) )) * 1/(k_b*T*T);
                double analytic_chi = (8*(exp(arg)+1)/(cosh(arg) + 3) )/ (k_b*T);
                cout << "Analytic <E> : " << analytic_E << endl;
                cout << "Analytic <|M|> : " << analytic_M << endl;
                cout << "Analytic C_v : " << analytic_Cv << endl;
                cout << "Analytic chi : " << analytic_chi << endl;
            }
        }

        //Saving values for this value of T for later
        E_tot(i0) = expec_E;
        M_tot(i0) = expec_M;
        M_abs_tot(i0) = sqrt(expec_M_2);
        E_tot_2(i0) = expec_E_2;
        M_tot_2(i0) = expec_M_2;
        C_v_tot(i0) = C_v;
        chi_tot(i0) = chi;

    }
    return 0; //End of main program
}
