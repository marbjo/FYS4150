//Program which does stuff
//WITH PARALLELIZATION

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

#include <mpi.h>

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

int main(int argc, char *argv[]){
    if (argc < 2) {
        cout << "Error: missing command line argument. Must provide 1 if you want printed output, or anything else if not." << endl;
        return 1;
    }

    MPI_Init (&argc, &argv);
    int numprocs;
    MPI_Comm_size (MPI_COMM_WORLD, &numprocs); //Letting everybody know the size of communicator
    int my_rank;
    MPI_Comm_rank (MPI_COMM_WORLD, &my_rank); //Assigning ranks

    int print = atoi(argv[1]);
    //All natural units
    int L = 100; //Dimensionality
    double J = 1.0; //Interaction strength
    double k_b = 1.0; //1.38*pow(10,-23); //Boltzmann constant

    //arma::mat A(L,L, arma::fill::ones); //Initial state doesn't mattter, so just setting all ones

    //Defining random initial state (1 or -1)
    arma::mat A(L,L);
    for(int i=0; i<L; i++){
        for(int k=0; k<L; k++){
            int number = rand()%2;

            if(number==0){
                number = -1;
            }
            A(i,k) = number;
        }
    }

    //Setting temperatures to loop over.
    int N_temp = 32; //Should be dividable by number of processes, or else fuckery might occur.
    int N_temp_local = (N_temp) / numprocs; //Number of points for each core.

    double T_start = 2.00;
    double T_end = 2.30;

    double delta_T = (T_end-T_start)/(N_temp);

    double T_start_local = T_start + my_rank*delta_T*N_temp_local;
    double T_end_local = T_start + (my_rank+1)*delta_T*N_temp_local - delta_T;

    arma::vec T_vec = arma::linspace(T_start_local,T_end_local,N_temp_local);

    int MC_max = 1E4;

    //Creating matrices for plotting against temperature
    arma::vec E_tot(N_temp_local,arma::fill::zeros); //E(T)
    arma::vec M_tot(N_temp_local,arma::fill::zeros); //M(T)
    arma::vec M_abs_tot(N_temp_local,arma::fill::zeros); //|M(T)|
    arma::vec E_tot_2(N_temp_local,arma::fill::zeros); // [E(T)]^2
    arma::vec M_tot_2(N_temp_local,arma::fill::zeros); // [M(T)]^2
    arma::vec C_v_tot(N_temp_local,arma::fill::zeros); // C_v(T)
    arma::vec chi_tot(N_temp_local,arma::fill::zeros); // chi(T)

    //Creating random distributions
    random_device rd; //Seed
    mt19937 generate(rd()); //Mersenne-Twister generator
    uniform_real_distribution<double> my_dist1(0,1);
    uniform_real_distribution<double> my_dist2(0,L);

    for(int i0=0; i0<N_temp_local; i0++){
        //Looping over temperatures

        double T = T_vec(i0);
        double beta = 1/(k_b*T);

        arma::vec acceptance_tol(5); //Precalced array of ratios for accepting a MC step.
        for(int i=0; i<5; i++){
            //Filling precalc-array for ratios
            acceptance_tol(i) = exp(-beta*(-8 + 4*i));
        }


        //Initizaling variables for chosen microstate, and arrays for E,M,E^2 and M^2
        double E_init = 0;
        double M_init = 0;
        arma::vec E_arr(MC_max);
        arma::vec M_arr(MC_max);
        arma::vec E_arr_2(MC_max);
        arma::vec M_arr_2(MC_max);

        //Vectors for plotting |E| and |M| as a function of MC steps
        arma::vec E_of_MC(MC_max);
        arma::vec M_of_MC(MC_max);

        //Calculating energy and magnetic momentum for intial state
        for(int i=0; i<L; i++){
            for(int j=0; j<L; j++){
                E_init += -J *( A(i,j)*A(periodic_index(i+1,L),j) + A(i,j)*A(i,periodic_index(j+1,L)) );
                M_init += A(i,j);
            }
        }

        //Intermediate variables to plot |E| and |M| as a function of MC steps
        double E_of_MC_counter = 0;
        double M_of_MC_counter = 0;

        arma::vec accept_arr(MC_max);

        for(int mc=0; mc < MC_max; mc++){
            double accept_counter = 0;
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

                    accept_counter += 1;
                }//End acceptance rule
            }//End flipping loop

            //Saving values for E,M,E^2 and M^2 for one Monte Carlo iteration
            E_arr(mc) = E_init;
            M_arr(mc) = M_init;
            E_arr_2(mc) = E_arr(mc)*E_arr(mc);
            M_arr_2(mc) = M_arr(mc)*M_arr(mc);

            //Calculating |E| and |M| as a function of MC steps
            E_of_MC_counter += E_arr[mc];
            M_of_MC_counter += E_arr[mc];

            //Saving to array for plotting
            E_of_MC[mc] = E_of_MC_counter / (mc+1);
            M_of_MC[mc] = fabs(M_of_MC_counter / (mc+1));

            //Acceptance as a function of MC steps
            accept_arr(mc) = accept_counter;

        }//End Monte Carlo loop

        //Computing expectation values of E,M,E^2,M^2 and the values C_v and chi
        double expec_E = arma::sum(E_arr) / MC_max;  // <E>
        double expec_M = arma::sum(M_arr) / MC_max; // <M>
        double M_abs = sqrt(expec_M*expec_M);
        double expec_E_2 = arma::sum(E_arr_2) / MC_max; // <E^2>
        double expec_M_2 = arma::sum(M_arr_2) / MC_max; // <M^2>
        double C_v = (expec_E_2 - expec_E*expec_E) / (k_b*T*T); //Heat capacity
        double chi = (expec_M_2 - M_abs*M_abs) / (k_b*T); //Susceptibility

        //Saving values for this value of T for later
        E_tot(i0) = expec_E;
        M_tot(i0) = expec_M;
        M_abs_tot(i0) = M_abs;
        E_tot_2(i0) = expec_E_2;
        M_tot_2(i0) = expec_M_2;
        C_v_tot(i0) = C_v;
        chi_tot(i0) = chi;

        //Writing |E| and |M| to file for plotting as a function of MC steps.
        // ofstream montefile;
        // char s[80];
        // sprintf(s,"%.3f_MC_results.txt", T);
        // montefile.open(s);
        // montefile << setiosflags(ios::showpoint | ios::uppercase);
        //
        // montefile << setw(15) << setprecision(1) << "Iteration";
        // montefile << setw(15) << setprecision(8) << "<E>";
        // montefile << setw(15) << setprecision(8) << "E(MC) (normalized)";
        // montefile << setw(15) << setprecision(8) << "<|M|>";
        // montefile << setw(15) << setprecision(8) << "Accepted flips";
        // montefile << endl;
        //
        // for (int i = 0; i < MC_max; i++){
        //   montefile << setw(15) << setprecision(1) << i;
        //   montefile << setw(15) << setprecision(8) << E_arr[i];
        //   montefile << setw(15) << setprecision(8) << E_of_MC[i];
        //   montefile << setw(15) << setprecision(8) << M_of_MC[i];
        //   montefile << setw(15) << setprecision(8) << accept_arr[i];
        //   montefile << endl;
        // }
        // montefile.close();

        //This code block is only for printing output
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
        }//End if-print

    }//End temperature loop

    //Gathering all data in root process (0)
    arma::vec T_gather(N_temp, arma::fill::zeros);
    MPI_Gather(T_vec.memptr(), N_temp_local, MPI_DOUBLE, T_gather.memptr(), N_temp_local, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    arma::vec E_gather(N_temp, arma::fill::zeros);
    MPI_Gather(E_tot.memptr(), N_temp_local, MPI_DOUBLE, E_gather.memptr(), N_temp_local, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    arma::vec M_gather(N_temp, arma::fill::zeros);
    MPI_Gather(M_abs_tot.memptr(), N_temp_local, MPI_DOUBLE, M_gather.memptr(), N_temp_local, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    arma::vec Cv_gather(N_temp, arma::fill::zeros);
    MPI_Gather(C_v_tot.memptr(), N_temp_local, MPI_DOUBLE, Cv_gather.memptr(), N_temp_local, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    arma::vec chi_gather(N_temp, arma::fill::zeros);
    MPI_Gather(chi_tot.memptr(), N_temp_local, MPI_DOUBLE, chi_gather.memptr(), N_temp_local, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //Writing <E(T)> , <M(T)>, <chi(T)> and <C_v(T)> to file. File is named with appropriate spin number
    if(my_rank==0){ //Only root process writes to file
        ofstream expecvalues;
        char name[80];
        sprintf(name,"L%d_expec_values.txt", L);

        expecvalues.open(name);
        expecvalues << setiosflags(ios::showpoint | ios::uppercase);

        expecvalues << setw(15) << setprecision(8) << "Temperature";
        expecvalues << setw(15) << setprecision(8) << "<E(T)>";
        expecvalues << setw(15) << setprecision(8) << "<M(T)>";
        expecvalues << setw(15) << setprecision(8) << "<C_v(T)>";
        expecvalues << setw(15) << setprecision(8) << "<Chi(T)>";
        expecvalues << endl;

        for(int i=0; i<N_temp; i++){
            expecvalues << setw(15) << setprecision(8) << T_gather[i];
            expecvalues << setw(15) << setprecision(8) << E_gather[i];
            expecvalues << setw(15) << setprecision(8) << M_gather[i];
            expecvalues << setw(15) << setprecision(8) << Cv_gather[i];
            expecvalues << setw(15) << setprecision(8) << chi_gather[i];
            expecvalues << endl;
        }
        expecvalues.close();
    }
    MPI_Finalize();

    //THINGS TO DO: LOOK AT LOGIC BEHIND N_local stuff, not correct division between cores.
    //ALSO DO THE SAME STUFF FOR OTHER VALUES AS YOU DID FOR Temperature
    return 0; //End of main program
}
