#ifndef MC_H_
#define MC_H_

#include<time.h>
#include<stdlib.h>
#include <vector>
#include"scalapack.h"
#include"environment.h"
#include<mpi.h>

using namespace std;
using namespace dgdft::scalapack;

const int I_ZERO = 0;
const int I_ONE = 1;

extern int L;
extern int K;
extern int pad;
extern int channel;
extern double J2;

inline int IRound(double a){ 
  int b = 0;
  if(a>0) b = (a-int(a)<0.5)?int(a):(int(a)+1);
  else b = (int(a)-a<0.5)?int(a):(int(a)-1);
  return b; 
}

extern "C" {

    void dpotrf_(char* UPLO, int* N, double* A, int *lda, int* INFO);

    void dtrsv_(char* UPLO, char* TRANSA, char* DIAG, int* N, double* A, int* lda, double* B, int* incb);

    void dgemm_(char* TRANSA, char* TRANSB, int* M, int* N, int* K, double* ALPHA, double* A, int* lda,
                    double* B, int* ldb, double* BETA, double* C, int* ldc);

    void dgemv_(char* TRANS, int* M, int* N, double* ALPHA, double* A, int* lda, double* X, int* incx,
                    double* BETA, double* Y, int* incy);

    void daxpy_(int* N, double* ALPHA, double* X, int* incx, double* Y, int* incy);
    
}

struct proposeEntry{
    int source;
    int target;
};

struct proposePrime{
    int* spin_lattice;
    double energy;
    double ws;
    int index;
    int fly;
    std::vector<proposeEntry> propose_J1;
    std::vector<proposeEntry> propose_J2;
};

void batch_sprime(int* input_ptr, int batch_size, std::vector<proposePrime>& batchPrime);

void PBC(int* input_ptr, int* output_ptr, int batch_size);

void sign_rule(int* input_ptr, int batch_size, int* sign_result);

int select_samples(double* Es_list, double* Os_list, int batch_size, int params_size, std::vector<proposePrime>& batchPrime, int* init_spin_lattice, int rank);

void calculate_parameter(double* Es, double* Os, int batch_size, int params_size, double* Es_avg, double* Os_avg, double* OsEs_avg, double* OO_avg);

void compute_grad(double* Os_avg, double* Es_avg, double* OsEs_avg, double* dt, int params_size, double* grad);

void covariance_matrix(double* OO_avg, double* Os_avg, double shift, int params_size);

void compute_delta_scalapack(double* A, double* B, int N, double* C, int numProcess);

void compute_delta(double* A, double* B, int N, double* C);

#endif