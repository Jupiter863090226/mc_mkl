#include <fstream>
#include <sstream>
#include <utility>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

#include<time.h>
#include<stdlib.h>

#include"scalapack.h"
#include"environment.h"

#include<mpi.h>

using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;
using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

using namespace dgdft::scalapack;

inline int IRound(double a){ 
  int b = 0;
  if(a>0) b = (a-int(a)<0.5)?int(a):(int(a)+1);
  else b = (int(a)-a<0.5)?int(a):(int(a)-1);
  return b; 
}

const int I_ZERO = 0;
const int I_ONE = 1;

extern "C" {

    void dpotrf_(char* UPLO, int* N, double* A, int *lda, int* INFO);

    void dtrsv_(char* UPLO, char* TRANSA, char* DIAG, int* N, double* A, int* lda, double* B, int* incb);

    void dgemm_(char* TRANSA, char* TRANSB, int* M, int* N, int* K, double* ALPHA, double* A, int* lda,
                    double* B, int* ldb, double* BETA, double* C, int* ldc);

    void dgemv_(char* TRANS, int* M, int* N, double* ALPHA, double* A, int* lda, double* X, int* incx,
                    double* BETA, double* Y, int* incy);

    void daxpy_(int* N, double* ALPHA, double* X, int* incx, double* Y, int* incy);
    
}

int L = 10;
int K = 5;
int pad = (K-1)/2;
int channel = 1;
double J2 = 0.5;

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

void batch_sprime(int* input_ptr, int batch_size, std::vector<proposePrime>& batchPrime) {
        
    batchPrime.clear();
    int* single_ptr = input_ptr;

    for(int i=0; i<batch_size; i++) {
        proposePrime singlePrime;
        singlePrime.spin_lattice = single_ptr;

        double energy = 0;
        int i_target, j_target;

        for(int i=0; i<L; i++){
            for(int j=0; j<L; j++){
                i_target = (i+1)%L;
                j_target = j;
                if (single_ptr[i*L+j]*single_ptr[i_target*L+j_target] == -1) {
                    proposeEntry pe;
                    pe.source = i*L+j;
                    pe.target = i_target*L+j_target;
                    singlePrime.propose_J1.push_back(pe);
                    energy -= 1;
                } else {
                    energy += 1;
                }

                i_target = i;
                j_target = (j+1)%L;
                if (single_ptr[i*L+j]*single_ptr[i_target*L+j_target] == -1) {
                    proposeEntry pe;
                    pe.source = i*L+j;
                    pe.target = i_target*L+j_target;
                    singlePrime.propose_J1.push_back(pe);
                    energy -= 1;
                } else {
                    energy += 1;
                }

                i_target = (i+1)%L;
                j_target = (j+1)%L;
                if (single_ptr[i*L+j]*single_ptr[i_target*L+j_target] == -1) {
                    proposeEntry pe;
                    pe.source = i*L+j;
                    pe.target = i_target*L+j_target;
                    singlePrime.propose_J2.push_back(pe);
                    energy -= J2;
                } else {
                    energy += J2;
                }

                i_target = (i+1)%L;
                j_target = (j-1+L)%L;
                if (single_ptr[i*L+j]*single_ptr[i_target*L+j_target] == -1) {
                    proposeEntry pe;
                    pe.source = i*L+j;
                    pe.target = i_target*L+j_target;
                    singlePrime.propose_J2.push_back(pe);
                    energy -= J2;
                } else {
                    energy += J2;
                }
            }
        }
        singlePrime.energy = energy;
        singlePrime.ws = 0;
        singlePrime.fly = 0;
        batchPrime.push_back(singlePrime);
        single_ptr += L*L;
    }
}

void PBC(int* input_ptr, int* output_ptr, int batch_size) {
    int* start_input = input_ptr;
    int* start_output = output_ptr;
    int j_target;
    int k_target;
    for(int i=0; i<batch_size; i++) {
        for(int j=0; j<L+2*pad; j++) {
            for(int k=0; k<L+2*pad;k++) {
                j_target = (j-pad+L)%L;
                k_target = (k-pad+L)%L;
                start_output[j*(L+2*pad)+k] = start_input[j_target*L+k_target];
            }
        }
        start_input += L*L;
        start_output += (L+2*pad)*(L+2*pad);
    }
}

void sign_rule(int* input_ptr, int batch_size, int* sign_result) {
    int* start = input_ptr;
    int count_positive;
    for(int i=0; i<batch_size; i++) {
        count_positive = 0;
        for(int j=0; j<L; j++)
            for(int k=0; k<L; k++)
                if((j%2==0 && k%2==0) || (j%2==1 && k%2==1)){
                    if (start[j*L+k]==1)
                        count_positive += 1;
            }
                
        if(count_positive%2 == 0) {
            sign_result[i] = 1;
        } else {
            sign_result[i] = -1;
        }
        start += L*L;
    }
}

int select_samples(double* Es_list, double* Os_list, int batch_size, int params_size, std::vector<proposePrime>& batchPrime, int* init_spin_lattice, int rank){
    int accept_samples_size = 0;
    for(int i=0; i<batch_size; i++){
        if(Es_list[i]/400<-0.3 && Es_list[i]/400>-2){
            Es_list[accept_samples_size] = Es_list[i];
            for(int j=0; j<params_size; j++)
                Os_list[accept_samples_size*params_size+j] = Os_list[i*params_size+j];
            accept_samples_size += 1;
            batchPrime[i].fly = 0;
        } else {
            batchPrime[i].fly += 1;

            FILE *fp;
            if((fp=fopen("flyaway_record.txt", "a"))==NULL) {
                printf("Cannot open file.\n");
                exit(1);
            }
            fprintf(fp, "Rank: %d, chain_id: %d, fly_count: %d, ws: %f, Es: %f\n", rank, i, batchPrime[i].fly, batchPrime[i].ws, Es_list[i]/400);
            fclose(fp);

            if(batchPrime[i].fly > 3){
                for(int j=0; j<L; j++)
                    for(int k=0; k<L; k++)
                        batchPrime[i].spin_lattice[j*L+k] = init_spin_lattice[j*L+k];
                batchPrime[i].fly = 0;
            }
        }
    }
    return accept_samples_size;
}

void calculate_parameter(double* Es, double* Os, int batch_size, int params_size, double* Es_avg, double* Os_avg, double* OsEs_avg, double* OO_avg){
    double Es_sum = 0;
    for(int i=0; i<batch_size; i++)
        Es_sum += Es[i];
    *Es_avg = Es_sum/batch_size;

    for(int i=0; i<params_size; i++)
        Os_avg[i] = 0;
    for(int i=0; i<batch_size; i++)
        for(int j=0; j<params_size; j++)
            Os_avg[j] += Os[i*params_size+j];
    for(int i=0; i<params_size; i++)
        Os_avg[i] = Os_avg[i]/batch_size;

    double check_sum = 0;
    for(int i=0; i<params_size; i++)
        check_sum += Os_avg[i];
    printf("Os_avg_sum: %f, Os_avg[0]: %f, Os_avg[-1]: %f\n", check_sum, Os_avg[0], Os_avg[params_size-1]);

    char TRANSA = 'N';
    char TRANSB = 'T';
    double alpha = 1.0/batch_size;
    double beta = 0;
    int M = params_size;
    int N = params_size;
    int K = batch_size;
    int lda = M;
    int ldb = N;
    int ldc = M;
    double* OS_T = (double*)malloc(batch_size*params_size*sizeof(double));
    for(int i=0; i<batch_size*params_size; i++)
        OS_T[i] = Os[i];
    for(int i=0; i<params_size*params_size; i++)
        OO_avg[i] = 0;
    dgemm_(&TRANSA, &TRANSB, &M, &N, &K, &alpha, Os, &lda, OS_T, &ldb, &beta, OO_avg, &ldc);
    free(OS_T);

    char TRANS = 'N';
    for(int i=0; i<params_size; i++)
        OsEs_avg[i] = 0;
    int incx = 1;
    int incy = 1;
    dgemv_(&TRANS, &M, &K, &alpha, Os, &lda, Es, &incx, &beta, OsEs_avg, &incy);
}

void compute_grad(double* Os_avg, double* Es_avg, double* OsEs_avg, double* dt, int params_size, double* grad){
    for(int i=0; i<params_size; i++)
        grad[i] = OsEs_avg[i];
    
    int N = params_size;
    double alpha = -1.0 * (*Es_avg);
    int incx = 1;
    int incy = 1;
    
    daxpy_(&N, &alpha, Os_avg, &incx, grad, &incy);
    for(int i=0; i<params_size; i++)
        grad[i] =  -1.0 * (*dt) * grad[i];
}

void covariance_matrix(double* OO_avg, double* Os_avg, double shift, int params_size){
    char TRANSA = 'N';
    char TRANSB = 'T';
    double alpha = -1.0;
    double beta = 1.0;
    int M = params_size;
    int N = params_size;
    int K = 1;
    int lda = M;
    int ldb = N;
    int ldc = M;
    double* Os_avg_T = (double*)malloc(params_size*sizeof(double));
    for(int i=0; i<params_size; i++)
        Os_avg_T[i] = Os_avg[i];
    dgemm_(&TRANSA, &TRANSB, &M, &N, &K, &alpha, Os_avg, &lda, Os_avg_T, &ldb, &beta, OO_avg, &ldc);
    free(Os_avg_T);
    for(int i=0; i<params_size; i++)
        OO_avg[i*params_size+i] = OO_avg[i*params_size+i] + shift;
}

void compute_delta_scalapack(double* A, double* B, int N, double* C, int numProcess){

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int numKeep = N;

    int contxt;
    int nprow, npcol, myrow, mycol, info;
    int numProcScaLAPACK = numProcess;

    for(int i = IRound(sqrt(double(numProcScaLAPACK))); i <= numProcScaLAPACK; i++) {
        nprow = i;
        npcol = int(numProcScaLAPACK / nprow);
        if (nprow * npcol == numProcScaLAPACK) break;
    }
    int scaBlockSize = int(numKeep/nprow);

    Cblacs_get(0, 0, &contxt);
    Cblacs_gridinit(&contxt, "C", nprow, npcol);
    Cblacs_gridinfo(contxt, &nprow, &npcol, &myrow, &mycol);

    if(nprow>0 && npcol>0){

        int lda = numKeep;
        ScaLAPACKMatrix<double> square_mat_scala;
        Descriptor descReduceSeq, descReducePar;

        // Leading dimension provided
        descReduceSeq.Init(numKeep, numKeep, numKeep, numKeep, I_ZERO, I_ZERO, contxt, lda);

        // Automatically comptued Leading Dimension
        descReducePar.Init(numKeep, numKeep, scaBlockSize, scaBlockSize, I_ZERO, I_ZERO, contxt);

        square_mat_scala.SetDescriptor(descReducePar);
	    printf("start pdgemr2d\n");

        SCALAPACK(pdgemr2d)(&numKeep, &numKeep, A, &I_ONE, &I_ONE, descReduceSeq.Values(), 
                            &square_mat_scala.LocalMatrix()[0], &I_ONE, &I_ONE, square_mat_scala.Desc().Values(), &contxt);
        printf("pdgemr2d OK\n");

        SCALAPACK(pdpotrf)("L", &numKeep, square_mat_scala.Data(), &I_ONE, &I_ONE, square_mat_scala.Desc().Values(), &info);
        printf("pdpotrf OK\n");

        const int ht = square_mat_scala.Height();     
        // Set up v0
        dgdft::scalapack::Descriptor vec_desc;
        vec_desc.Init(ht, 1, scaBlockSize, scaBlockSize, 0, 0, contxt);   
        dgdft::scalapack::ScaLAPACKMatrix<double>  vec;
        vec.SetDescriptor(vec_desc);

        double* vec_data = vec.Data();
        int vec_height = vec.LocalHeight();
        int vec_width = vec.LocalWidth();

        SCALAPACK(pdgemr2d)(&numKeep, &I_ONE, B, &I_ONE, &I_ONE, descReduceSeq.Values(), 
                            &vec.LocalMatrix()[0], &I_ONE, &I_ONE, vec.Desc().Values(), &contxt);
        printf("pdgemr2d OK\n");

        SCALAPACK(pdtrsv)("L", "N", "N", &ht, square_mat_scala.Data(), &I_ONE, &I_ONE, square_mat_scala.Desc().Values(), 
                            vec.Data(), &I_ONE, &I_ONE, vec.Desc().Values(), &I_ONE);
        SCALAPACK(pdtrsv)("L", "T", "N", &ht, square_mat_scala.Data(), &I_ONE, &I_ONE, square_mat_scala.Desc().Values(), 
                            vec.Data(), &I_ONE, &I_ONE, vec.Desc().Values(), &I_ONE);
        printf("pdtrsv OK\n");

        for(int i=0; i<numKeep; i++)
            C[i] = 0;

        SCALAPACK(pdgemr2d)(&numKeep, &I_ONE, vec.Data(), &I_ONE, &I_ONE, vec.Desc().Values(),
                            C, &I_ONE, &I_ONE, descReduceSeq.Values(), &contxt);
        printf("pdgemr2d OK\n");

        Cblacs_gridexit(contxt);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

void compute_delta(double* A, double* B, int N, double* C){
    int INFO;
    char UPLO = 'L';

    int num = N;

    for(int i=0; i<num; i++)
       C[i] = B[i];

    printf("start call dpotrf_\n");
    dpotrf_(&UPLO, &num, A, &num, &INFO);
    printf("finish call dpotrf_\n");

    char TRANSA = 'N';
    char DIAG = 'N';
    int lda = N;
    int incc = 1;
    printf("start call dtrsv_\n");
    dtrsv_(&UPLO, &TRANSA, &DIAG, &num, A, &lda, C, &incc);
    TRANSA = 'T';
    dtrsv_(&UPLO, &TRANSA, &DIAG, &num, A, &lda, C, &incc);
    printf("finish call dtrsv_\n");
}

void ReadTensorNames(const std::string filename, std::vector<std::string> &vars_list, std::vector<std::string> &grads_and_vars_list){

    std::ifstream ifile(filename);
    std::ostringstream buf;
    char ch;
    while(buf&&ifile.get(ch))
        buf.put(ch);
    std::string str = buf.str();

    int pos = 0;
    std::vector<std::string> grads, vars;
    int shift = 0;
    while(pos < str.length()){
        if(str[pos] == '\''){
            std::string temp;
            ++pos;
            while(pos < str.length() && str[pos] != '\''){
                temp += str[pos];
                ++pos;
            }
            if(shift == 0){
                grads.push_back(temp);
                shift = 1;
            } else {
                vars.push_back(temp);
                shift = 0;
            }
        }
        ++pos;
    }
    for(int i = 0; i < grads.size(); ++i)
        grads_and_vars_list.push_back(grads[i]);

    for(int i = 0; i < vars.size(); ++i){
        grads_and_vars_list.push_back(vars[i]);
        vars_list.push_back(vars[i]);
    }
}

int GetTotalLength(std::vector<tensorflow::Tensor>& out_tensors, int number){
    int total_length = 0;
    for(int i = 0; i < number; ++i)
        total_length += out_tensors[i].flat<double>().size();
    return total_length;
}

void AssignVars(std::vector<tensorflow::Tensor>& out_tensors, double* vars_buffer, int number){
    int idx_base = 0;
    for(int i = 0; i < number; ++i){
        auto vars_length = out_tensors[i].flat<double>().size();
        auto vars_ptr = out_tensors[i].flat<double>().data();
        for(int ii = 0; ii < vars_length; ++ii)
            vars_ptr[ii] = vars_buffer[idx_base + ii];
        idx_base += vars_length;
    }
}

void ReadVars(std::vector<tensorflow::Tensor>& out_tensors, double* vars_buffer, int number){
    int idx_base = 0;
    for(int i = 0; i < number; ++i){
        auto vars_length = out_tensors[i].flat<double>().size();
        auto vars_ptr = out_tensors[i].flat<double>().data();
        for (int ii = 0; ii < vars_length; ii++)
            vars_buffer[idx_base + ii] = vars_ptr[ii];
        idx_base += vars_length;
    }
}

void UpdateVars(std::vector<tensorflow::Tensor>& out_tensors, double* delta_buffer, int number){
    int idx_base = 0;
    for(int i = 0; i < number; ++i){
        auto vars_length = out_tensors[i].flat<double>().size();
        auto vars_ptr = out_tensors[i].flat<double>().data();
        for(int ii = 0; ii < vars_length; ++ii)
            vars_ptr[ii] = vars_ptr[ii] + delta_buffer[idx_base + ii];
        idx_base += vars_length;
    }
}

void ReadLogits(std::vector<tensorflow::Tensor>& out_tensors, double* logits){
    auto logits_ptr = out_tensors[0].flat<double>().data();
    auto logits_length = out_tensors[0].flat<double>().size();
    for(int i=0; i<logits_length; i++)
        logits[i] = logits_ptr[i];
}

void ReadGradsLogits(std::vector<tensorflow::Tensor>& out_tensors, double* grads_buffer, double* logits, int number){
    int idx_grad_base = 0;
    for(int i = 0; i < number; ++i){
        auto grads_length = out_tensors[i].flat<double>().size();
        auto grads_ptr = out_tensors[i].flat<double>().data();
        for (int ii = 0; ii < grads_length; ii++)
            grads_buffer[idx_grad_base + ii] = grads_ptr[ii];
        idx_grad_base += grads_length;
    }
    auto logits_ptr = out_tensors[number].flat<double>().data();
    auto logits_length = out_tensors[number].flat<double>().size();
    for(int i=0; i<logits_length; i++)
        logits[i] = logits_ptr[i];
}


void get_forward_batch(tensorflow::Tensor& forward_tensor, int batch_size, std::vector<proposePrime>& batchPrime){
    int* forward_spin_lattice = (int*)malloc(batch_size*L*L*sizeof(int));
    for (int i=0; i<batch_size; i++)
        for (int j=0; j<L; j++)
            for (int k=0; k<L; k++)
                forward_spin_lattice[i*L*L+j*L+k] = batchPrime[i].spin_lattice[j*L+k];    

    for (int i=0; i<batch_size; i++){
        srand((unsigned)time(0));
        int forward_index = rand() % batchPrime[i].propose_J1.size();
        batchPrime[i].index = forward_index;
        int index_source =  batchPrime[i].propose_J1[forward_index].source;
        int index_target = batchPrime[i].propose_J1[forward_index].target;
        forward_spin_lattice[i*L*L+index_source] = batchPrime[i].spin_lattice[index_target];
        forward_spin_lattice[i*L*L+index_target] = batchPrime[i].spin_lattice[index_source];
    }

    int* batch_PBC = (int*)malloc(batch_size*(L+2*pad)*(L+2*pad)*sizeof(int));
    PBC(forward_spin_lattice, batch_PBC, batch_size);
    free(forward_spin_lattice);

    auto forward_tensor_mapped = forward_tensor.tensor<double, 4>();
    for (int i=0; i<batch_size; i++)
        for (int j=0; j<(L+2*pad); j++)
            for (int k=0; k<(L+2*pad); k++)
                forward_tensor_mapped(i, 0, j, k) = double(batch_PBC[i*(L+2*pad)*(L+2*pad)+j*(L+2*pad)+k]);
    free(batch_PBC);          
}

void get_sprime_batch(tensorflow::Tensor& sprime_tensor, int batch_size, int total_sprime_size, std::vector<proposePrime>& batchPrime, int* sign_result){
    int* sprime_spin_lattice = (int*)malloc(total_sprime_size*L*L*sizeof(int));
    int* start_sprime = sprime_spin_lattice;
    for(int i=0; i<batch_size; i++){
        int J1_size = batchPrime[i].propose_J1.size();
        int J2_size = batchPrime[i].propose_J2.size();
        int sprime_size = J1_size+J2_size;
        for(int j=0; j<sprime_size; j++)
            for(int k=0; k<L; k++)
                for(int m=0; m<L; m++)
                    start_sprime[j*L*L+k*L+m] = batchPrime[i].spin_lattice[k*L+m];
        for(int j=0; j<J1_size; j++){
            int index_source =  batchPrime[i].propose_J1[j].source;
            int index_target = batchPrime[i].propose_J1[j].target;
            start_sprime[j*L*L+index_source] = batchPrime[i].spin_lattice[index_target];
            start_sprime[j*L*L+index_target] = batchPrime[i].spin_lattice[index_source];
        }
        for(int j=0; j<J2_size; j++){
            int index_source =  batchPrime[i].propose_J2[j].source;
            int index_target = batchPrime[i].propose_J2[j].target;
            start_sprime[(j+J1_size)*L*L+index_source] = batchPrime[i].spin_lattice[index_target];
            start_sprime[(j+J1_size)*L*L+index_target] = batchPrime[i].spin_lattice[index_source];
        }
        start_sprime += sprime_size*L*L;
    }

    sign_rule(sprime_spin_lattice, total_sprime_size, sign_result);

    int* batch_PBC = (int*)malloc(total_sprime_size*(L+2*pad)*(L+2*pad)*sizeof(int));
    PBC(sprime_spin_lattice, batch_PBC, total_sprime_size);
    free(sprime_spin_lattice);

    auto sprime_tensor_mapped = sprime_tensor.tensor<double, 4>();
    for (int j = 0; j < total_sprime_size; j++)
        for (int k = 0; k < (L+2*pad); k++)
            for (int m = 0; m < (L+2*pad); m++)
                sprime_tensor_mapped(j, 0, k, m) = double(batch_PBC[j*(L+2*pad)*(L+2*pad)+k*(L+2*pad)+m]);
    free(batch_PBC);
}

void get_sprime_batch_list(std::vector<tensorflow::Tensor> &split_sprime_tensor_list, int split_size, int batch_size, int total_sprime_size, std::vector<proposePrime>& batchPrime, int* sign_result){
    int* sprime_spin_lattice = (int*)malloc(total_sprime_size*L*L*sizeof(int));
    int* start_sprime = sprime_spin_lattice;
    for(int i=0; i<batch_size; i++){
        int J1_size = batchPrime[i].propose_J1.size();
        int J2_size = batchPrime[i].propose_J2.size();
        int sprime_size = J1_size+J2_size;
        for(int j=0; j<sprime_size; j++)
            for(int k=0; k<L; k++)
                for(int m=0; m<L; m++)
                    start_sprime[j*L*L+k*L+m] = batchPrime[i].spin_lattice[k*L+m];
        for(int j=0; j<J1_size; j++){
            int index_source =  batchPrime[i].propose_J1[j].source;
            int index_target = batchPrime[i].propose_J1[j].target;
            start_sprime[j*L*L+index_source] = batchPrime[i].spin_lattice[index_target];
            start_sprime[j*L*L+index_target] = batchPrime[i].spin_lattice[index_source];
        }
        for(int j=0; j<J2_size; j++){
            int index_source =  batchPrime[i].propose_J2[j].source;
            int index_target = batchPrime[i].propose_J2[j].target;
            start_sprime[(j+J1_size)*L*L+index_source] = batchPrime[i].spin_lattice[index_target];
            start_sprime[(j+J1_size)*L*L+index_target] = batchPrime[i].spin_lattice[index_source];
        }
        start_sprime += sprime_size*L*L;
    }

    sign_rule(sprime_spin_lattice, total_sprime_size, sign_result);

    int* batch_PBC = (int*)malloc(total_sprime_size*(L+2*pad)*(L+2*pad)*sizeof(int));
    PBC(sprime_spin_lattice, batch_PBC, total_sprime_size);
    free(sprime_spin_lattice);

    int repeat_times = total_sprime_size/split_size;
    int start_index = 0;
    for (int i=0; i<repeat_times; i++){
        tensorflow::Tensor split_spin_lattice_tensor(tensorflow::DT_DOUBLE, tensorflow::TensorShape({split_size, channel, L+4, L+4}));
        auto split_spin_lattice_tensor_mapped = split_spin_lattice_tensor.tensor<double, 4>();
        for(int ii=0; ii<split_size; ii++)
            for (int j=0; j<(L+2*pad); j++)
                for (int k=0; k<(L+2*pad); k++)
                    split_spin_lattice_tensor_mapped(ii, 0, j, k) = double(batch_PBC[start_index+ii*(L+2*pad)*(L+2*pad)+j*(L+2*pad)+k]);
        split_sprime_tensor_list.push_back(split_spin_lattice_tensor);
        start_index += split_size*(L+2*pad)*(L+2*pad);
    }
    int tail_size = total_sprime_size%split_size;
    if(tail_size > 0){
        tensorflow::Tensor split_spin_lattice_tensor(tensorflow::DT_DOUBLE, tensorflow::TensorShape({tail_size, channel, L+4, L+4}));
        auto split_spin_lattice_tensor_mapped = split_spin_lattice_tensor.tensor<double, 4>();
        for(int ii=0; ii<tail_size; ii++)
            for (int j=0; j<(L+2*pad); j++)
                for (int k=0; k<(L+2*pad); k++)
                    split_spin_lattice_tensor_mapped(ii, 0, j, k) = double(batch_PBC[start_index+ii*(L+2*pad)*(L+2*pad)+j*(L+2*pad)+k]);
        split_sprime_tensor_list.push_back(split_spin_lattice_tensor);
    }
    free(batch_PBC);
}

void get_spin_lattice_batch(tensorflow::Tensor& batch_spin_lattice_tensor, int batch_size, int* batch_spin_lattice){
    int* batch_PBC = (int*)malloc(batch_size*(L+2*pad)*(L+2*pad)*sizeof(int));
    PBC(batch_spin_lattice, batch_PBC, batch_size);


    auto batch_spin_lattice_tensor_mapped = batch_spin_lattice_tensor.tensor<double, 4>();
    for (int i=0; i<batch_size; i++)
        for (int j=0; j<(L+2*pad); j++)
            for (int k=0; k<(L+2*pad); k++)
                batch_spin_lattice_tensor_mapped(i, 0, j, k) = double(batch_PBC[i*(L+2*pad)*(L+2*pad)+j*(L+2*pad)+k]);
    free(batch_PBC);          
}

void get_backward_tensor_list(std::vector<tensorflow::Tensor> &split_spin_lattice_tensor_list, int batch_size, int* batch_spin_lattice, int* sign_result){
    sign_rule(batch_spin_lattice, batch_size, sign_result);
    split_spin_lattice_tensor_list.clear();
    int* batch_PBC = (int*)malloc(batch_size*(L+2*pad)*(L+2*pad)*sizeof(int));
    PBC(batch_spin_lattice, batch_PBC, batch_size);
    
    int split_size = 1;

    for (int i=0; i<batch_size; i++){
        tensorflow::Tensor split_spin_lattice_tensor(tensorflow::DT_DOUBLE, tensorflow::TensorShape({split_size, channel, L+4, L+4}));
        auto split_spin_lattice_tensor_mapped = split_spin_lattice_tensor.tensor<double, 4>();
        for (int j=0; j<(L+2*pad); j++)
            for (int k=0; k<(L+2*pad); k++)
                split_spin_lattice_tensor_mapped(0, 0, j, k) = double(batch_PBC[i*(L+2*pad)*(L+2*pad)+j*(L+2*pad)+k]);
        split_spin_lattice_tensor_list.push_back(split_spin_lattice_tensor);
    }
    free(batch_PBC);

}

void update_batch_spin_lattice(std::vector<proposePrime>& batchPrime, int* batch_spin_lattice, int batch_size, double* ws_logits){
    double ws_new[batch_size];
    int fly_old[batch_size];

    for(int i=0; i<batch_size; i++){
        fly_old[i] = batchPrime[i].fly;
        srand((unsigned)time(0));
        double r = 0;
        double P = ws_logits[i]/batchPrime[i].ws;
        int tmp;
        if(bool(P*P>r)){
            int last_index = batchPrime[i].index;
            int index_source = batchPrime[i].propose_J1[last_index].source;
            int index_target = batchPrime[i].propose_J1[last_index].target;
            
            tmp = batchPrime[i].spin_lattice[index_source];
            batchPrime[i].spin_lattice[index_source] = batchPrime[i].spin_lattice[index_target];
            batchPrime[i].spin_lattice[index_target] = tmp;
            ws_new[i] = ws_logits[i];
        }else{
            ws_new[i] = batchPrime[i].ws;
        }
    }
    batch_sprime(batch_spin_lattice, batch_size, batchPrime);
    for(int i=0; i<batch_size; i++){
        batchPrime[i].ws = ws_new[i];
        batchPrime[i].fly = fly_old[i];
    }
}

int main(int argc, char* argv[]) {

    MPI_Init(NULL, NULL);

    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::cout.precision(20);

    const string graph_def_filename = "graph.pb";
    const string init_model_prefix = "model_restore";
    const string spin_lattice_prefix = "spin_lattice_restore";

    std::unique_ptr<tensorflow::Session> session_;
    tensorflow::GraphDef graph_def;
    TF_CHECK_OK(tensorflow::ReadBinaryProto(tensorflow::Env::Default(), graph_def_filename, &graph_def));
    session_.reset(tensorflow::NewSession(tensorflow::SessionOptions()));
    TF_CHECK_OK(session_->Create(graph_def));

    TF_CHECK_OK(session_->Run({}, {}, {"init"}, nullptr));
    printf("model init OK\n");

    std::vector<std::string> grads_and_vars_list;
    std::vector<std::string> vars_list;
    std::string grads_and_vars_list_path = "grads_and_vars.txt";
    ReadTensorNames(grads_and_vars_list_path, vars_list, grads_and_vars_list);
    int number_vars = vars_list.size();
    std::vector<std::string> logits_list;
    logits_list.push_back("logits");

    std::vector<tensorflow::Tensor> vars_tensors;
    TF_CHECK_OK(session_->Run({}, {vars_list}, {}, &vars_tensors));

    int total_length = GetTotalLength(vars_tensors, number_vars);
    double *vars_buffer = (double*)malloc(total_length*sizeof(double));

    int* init_spin_lattice = (int*)malloc(L*L*sizeof(int));
    int batch_size = 64;
    int* batch_spin_lattice = (int*)malloc(batch_size*L*L*sizeof(int));

    int resotre_size = 64;
    int* resotre_batch_spin_lattice = (int*)malloc(resotre_size*L*L*sizeof(int));
    int restore_step = 0;
    FILE *fp;

    if(rank==0){

        const string init_spin_lattice_path = "init_spin_lattice";
        if((fp=fopen(init_spin_lattice_path.c_str(), "r"))==NULL) {
            printf("Cannot open file.\n");
            exit(1);
        }
        fread(init_spin_lattice, sizeof(int), L*L, fp);
        fclose(fp);

        if(argc > 1){
            
            const string init_model_path = init_model_prefix + "_" + argv[1];
            if((fp=fopen(init_model_path.c_str(), "r"))==NULL) {
                printf("Cannot open file.\n");
                exit(1);
            }
            fread(vars_buffer, sizeof(double), total_length, fp);
            fclose(fp);

            const string init_spin_lattice_path = spin_lattice_prefix + "_" + argv[1];
            if((fp=fopen(init_spin_lattice_path.c_str(), "r"))==NULL) {
                printf("Cannot open file.\n");
                exit(1);
            }
            fread(resotre_batch_spin_lattice, sizeof(int), resotre_size*L*L, fp);
            fclose(fp);          
        } else {
            TF_CHECK_OK(session_->Run({}, {vars_list}, {}, &vars_tensors));
            ReadVars(vars_tensors, vars_buffer, number_vars);
        }
    }
    MPI_Bcast(vars_buffer, total_length, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    AssignVars(vars_tensors, vars_buffer, number_vars);
    MPI_Bcast(init_spin_lattice, L*L, MPI_INT, 0, MPI_COMM_WORLD);

    if(argc > 1){
        restore_step = atoi(argv[1]);
        MPI_Bcast(resotre_batch_spin_lattice, resotre_size*L*L, MPI_INT, 0, MPI_COMM_WORLD);
        //int start_batch_index = (rank * batch_size) % (resotre_size - batch_size);
        int start_batch_index = 0;
        for(int i=0; i<batch_size; i++)
            for(int j=0; j<L; j++)
                for(int k=0; k<L; k++)
                    batch_spin_lattice[i*L*L+j*L+k] = resotre_batch_spin_lattice[(start_batch_index+i)*L*L+j*L+k];
        free(resotre_batch_spin_lattice);
    } else {
        for(int i=0; i<batch_size; i++)
            for(int j=0; j<L; j++)
                for(int k=0; k<L; k++)
                    batch_spin_lattice[i*L*L+j*L+k] = init_spin_lattice[j*L+k];
    }

    double* logits_buffer = (double*)malloc(batch_size*sizeof(double));
    tensorflow::Tensor batch_spin_lattice_tensor(tensorflow::DT_DOUBLE, tensorflow::TensorShape({batch_size, channel, L+4, L+4}));
    get_spin_lattice_batch(batch_spin_lattice_tensor, batch_size, batch_spin_lattice); 
    std::vector<tensorflow::Tensor> logits_tensors;
    TF_CHECK_OK(session_->Run({{"spin_lattice", batch_spin_lattice_tensor}}, {logits_list}, {}, &logits_tensors));
    ReadLogits(logits_tensors, logits_buffer);

    std::vector<proposePrime> target_batch_prime;
    batch_sprime(batch_spin_lattice, batch_size, target_batch_prime);
    for(int i=0; i<batch_size; i++)
        target_batch_prime[i].ws = logits_buffer[i];

    int step;
    int total_steps = 3;
    tensorflow::Tensor forward_batch_tensor(tensorflow::DT_DOUBLE, tensorflow::TensorShape({batch_size, channel, L+4, L+4}));

    std::vector<std::string> grads_logits_list;
    for(int i=0; i<number_vars; i++)
        grads_logits_list.push_back(grads_and_vars_list[i]);
    grads_logits_list.push_back("logits");
    double dt = 0.01;
    double shift = 0.01;

    int* sign_batch_result = (int*)malloc(batch_size*sizeof(int));
    double* Os_list = (double*)malloc(batch_size*total_length*sizeof(double));
    double* Es_list = (double*)malloc(batch_size*sizeof(double));
    
    double* Os_avg = (double*)malloc(total_length*sizeof(double));
    double* Es_avg = (double*)malloc(sizeof(double));
    double* OsEs_avg = (double*)malloc(total_length*sizeof(double));
    double* OO_avg = (double*)malloc(total_length*total_length*sizeof(double));

    double* first_order_grad_data = (double*)malloc(total_length*sizeof(double));
    double* delta = (double*)malloc(total_length*sizeof(double));

    for(step=restore_step+1; step<total_steps; step++){
        printf("step: %d\n", step);
        
/*
        get_forward_batch(forward_batch_tensor, batch_size, target_batch_prime);
        TF_CHECK_OK(session_->Run({{"spin_lattice", forward_batch_tensor}}, {logits_list}, {}, &logits_tensors));
        ReadLogits(logits_tensors, logits_buffer);
        for(int i=0; i<batch_size; i++){
            std::cout << logits_buffer[i] << "\n";
        }  
        update_batch_spin_lattice(target_batch_prime, batch_spin_lattice, batch_size, logits_buffer);
        for(int i=0; i<batch_size; i++){
            std::cout << target_batch_prime[i].ws << "\n";
        }

*/
        if(step%11 == 10){
            int* gather_spin_lattice = (int*)malloc(size*L*L*sizeof(int));
            MPI_Gather(batch_spin_lattice, L*L, MPI_INT, gather_spin_lattice, L*L, MPI_INT, 0, MPI_COMM_WORLD);
            
            if(rank == 0){
                TF_CHECK_OK(session_->Run({}, {vars_list}, {}, &vars_tensors));
                ReadVars(vars_tensors, vars_buffer, number_vars);
                const string init_model_path = init_model_prefix + "_" + std::to_string(step);
                if((fp=fopen(init_model_path.c_str(), "w"))==NULL) {
                    printf("Cannot open file.\n");
                    exit(1);
                }
                fwrite(vars_buffer, sizeof(double), total_length, fp);
                fclose(fp);

                const string init_spin_lattice_path = spin_lattice_prefix + "_" + std::to_string(step);
                if((fp=fopen(init_spin_lattice_path.c_str(), "w"))==NULL) {
                    printf("Cannot open file.\n");
                    exit(1);
                }
                fwrite(gather_spin_lattice, sizeof(int), resotre_size*L*L, fp);
                fclose(fp);
            }
            free(gather_spin_lattice);
        }
        if(step%1 == 0){
            double* grads_ptr = Os_list;
            std::vector<tensorflow::Tensor> backward_tensor_list;
            get_backward_tensor_list(backward_tensor_list, batch_size, batch_spin_lattice, sign_batch_result);
            
            for(int i=0; i<batch_size; i++){
                std::vector<tensorflow::Tensor> grads_logits_tensors;
                TF_CHECK_OK(session_->Run({{"spin_lattice", backward_tensor_list[i]}}, {grads_logits_list}, {}, &grads_logits_tensors));
                ReadGradsLogits(grads_logits_tensors, grads_ptr, logits_buffer+i, number_vars);
                double checksum;
                for(int j=0; j<total_length; j++){
                    grads_ptr[j] = grads_ptr[j]/logits_buffer[i];
                }
                target_batch_prime[i].ws = logits_buffer[i];
                checksum = 0;
                for(int j=0; j<total_length; j++)
                    checksum += grads_ptr[j];
                grads_ptr += total_length;
                printf("i: %d grads_checksum: %f\n", i, checksum);
            }

            int total_sprime_size = 0;
            for(int i=0; i<batch_size; i++)
                total_sprime_size = total_sprime_size + target_batch_prime[i].propose_J1.size() + target_batch_prime[i].propose_J2.size();

            int* sign_sprime_result = (int*)malloc(total_sprime_size*sizeof(int));
            double* logits_sprime_result = (double*)malloc(total_sprime_size*sizeof(double));

            std::vector<tensorflow::Tensor> sprime_batch_tensor_list;
            int split_size = 1024;
            get_sprime_batch_list(sprime_batch_tensor_list, split_size, batch_size, total_sprime_size, target_batch_prime, sign_sprime_result);
            for(int i=0; i<sprime_batch_tensor_list.size(); i++){
                TF_CHECK_OK(session_->Run({{"spin_lattice", sprime_batch_tensor_list[i]}}, {logits_list}, {}, &logits_tensors));
                ReadLogits(logits_tensors, logits_sprime_result+split_size*i); 
            }
    
            int start_prime_index = 0;
            for(int i=0; i<batch_size; i++){
                double J1_ws = 0;
                double J2_ws = 0;
                double sign_ws = target_batch_prime[i].ws * sign_batch_result[i];
                int J1_size = target_batch_prime[i].propose_J1.size();
                int J2_size = target_batch_prime[i].propose_J2.size();
                for(int j=0; j<J1_size; j++)
                    J1_ws = J1_ws + logits_sprime_result[start_prime_index+j]*sign_sprime_result[start_prime_index+j]/sign_ws;
                start_prime_index += J1_size;
                for(int j=0; j<J2_size; j++)
                    J2_ws = J2_ws + logits_sprime_result[start_prime_index+j]*sign_sprime_result[start_prime_index+j]/sign_ws;
                start_prime_index += J2_size;
                Es_list[i] = target_batch_prime[i].energy + 2 * (J1_ws + J2*J2_ws);
                printf("i: %d Es_list: %f\n", i, Es_list[i]);
            }
            free(sign_sprime_result);
            free(logits_sprime_result);
            
            int accept_samples_size;
            accept_samples_size = select_samples(Es_list, Os_list, batch_size, total_length, target_batch_prime, init_spin_lattice, rank);
            if(accept_samples_size == 0){
                *Es_avg = 0;
                for(int i=0; i<total_length; i++)
                    Os_avg[i] = 0;
                for(int i=0; i<total_length; i++)
                    OsEs_avg[i] = 0;
                for(int i=0; i<total_length*total_length; i++)
                    OO_avg[i] = 0;
            } else {
                calculate_parameter(Es_list, Os_list, accept_samples_size, total_length, Es_avg, Os_avg, OsEs_avg, OO_avg);
            }

            MPI_Allreduce(MPI_IN_PLACE, Es_avg, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, Os_avg, total_length, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, OsEs_avg, total_length, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, OO_avg, total_length*total_length, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            *Es_avg = *Es_avg/size;
            for(int i=0; i<total_length; i++)
                Os_avg[i] = Os_avg[i]/size;
            for(int i=0; i<total_length; i++)
                OsEs_avg[i] = OsEs_avg[i]/size;
            for(int i=0; i<total_length*total_length; i++)
                OO_avg[i] = OO_avg[i]/size;

            printf("Rank: %d, Size: %d, Es_avg: %f, Os_avg: %f, %f\n", rank, size, *Es_avg, Os_avg[0], Os_avg[1]);
            compute_grad(Os_avg, Es_avg, OsEs_avg, &dt, total_length, first_order_grad_data);
            covariance_matrix(OO_avg, Os_avg, shift, total_length);
            //compute_delta(OO_avg, first_order_grad_data, total_length, delta);
            int numProcess = size;
            compute_delta_scalapack(OO_avg, first_order_grad_data, total_length, delta, numProcess);
            MPI_Bcast(delta, total_length, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            TF_CHECK_OK(session_->Run({}, {vars_list}, {}, &vars_tensors));
            double checksum = 0;
            for(int i=0; i<10; i++)
                std::cout << delta[i] << " ";
            std::cout << std::endl;
            for(int j=0; j<total_length; j++)
                checksum += delta[j];
            std::cout << "Rank: " << rank << " Size: " << size << " delta: " << checksum << std::endl;
            std::cout << std::endl;
            UpdateVars(vars_tensors, delta, number_vars);
        }
    }
    free(vars_buffer);
    free(sign_batch_result);
    free(Es_list);
    free(Os_list);

    free(Es_avg);
    free(Os_avg);
    free(OsEs_avg);
    free(OO_avg);

    free(first_order_grad_data);
    free(delta);
    free(logits_buffer);
    free(init_spin_lattice);
    free(batch_spin_lattice);

    MPI_Finalize();

    return 0;
}
