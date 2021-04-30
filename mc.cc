#include "mc.h"

int L = 10;
int K = 5;
int pad = (K-1)/2;
int channel = 1;
double J2 = 0.5;

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