#include "mc.h"
#include "nn.h"

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