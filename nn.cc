#include "nn.h"

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