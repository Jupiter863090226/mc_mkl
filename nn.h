#ifndef NN_H_
#define NN_H_

#include <fstream>
#include <sstream>
#include <utility>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include<time.h>
#include<stdlib.h>

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

#include "mc.h"

using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;
using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

void ReadTensorNames(const std::string filename, std::vector<std::string> &vars_list, std::vector<std::string> &grads_and_vars_list);

int GetTotalLength(std::vector<tensorflow::Tensor>& out_tensors, int number);

void AssignVars(std::vector<tensorflow::Tensor>& out_tensors, double* vars_buffer, int number);

void ReadVars(std::vector<tensorflow::Tensor>& out_tensors, double* vars_buffer, int number);

void UpdateVars(std::vector<tensorflow::Tensor>& out_tensors, double* delta_buffer, int number);

void ReadLogits(std::vector<tensorflow::Tensor>& out_tensors, double* logits);

void ReadGradsLogits(std::vector<tensorflow::Tensor>& out_tensors, double* grads_buffer, double* logits, int number);

void get_forward_batch(tensorflow::Tensor& forward_tensor, int batch_size, std::vector<proposePrime>& batchPrime);

void get_sprime_batch(tensorflow::Tensor& sprime_tensor, int batch_size, int total_sprime_size, std::vector<proposePrime>& batchPrime, int* sign_result);

void get_sprime_batch_list(std::vector<tensorflow::Tensor> &split_sprime_tensor_list, int split_size, int batch_size, int total_sprime_size, std::vector<proposePrime>& batchPrime, int* sign_result);

void get_spin_lattice_batch(tensorflow::Tensor& batch_spin_lattice_tensor, int batch_size, int* batch_spin_lattice);

void get_backward_tensor_list(std::vector<tensorflow::Tensor> &split_spin_lattice_tensor_list, int batch_size, int* batch_spin_lattice, int* sign_result);

void update_batch_spin_lattice(std::vector<proposePrime>& batchPrime, int* batch_spin_lattice, int batch_size, double* ws_logits);

#endif