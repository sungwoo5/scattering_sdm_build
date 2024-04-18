#ifndef COMPUTE_T_GPU_H
#define COMPUTE_T_GPU_H

#include "global_variables.h"

#include <complex>
#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Dense>

Eigen::Tensor<std::complex<double>,4> create_Tsink_cublas(std::vector<Eigen::MatrixXcd> &evectors, int t, int px, int py, int pz);
Eigen::Tensor<std::complex<double>,4> create_Tsink_gpu_v2(std::vector<Eigen::MatrixXcd> &evectors, int t, int px, int py, int pz);
Eigen::Tensor<std::complex<double>,4> create_Tsink_gpu(std::vector<Eigen::MatrixXcd> &evectors, int t, int px, int py, int pz);
Eigen::Tensor<std::complex<double>,4> create_Tsource_gpu_v2(std::vector<Eigen::MatrixXcd> &evectors, int t, int px, int py, int pz);
Eigen::Tensor<std::complex<double>,4> create_Tsource_gpu(std::vector<Eigen::MatrixXcd> &evectors, int t, int px, int py, int pz);

#endif
