#ifndef UTILS_H
#define UTILS_H


#include <vector>
#include <complex>
#include <unsupported/Eigen/CXX11/Tensor>

bool nearlyequal2(std::complex<double> a, std::complex<double> b, double TOL=1e-8);

bool compare_Tfuncs(Eigen::Tensor<std::complex<double>,4> &A, Eigen::Tensor<std::complex<double>,4> &B, int N);

int epsilon(const std::vector<int> &indices);

#endif
