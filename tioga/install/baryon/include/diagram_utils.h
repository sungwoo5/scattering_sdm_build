#ifndef DIAGRAM_UTILS_H
#define DIAGRAM_UTILS_H

#include <vector>
#include <complex>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>


int epsilon(const std::vector<int> &indices);
std::complex<double> cuTensor_contract(const Eigen::Tensor<std::complex<double>,4> &tensA, 
                                       const Eigen::Tensor<std::complex<double>,4> &tensB,
                                       std::vector< std::vector<int> > diag);

#endif
