#ifndef DEFINE_DIAGRAMS_H
#define DEFINE_DIAGRAMS_H

#include <vector>
#include <complex>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

void compute_diagrams(std::vector< std::complex<double> > &res, 
                     const std::vector<Eigen::Tensor<std::complex<double>,4>> &bprops,
                     const std::vector<Eigen::Tensor<std::complex<double>,4>> &bsinks);


#endif
