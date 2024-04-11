#ifndef COMPUTE_BPROP_GPU_H
#define COMPUTE_BPROP_GPU_H

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <complex>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <global_variables.h>
#include <spin_sparse.h>	/* Spin4 */

using Tensor2 = Eigen::Tensor<cd,2>;
using Tensor4 = Eigen::Tensor<cd,4>;

Tensor4 multiply_Tensor2xTensor4(const Tensor2 A,
				 const Tensor4 B);

Tensor4 compute_Bprop_gpu(vector<Spin4> &spintensor, 
			  const Tensor2 omega, 
			  const Tensor4 tsource,
			  int NSPIN,
			  int NVEC
			  );

Tensor4 compute_Bsink_gpu(vector<Spin4> &spintensor,
			  const Tensor4 tsink,
			  int NSPIN,
			  int NVEC
			  );

#endif
