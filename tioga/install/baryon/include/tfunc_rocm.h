#include "hip/hip_runtime.h"
#ifndef TFUNC_H
#define TFUNC_H

#include <hip/hip_complex.h>

__global__ void calc_phase(hipDoubleComplex *d_phase, 
			   int px, int py, int pz, int NX);

__global__ void calc_baryonTfunc(hipDoubleComplex *baryonTfunc, 
				 hipDoubleComplex *evecs_ft,
				 int NVEC);

__global__ void ft_evecs_readphase(hipDoubleComplex *res, 
				   hipDoubleComplex *v0, 
				   hipDoubleComplex *phase);

__global__ void tfunc_sink_spatialsum_print(hipDoubleComplex *res,
    hipDoubleComplex *v0, hipDoubleComplex *v1, hipDoubleComplex *v2, hipDoubleComplex *v3,
    int px, int py, int pz, int NX);

__global__ void tfunc_sink_spatialsum(hipDoubleComplex *res,
    hipDoubleComplex *v0, hipDoubleComplex *v1, hipDoubleComplex *v2, hipDoubleComplex *v3,
    int px, int py, int pz, int NX);

__global__ void tfunc_sink_spatialsum_readphase(hipDoubleComplex *res,
						hipDoubleComplex *v0, hipDoubleComplex *v1, hipDoubleComplex *v2, hipDoubleComplex *v3,
						hipDoubleComplex *phase,
						int px, int py, int pz, int NX);

__global__ void tfunc_sink_spatialsum_readphase_v2(hipDoubleComplex *res,
						hipDoubleComplex *v0, hipDoubleComplex *v1, hipDoubleComplex *v2, hipDoubleComplex *v3,
						   hipDoubleComplex *phase, int NX);
__global__ void tfunc_source_spatialsum_readphase_v2(hipDoubleComplex *res,
						hipDoubleComplex *v0, hipDoubleComplex *v1, hipDoubleComplex *v2, hipDoubleComplex *v3,
						   hipDoubleComplex *phase, int NX);

__global__ void tfunc_source_spatialsum(hipDoubleComplex *res,
    hipDoubleComplex *v0, hipDoubleComplex *v1, hipDoubleComplex *v2, hipDoubleComplex *v3,
    int px, int py, int pz, int NX);

#endif
