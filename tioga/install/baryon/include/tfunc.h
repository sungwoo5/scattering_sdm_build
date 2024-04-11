#ifndef TFUNC_H
#define TFUNC_H

#include <cuComplex.h>

__global__ void calc_phase(cuDoubleComplex *d_phase, 
			   int px, int py, int pz, int NX);

__global__ void calc_baryonTfunc(cuDoubleComplex *baryonTfunc, 
				 cuDoubleComplex *evecs_ft,
				 int NVEC);

__global__ void ft_evecs_readphase(cuDoubleComplex *res, 
				   cuDoubleComplex *v0, 
				   cuDoubleComplex *phase);

__global__ void tfunc_sink_spatialsum_print(cuDoubleComplex *res,
    cuDoubleComplex *v0, cuDoubleComplex *v1, cuDoubleComplex *v2, cuDoubleComplex *v3,
    int px, int py, int pz, int NX);

__global__ void tfunc_sink_spatialsum(cuDoubleComplex *res,
    cuDoubleComplex *v0, cuDoubleComplex *v1, cuDoubleComplex *v2, cuDoubleComplex *v3,
    int px, int py, int pz, int NX);

__global__ void tfunc_sink_spatialsum_readphase(cuDoubleComplex *res,
						cuDoubleComplex *v0, cuDoubleComplex *v1, cuDoubleComplex *v2, cuDoubleComplex *v3,
						cuDoubleComplex *phase,
						int px, int py, int pz, int NX);

__global__ void tfunc_sink_spatialsum_readphase_v2(cuDoubleComplex *res,
						cuDoubleComplex *v0, cuDoubleComplex *v1, cuDoubleComplex *v2, cuDoubleComplex *v3,
						   cuDoubleComplex *phase, int NX);
__global__ void tfunc_source_spatialsum_readphase_v2(cuDoubleComplex *res,
						cuDoubleComplex *v0, cuDoubleComplex *v1, cuDoubleComplex *v2, cuDoubleComplex *v3,
						   cuDoubleComplex *phase, int NX);

__global__ void tfunc_source_spatialsum(cuDoubleComplex *res,
    cuDoubleComplex *v0, cuDoubleComplex *v1, cuDoubleComplex *v2, cuDoubleComplex *v3,
    int px, int py, int pz, int NX);

#endif
