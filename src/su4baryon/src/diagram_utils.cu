#include "diagram_utils.h"
#include "gpu_kernel.h"

#include <complex>
#include <iostream>
#include "timer.h"

using namespace std;
using Tensor4 = Eigen::Tensor<std::complex<double>,4>;
using cd = std::complex<double>;




// CUDA runtime
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define MAX_BLOCK 65535 //2^15=32768, 2^16=65536
#define MAX_THREADS 1024

__global__ void gpu_permutation_rank4(cuDoubleComplex *out,
				  cuDoubleComplex *in, 
				  int N,
				  size_t base0, size_t base1, size_t base2, size_t base3)
{
  size_t idx=blockIdx.x*blockDim.x + threadIdx.x; 
  size_t idx_permutation;
  int i[4], tmp;
  i[0]=idx%N;
  tmp=idx/N;
  i[1]=tmp%N;
  tmp=tmp/N;
  i[2]=tmp%N;
  tmp=tmp/N;
  i[3]=tmp%N;

  // bruteforce permutation!
  idx_permutation = i[0]*base0 + i[1]*base1 + i[2]*base2 + i[3]*base3;  
  out[idx_permutation] = in[idx];
}

// calculate n^m
size_t power_integer(int n, int m)
{
  size_t tmp=1;
  for(int i=0; i<m; i++)
    tmp=tmp*n;
  return tmp;
}

void inverse_permutation(std::vector<int>& sinv, const std::vector<int>& s)
{
  for(int i=0; i<sinv.size(); i++)
    {
      int j=0;
      do{
	if(s[j]==i) break;    
	j++;
      }while(j<s.size());
      sinv[i]=j;
    }
}


// Permutation axis of B (rank4) and save into A (rank4)
// B[ l,k,j,i ] = B[ l*N^0 + k*N^1 + j*N^2 + i*N^3 ] <-- column major
//              = B[ idx ]
//
// ex) B->B' by trivial permutation (identity): s={0,1,2,3} -> sinv=s
// B[ l,k,j,i ] = B'[ l*N^0 + k*N^1 + j*N^2 + i*N^3 ]
//                B'[ l*N^s[0] + k*N^s[1] + j*N^s[2] + i*N^s[3] ]
//
// ex) B->B' by permutation the last two axis: s={0,1,3,2} -> sinv=s
// B[ l,k,j,i ] = B'[ l,k,i,j ] 
//                B'[ l*N^0 + k*N^1 + i*N^2 + j*N^3 ]
//                B'[ l*N^s[0] + k*N^s[1] + j*N^s[2] + i*N^s[3] ]
//
// ex) B->B' by permutation over three axis: s={1,2,0,3} -> sinv={2,0,1,3}
// B[ l,k,j,i ] = B'[ k,j,l,i ] 
//                B'[ k*N^0 + j*N^1 + l*N^2 + i*N^3 ]
//                B'[ l*N^sinv[0] + j*N^sinv[1] + k*N^sinv[2] + i*N^sinv[3] ] -> general expression
void permutation_rank4(cuDoubleComplex *d_out,
		       cuDoubleComplex *d_in, 
		       size_t N,
		       const std::vector<int> &s)
{
  size_t base[4];
  std::vector<int> sinv(s.size());
  
  inverse_permutation(sinv, s);

  // this part is common for all threads,
  // so doesn't have to be within this gpu kernel
  base[0]=power_integer(N,sinv[0]);
  base[1]=power_integer(N,sinv[1]);
  base[2]=power_integer(N,sinv[2]);
  base[3]=power_integer(N,sinv[3]);

  // permutation d_B
  int threads;
  size_t size = N*N*N*N;
  for(threads = MAX_THREADS; threads >= 1; threads--){
    if( size%threads == 0) break;
  }
  int block = size/threads;
  // std::cout << "block: " << block <<  ", threads: "<< threads << std::endl;
  gpu_permutation_rank4<<<block, threads >>>(d_out, d_in, N, 
					 base[0], base[1], base[2], base[3]);
} 



// replace the original cuTensor function 
// with cuBLAS but keeps the function name for the backward compatibility
// THIS VERSION uses gpu for permutation
cd cuTensor_contract(const Tensor4 &tensA, const Tensor4 &tensB, std::vector<std::vector<int>> diagram)
{
  int dim = tensA.dimensions()[0];
  size_t tdim = dim*dim*dim*dim;

  // cout<< "new cuTensor_contract for CUBLAS" << endl;
  // cd *A = new cd[tdim];
  // cd *B = new cd[tdim];
  cd res(0.,0.);

  const cd *A = tensA.data();		// assume diagram[i][0]->3,2,1,0, no need to permutation
  const cd *B = tensB.data();

  // CUDA
  cudaError status;
  cublasHandle_t handle;	// reuse as much as we can
  cublasCreate(&handle);

  // allocate device memory
  cuDoubleComplex *d_A, *d_B, *d_B_permutation;
  cuDoubleComplex *d_res;
  size_t mem_size_A=tdim*sizeof(cd);
  size_t mem_size_B=tdim*sizeof(cd);

  cudaMalloc((void **) &d_B, mem_size_B);
  cudaMalloc((void **) &d_B_permutation, mem_size_B);
  status = cudaMemcpy(d_B, B, mem_size_B, cudaMemcpyHostToDevice);
  if(status != cudaSuccess)
    {
      fprintf(stderr, "cudaMemcpyHostToDevice for d_B\n");
      exit(0);
    }

  // indexing test for tensA
  if ( diagram[3][0]!=0  ||  diagram[2][0]!=1  ||  diagram[1][0]!=2  ||  diagram[0][0]!=3 )
    {
      fprintf(stderr, "index for tensA is not trivial\n");
      exit(0);      
    }  

  // permutation of tensB, the rank 4 tensor
  vector<int> s(4);			
  s[0]=diagram[3][1];
  s[1]=diagram[2][1];
  s[2]=diagram[1][1];
  s[3]=diagram[0][1];

  permutation_rank4(d_B_permutation, d_B, dim, s);
  cudaFree(d_B);  

  // DOT PRODUCT
  cudaMalloc((void **) &d_A, mem_size_A);
  status = cudaMemcpy(d_A, A, mem_size_A, cudaMemcpyHostToDevice);
  if(status != cudaSuccess)
    {
      fprintf(stderr, "cudaMemcpyHostToDevice for d_A\n");
      exit(0);
    }
  cudaMalloc((void **) &d_res, sizeof(cd));
  cublasZdotu(handle, tdim,
	      d_A, 1,
	      d_B_permutation, 1,
	      d_res);

  // copy result to host
  cd *h_res = (cd *)malloc((int)sizeof(cd));
  status = cudaMemcpy(h_res, d_res, sizeof(cd), cudaMemcpyDeviceToHost);
  if(status != cudaSuccess)
    {
      fprintf(stderr, "cudaMemcpyHostToDevice for d_res\n");
      exit(0);
    }
  res=h_res[0];

  // free
  cudaFree(d_A);
  cudaFree(d_B_permutation);
  cudaFree(d_res);
  free(h_res);
  cublasDestroy(handle);

  return res;
}
