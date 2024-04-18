#include "compute_t_gpu.h"
#include "tfunc_rocm.h"
#include "diagram_utils.h"

#include <hip/hip_runtime.h>
#include <Eigen/Dense>

#include <complex>
#include <iostream>

#include <iostream>
#include "timer.h"

#include <hipblas/hipblas.h>

using Tensor4 = Eigen::Tensor<std::complex<double>,4>;
using cd = std::complex<double>;

#define MAX_BLOCK 65535 //2^15=32768, 2^16=65536
#define MAX_THREADS 1024

void outerprod_rank4_cublas(hipDoubleComplex **d0,
			    hipDoubleComplex **d1, 
			    hipDoubleComplex **d2, 
			    hipDoubleComplex **d3,
			    hipDoubleComplex **d_A,
			    hipDoubleComplex **d_B,
			    hipDoubleComplex **d_res,
			    hipDoubleComplex *alpha,
			    hipblasHandle_t *handle,
			    int N
			    )
{

  // CUBLAS version 2.0
  {
    hipDoubleComplex one = make_hipDoubleComplex(1.0, 0.);
    hipDoubleComplex zero = make_hipDoubleComplex(0.0, 0.);

    // Note that Zgeru was replaced by Zgemm
    // Zgeru requires a separate hipMemset to initialize which takes quite amount of time
    // but Zgemm can do the initialization at once


    // Timer<> timer_dA("dA");
    // d0[i] X d1[j]^T = d_A[i,j]=d_A[i+N*j == ij]
    // hipblasZgeru_v2(*handle, N, N,
    // 		&one, 
    // 		*d0, 1,
    // 		*d1, 1,
    // 		*d_A, N);
    hipblasZgemm_v2(*handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
		N, N, 1,
		&one,
		*d0, N,
		*d1, 1,
		&zero,
		*d_A, N);
    // timer_dA.stop<std::chrono::nanoseconds>("ns");
    
    // d2[k] X d3[l]^T = d_B[k,l]=d_B[k+N*l == kl]
    // hipblasZgeru_v2(*handle, N, N,
    // 		&one, 
    // 		*d2, 1,
    // 		*d3, 1,
    // 		*d_B, N);
    hipblasZgemm_v2(*handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
		N, N, 1,
		&one,
		*d2, N,
		*d3, 1,
		&zero,
		*d_B, N);

    // d_A[ij] X d_B[kl]^T = d_AB[ij,kl]=d_AB[ij+N*N*kl]
    // ijkl == ij+N*N*kl
    //       = i+N*j + N*N*( k+N*l )
    //       = i + N*j + N*N*k + N*N*N*l
    //       = [i,j,k,l] rank 4 index in column major
    // Timer<> timer_dAB("dAB");
    hipblasZgeru_v2(*handle, N*N, N*N,
		alpha, 
		*d_A, 1,
		*d_B, 1,
		*d_res, N*N);
    // timer_dAB.stop<std::chrono::nanoseconds>("ns");
  }
}

// evecs: should be in [t][i][ color*vol + site ] ordering from get_evecs()
Tensor4 create_Tsink_cublas(std::vector<Eigen::MatrixXcd> &evecs, int t, int px, int py, int pz)
{
  // CUDA
  hipError_t status;
  hipblasHandle_t handle;	// reuse as much as we can
  hipblasCreate(&handle);
  

  // copy evec to device
  hipDoubleComplex *d_evecs_0;	// original, before transpose
  hipDoubleComplex *d_evecs;	// after transpose
  int vol=Lx*Lx*Lx;		// spatial volume
  hipMalloc((void **) &d_evecs_0, vol*Nvec*Nc*sizeof(std::complex<double>));
  hipMalloc((void **) &d_evecs,   vol*Nvec*Nc*sizeof(std::complex<double>));

  for(int l=0; l<Nvec; ++l){
    Eigen::VectorXcd evec = evecs[t].row(l);
    hipMemcpy(d_evecs_0 + l*Nc*vol, 
	       evec.data(), 
	       Nc*vol*sizeof(cd),
	       hipMemcpyHostToDevice);
  }
 
  // transpose of evecs indices
  // ax==a*vol+x
  // d_evecs_0[ i*Nc*vol + ax ] = d_evecs[ ax*Nvec + i ]
  // -> in column major cuda convention:
  // d_evecs_0[ ax, i ] = d_evecs[ i, ax ]
  {
    hipDoubleComplex alpha = make_hipDoubleComplex(1.0, 0.);
    hipDoubleComplex beta = make_hipDoubleComplex(0.0, 0.);
    // BLAS-like extension, in-place mode for C=B
    // C= alpha*Tr(A) + 0*C
    hipblasZgeam_v2(handle, HIPBLAS_OP_T, HIPBLAS_OP_N,
		Nvec, Nc*vol,
		&alpha,
		d_evecs_0, Nc*vol,
		&beta,
		d_evecs, Nvec,
		d_evecs, Nvec);
  }  
  hipFree(d_evecs_0);		// free the original array


  // // phase factor precalculation
  // int threads;
  // for(threads = MAX_THREADS; threads >= 1; threads--){
  //   if( vol%threads == 0) break;
  // }
  // int block = vol/threads;
  // hipDoubleComplex *d_phase;
  // hipMalloc((void **) &d_phase, vol*sizeof(cd));
  // calc_phase<<<block, threads>>>(d_phase, px, py, pz, Lx);
  // hipDoubleComplex *h_phase;
  // hipMemcpy(h_phase, d_phase, vol*sizeof(cd),
  // 	     hipMemcpyDeviceToHost);

  
  // runs
  hipDoubleComplex *d_A, *d_B, *d_res; // *d_AB, *d_ABphase, 
  hipMalloc((void **) &d_A,   Nvec*Nvec*sizeof(cd)); // temp array from d1 X d2
  hipMalloc((void **) &d_B,   Nvec*Nvec*sizeof(cd)); // temp array from d3 X d4

  size_t size_tfunc=Nvec*Nvec*Nvec*Nvec*sizeof(cd);
  hipMalloc((void **) &d_res, size_tfunc); // result rank4 from d1 X d2 X d3 X d4
  
  // initalize d_res to 0
  hipMemset(d_res, 0, size_tfunc);

  for(int x=0; x<vol; ++x){

    // phase calculated here, the load should be minor
    // somehow the precalculated phase data makes an error when cublas routine reads it..
    hipDoubleComplex phase;
    int _z=x%Lx;
    int tmp=x/Lx;
    int _y=tmp%Lx;
    tmp=tmp/Lx;
    int _x=tmp%Lx;
    sincos(-2.0*M_PI*(double(px*_x+py*_y+pz*_z))/((double)Lx), &phase.y, &phase.x);
    // printf(" %f %f ", phase.y, phase.x);

    for(int c0=0; c0<4; ++c0)
    for(int c1=0; c1<4; ++c1)
    for(int c2=0; c2<4; ++c2)
    for(int c3=0; c3<4; ++c3)
      {
    	double eps=(c0-c1)*(c0-c2)*(c0-c3)*(c1-c2)*(c1-c3)*(c2-c3)/12;
    	hipDoubleComplex alpha = hipCmul(make_hipDoubleComplex(eps, 0.), phase);
    	if(eps!=0)
    	  {
	    int c0x=c0*vol+x;
	    int c1x=c1*vol+x;
	    int c2x=c2*vol+x;
	    int c3x=c3*vol+x;
	    
	    hipDoubleComplex *d0,*d1,*d2,*d3;
	    d0 = d_evecs + c0x*Nvec;
	    d1 = d_evecs + c1x*Nvec;
	    d2 = d_evecs + c2x*Nvec;
	    d3 = d_evecs + c3x*Nvec;

	    outerprod_rank4_cublas(&d0,
				   &d1,
				   &d2,
				   &d3,
				   &(d_A), &(d_B), &(d_res),
				   &alpha,
				   &handle, Nvec);
	    
    	  }
      }    
  }
  

  // return to the host
  Tensor4 baryonTFunc(Nvec,Nvec,Nvec,Nvec);
  cd *h_baryonTFunc=baryonTFunc.data();

  hipMemcpy(h_baryonTFunc, d_res, size_tfunc, hipMemcpyDeviceToHost);

  hipFree(d_evecs);
  hipFree(d_A);
  hipFree(d_B);
  hipFree(d_res);
  hipblasDestroy(handle);
  return baryonTFunc;
} 


Tensor4 create_Tsink_gpu_v2(std::vector<Eigen::MatrixXcd> &evecs, int t, int px, int py, int pz)
{
  Tensor4 baryonTFunc(Nvec,Nvec,Nvec,Nvec);
  baryonTFunc.setConstant(std::complex<double>(0.,0.));

  dim3 blocksize(1024);
  // std::cout << "dim3 blocksize(1024): " << blocksize.x << blocksize.y << blocksize.z << std::endl;
  

  int NBLOCKS=(Lx*Lx*Lx+1023)/1024;
  dim3 gridsize(NBLOCKS);

  // int threads;
  // size_t size = Lx*Lx*Lx;
  // for(threads = MAX_THREADS; threads >= 1; threads--){
  //   if( size%threads == 0) break;
  // }
  // size_t block = size/threads;
  // int threads=1024;
  // size_t block = 1;
  // std::cout << "block: " << block <<  ", threads: "<< threads << std::endl;

  std::complex<double> res[1]; 
  
  hipDoubleComplex *d_result;
  hipMalloc((void **) &d_result, sizeof(std::complex<double>)); 

  hipDoubleComplex *d_evecs;
  hipMalloc((void **) &d_evecs, Lx*Lx*Lx*Nvec*Nc*sizeof(std::complex<double>));

  for(int l=0; l<Nvec; ++l)
  {
    Eigen::VectorXcd evec = evecs[t].row(l);
    
    hipMemcpy(d_evecs + l*Nc*Lx*Lx*Lx, evec.data(), Nc*Lx*Lx*Lx*sizeof(std::complex<double>), hipMemcpyHostToDevice);
  }
    

  hipDoubleComplex *d_phase;
  hipMalloc((void **) &d_phase, Lx*Lx*Lx*sizeof(std::complex<double>));
  calc_phase<<<gridsize, blocksize>>>(d_phase, px, py, pz, Lx);

  size_t volNc=Nc*Lx*Lx*Lx;

  for(int l0=0; l0<Nvec; ++l0)
  for(int l1=0; l1<Nvec; ++l1)
  for(int l2=0; l2<Nvec; ++l2)
  for(int l3=0; l3<Nvec; ++l3)
    {
      if( l0!=l1 && l0!=l2 && l0!=l3 && l1!=l2 && l1!=l3 && l2!=l3 )
      {

        hipMemset(d_result, 0, sizeof(std::complex<double>));    
        res[0]=std::complex<double>(0,0);
        
        hipDeviceSynchronize();
        
        tfunc_sink_spatialsum_readphase_v2<<<gridsize, blocksize>>>(d_result,
							       d_evecs + l0*volNc, 
							       d_evecs + l1*volNc, 
							       d_evecs + l2*volNc, 
							       d_evecs + l3*volNc, 
							       d_phase, Lx); 
        hipDeviceSynchronize();
        
        hipMemcpy(res, d_result, sizeof(std::complex<double>), hipMemcpyDeviceToHost);
        baryonTFunc(l0,l1,l2,l3) = res[0];
      } // end if anti-symmetric component
    } // end evec loop

  if(d_result)
    hipFree(d_result);
  if(d_evecs)
    hipFree(d_evecs);

  return baryonTFunc;
} 



Tensor4 create_Tsink_gpu(std::vector<Eigen::MatrixXcd> &evecs, int t, int px, int py, int pz)
{
  Tensor4 baryonTFunc(Nvec,Nvec,Nvec,Nvec);
  baryonTFunc.setConstant(std::complex<double>(0.,0.));

  dim3 blocksize(1024);
  

  int NBLOCKS=(Lx*Lx*Lx+1023)/1024;
  dim3 gridsize(NBLOCKS);

  std::complex<double> res[1]; 
  
  hipDoubleComplex *d_result;
  hipMalloc((void **) &d_result, sizeof(std::complex<double>)); 

  hipDoubleComplex *d_evecs;
  hipMalloc((void **) &d_evecs, Lx*Lx*Lx*Nvec*Nc*sizeof(std::complex<double>));

  for(int l=0; l<Nvec; ++l)
  {
    Eigen::VectorXcd evec = evecs[t].row(l);
    
    hipMemcpy(d_evecs + l*Nc*Lx*Lx*Lx, evec.data(), Nc*Lx*Lx*Lx*sizeof(std::complex<double>), hipMemcpyHostToDevice);
  }
    

  hipDoubleComplex *d_phase;
  hipMalloc((void **) &d_phase, Lx*Lx*Lx*sizeof(std::complex<double>));
  calc_phase<<<gridsize, blocksize>>>(d_phase, 
				      px, py, pz, Lx);


  for(int c0=0; c0<Nc; ++c0)
  for(int c1=0; c1<Nc; ++c1)
  for(int c2=0; c2<Nc; ++c2)
  for(int c3=0; c3<Nc; ++c3)
  {
    auto eps = epsilon(std::vector<int>{c0,c1,c2,c3});
    if(eps!=0)
    {
      auto ecmplx = std::complex<double>((double)eps,0.);
      for(int l0=0; l0<Nvec; ++l0)
      for(int l1=0; l1<Nvec; ++l1)
      for(int l2=0; l2<Nvec; ++l2)
      for(int l3=0; l3<Nvec; ++l3)
      {
      if( l0!=l1 && l0!=l2 && l0!=l3 && l1!=l2 && l1!=l3 && l2!=l3 )
      {
/*
        if(l0==0 && l1==2 && l2==1 && l3==3)
        { 
        std::cout << "running gpu-sum for |"
                  << c0 << "," 
                  << c1 << "," 
                  << c2 << "," 
                  << c3 << "," 
                  << l0 << "," 
                  << l1 << "," 
                  << l2 << "," 
                  << l3 << std::endl;
        }
*/
        hipMemset(d_result, 0, sizeof(std::complex<double>));    
        res[0]=std::complex<double>(0,0);
        
        hipDeviceSynchronize();
        
        tfunc_sink_spatialsum_readphase<<<gridsize, blocksize>>>(d_result,
						       d_evecs + l0*Nc*Lx*Lx*Lx + c0*Lx*Lx*Lx, 
						       d_evecs + l1*Nc*Lx*Lx*Lx + c1*Lx*Lx*Lx, 
						       d_evecs + l2*Nc*Lx*Lx*Lx + c2*Lx*Lx*Lx, 
						       d_evecs + l3*Nc*Lx*Lx*Lx + c3*Lx*Lx*Lx, 
						       d_phase,
						       px, py, pz, Lx); 
        hipDeviceSynchronize();
        
        hipMemcpy(res, d_result, sizeof(std::complex<double>), hipMemcpyDeviceToHost);
       /* 
        if(l0==0 && l1==2 && l2==1 && l3==3)
        { 
       std::cout << "gpu-sum res |"
                  << c0 << "," 
                  << c1 << "," 
                  << c2 << "," 
                  << c3 << "," 
                  << l0 << "," 
                  << l1 << "," 
                  << l2 << "," 
                  << l3 << " | " << res[0] << std::endl;
        }
        */
        baryonTFunc(l0,l1,l2,l3) = baryonTFunc(l0,l1,l2,l3) + ecmplx*res[0];
      } // end if anti-symmetric component
      } // end evec loop
    } // end eps if
  } // end color loop

  if(d_result)
    hipFree(d_result);
  if(d_evecs)
    hipFree(d_evecs);

  return baryonTFunc;
} 




Tensor4 create_Tsource_gpu(std::vector<Eigen::MatrixXcd> &evecs, int t, int px, int py, int pz)
{
  Tensor4 baryonTFunc(Nvec,Nvec,Nvec,Nvec);
  baryonTFunc.setConstant(std::complex<double>(0.,0.));

  dim3 blocksize(1024);

  int NBLOCKS=(Lx*Lx*Lx+1023)/1024;
  dim3 gridsize(NBLOCKS);

  std::complex<double> res[1]; 
  
  hipDoubleComplex *d_result;
  hipMalloc((void **) &d_result, sizeof(std::complex<double>)); 

  hipDoubleComplex *d_evecs;
  hipMalloc((void **) &d_evecs, Lx*Lx*Lx*Nvec*Nc*sizeof(std::complex<double>));

  for(int l=0; l<Nvec; ++l)
  {
    Eigen::VectorXcd evec = evecs[t].row(l);
    
    hipMemcpy(d_evecs + l*Nc*Lx*Lx*Lx, evec.data(), Nc*Lx*Lx*Lx*sizeof(std::complex<double>), hipMemcpyHostToDevice);
  }
    

  for(int c0=0; c0<Nc; ++c0)
  for(int c1=0; c1<Nc; ++c1)
  for(int c2=0; c2<Nc; ++c2)
  for(int c3=0; c3<Nc; ++c3)
  {
    auto eps = epsilon(std::vector<int>{c0,c1,c2,c3});
    if(eps!=0)
    {
      auto ecmplx = std::complex<double>((double)eps,0.);
      for(int l0=0; l0<Nvec; ++l0)
      for(int l1=0; l1<Nvec; ++l1)
      for(int l2=0; l2<Nvec; ++l2)
      for(int l3=0; l3<Nvec; ++l3)
      {
      if( l0!=l1 && l0!=l2 && l0!=l3 && l1!=l2 && l1!=l3 && l2!=l3 )
      {
        hipMemset(d_result, 0, sizeof(std::complex<double>));    
        
        res[0]=std::complex<double>(0,0);
        
        hipDeviceSynchronize();
        
        tfunc_source_spatialsum<<<gridsize, blocksize>>>(d_result, 
							 d_evecs + l0*Nc*Lx*Lx*Lx + c0*Lx*Lx*Lx,
							 d_evecs + l1*Nc*Lx*Lx*Lx + c1*Lx*Lx*Lx,
							 d_evecs + l2*Nc*Lx*Lx*Lx + c2*Lx*Lx*Lx,
							 d_evecs + l3*Nc*Lx*Lx*Lx + c3*Lx*Lx*Lx,
							 px, py, pz, Lx);
        hipDeviceSynchronize();
        
        hipMemcpy(res, d_result, sizeof(std::complex<double>), hipMemcpyDeviceToHost);
        
       
        baryonTFunc(l0,l1,l2,l3) = baryonTFunc(l0,l1,l2,l3) + ecmplx*res[0];
      } // end if anti-symmetric component
      } // end evec loop
    } // end eps if
  } // end color loop

  if(d_result)
    hipFree(d_result);
  if(d_evecs)
    hipFree(d_evecs);

  return baryonTFunc;
} 







Tensor4 create_Tsource_gpu_v2(std::vector<Eigen::MatrixXcd> &evecs, int t, int px, int py, int pz)
{
  Tensor4 baryonTFunc(Nvec,Nvec,Nvec,Nvec);
  baryonTFunc.setConstant(std::complex<double>(0.,0.));

  dim3 blocksize(1024);
  

  int NBLOCKS=(Lx*Lx*Lx+1023)/1024;
  dim3 gridsize(NBLOCKS);

  // int threads;
  // size_t size = Lx*Lx*Lx;
  // for(threads = MAX_THREADS; threads >= 1; threads--){
  //   if( size%threads == 0) break;
  // }
  // size_t block = size/threads;
  // int threads=1024;
  // size_t block = 1;
  // std::cout << "block: " << block <<  ", threads: "<< threads << std::endl;

  std::complex<double> res[1]; 
  
  hipDoubleComplex *d_result;
  hipMalloc((void **) &d_result, sizeof(std::complex<double>)); 

  hipDoubleComplex *d_evecs;
  hipMalloc((void **) &d_evecs, Lx*Lx*Lx*Nvec*Nc*sizeof(std::complex<double>));

  for(int l=0; l<Nvec; ++l)
  {
    Eigen::VectorXcd evec = evecs[t].row(l);
    
    hipMemcpy(d_evecs + l*Nc*Lx*Lx*Lx, evec.data(), Nc*Lx*Lx*Lx*sizeof(std::complex<double>), hipMemcpyHostToDevice);
  }
    

  hipDoubleComplex *d_phase;
  hipMalloc((void **) &d_phase, Lx*Lx*Lx*sizeof(std::complex<double>));
  calc_phase<<<gridsize, blocksize>>>(d_phase, px, py, pz, Lx);

  size_t volNc=Nc*Lx*Lx*Lx;

  for(int l0=0; l0<Nvec; ++l0)
  for(int l1=0; l1<Nvec; ++l1)
  for(int l2=0; l2<Nvec; ++l2)
  for(int l3=0; l3<Nvec; ++l3)
    {
      if( l0!=l1 && l0!=l2 && l0!=l3 && l1!=l2 && l1!=l3 && l2!=l3 )
      {

        hipMemset(d_result, 0, sizeof(std::complex<double>));    
        res[0]=std::complex<double>(0,0);
        
        hipDeviceSynchronize();
        
        tfunc_source_spatialsum_readphase_v2<<<gridsize, blocksize>>>(d_result,
							       d_evecs + l0*volNc, 
							       d_evecs + l1*volNc, 
							       d_evecs + l2*volNc, 
							       d_evecs + l3*volNc, 
							       d_phase, Lx); 
        hipDeviceSynchronize();
        
        hipMemcpy(res, d_result, sizeof(std::complex<double>), hipMemcpyDeviceToHost);
        baryonTFunc(l0,l1,l2,l3) = res[0];
      } // end if anti-symmetric component
    } // end evec loop

  if(d_result)
    hipFree(d_result);
  if(d_evecs)
    hipFree(d_evecs);

  return baryonTFunc;
} 

