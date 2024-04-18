#include "hip/hip_runtime.h"
#include "tfunc_rocm.h"

//#include "sm_60_atomic_functions.h"

#include <stdio.h>
#include <cmath>


__global__ void calc_phase(hipDoubleComplex *phase, 
    int px, int py, int pz, int NX)
{
  //spatial-index
  int idx=blockIdx.x*blockDim.x + threadIdx.x; 

  if(idx<NX*NX*NX)
  {

    int z=idx%NX;
    int tmp=idx/NX;
    int y=tmp%NX;
    tmp=tmp/NX;
    int x=tmp%NX;
  

    // hipDoubleComplex phase;
    sincos(-2.0*M_PI*(double(px*x+py*y+pz*z))/((double)NX), &phase[idx].y, &phase[idx].x);
  }  
  
}

// __device__ int epsilon(const vector<int> &indices)
// {
//   ///make sure the epsilon tensor isn't zero.
//   ///by finding the first instance of two indices being equal
//   vector<int> idx=indices;
//   if( idx[0]==idx[1] || idx[0]==idx[2] || idx[0]==idx[3] || idx[1]==idx[2] || idx[1]==idx[3] || idx[2]==idx[3] )
//     return 0;

//   return (idx[0]-idx[1])*(idx[0]-idx[2])*(idx[0]-idx[3])*(idx[1]-idx[2])*(idx[1]-idx[3])*(idx[2]-idx[3])/12;
// }

// epsilon_{c0,c1,c2,c3}*v0[c0]*v1[c1]*v2[c2]*v3[c3]
// 4!=24 terms
__device__ hipDoubleComplex __baryonTfunc(hipDoubleComplex *v0,
					 hipDoubleComplex *v1,
					 hipDoubleComplex *v2,
					 hipDoubleComplex *v3)
{
  hipDoubleComplex tmp;
  tmp.x=0.0;
  tmp.y=0.0;

  // tmp+=v0[0]*v1[1]*v2[2]*v3[3];
  // tmp-=v0[0]*v1[1]*v2[3]*v3[2];
  // tmp-=v0[0]*v1[2]*v2[1]*v3[3];
  // tmp+=v0[0]*v1[2]*v2[3]*v3[1];
  // tmp+=v0[0]*v1[3]*v2[1]*v3[2];
  // tmp-=v0[0]*v1[3]*v2[2]*v3[1];
  			      
  // tmp+=v1[0]*v2[1]*v3[2]*v0[3];
  // tmp-=v1[0]*v2[1]*v3[3]*v0[2];
  // tmp-=v1[0]*v2[2]*v3[1]*v0[3];
  // tmp+=v1[0]*v2[2]*v3[3]*v0[1];
  // tmp+=v1[0]*v2[3]*v3[1]*v0[2];
  // tmp-=v1[0]*v2[3]*v3[2]*v0[1];
  			      
  // tmp+=v2[0]*v3[1]*v0[2]*v1[3];
  // tmp-=v2[0]*v3[1]*v0[3]*v1[2];
  // tmp-=v2[0]*v3[2]*v0[1]*v1[3];
  // tmp+=v2[0]*v3[2]*v0[3]*v1[1];
  // tmp+=v2[0]*v3[3]*v0[1]*v1[2];
  // tmp-=v2[0]*v3[3]*v0[2]*v1[1];
  			      
  // tmp+=v3[0]*v0[1]*v1[2]*v2[3];
  // tmp-=v3[0]*v0[1]*v1[3]*v2[2];
  // tmp-=v3[0]*v0[2]*v1[1]*v2[3];
  // tmp+=v3[0]*v0[2]*v1[3]*v2[1];
  // tmp+=v3[0]*v0[3]*v1[1]*v2[2];
  // tmp-=v3[0]*v0[3]*v1[2]*v2[1];

  // tmp=hipCadd(tmp,hipCmul(hipCmul(v0[0],v1[1]),hipCmul(v2[2],v3[3])));
  // tmp=hipCsub(tmp,hipCmul(hipCmul(v0[0],v1[1]),hipCmul(v2[3],v3[2])));
  // tmp=hipCsub(tmp,hipCmul(hipCmul(v0[0],v1[2]),hipCmul(v2[1],v3[3])));
  // tmp=hipCadd(tmp,hipCmul(hipCmul(v0[0],v1[2]),hipCmul(v2[3],v3[1])));
  // tmp=hipCadd(tmp,hipCmul(hipCmul(v0[0],v1[3]),hipCmul(v2[1],v3[2])));
  // tmp=hipCsub(tmp,hipCmul(hipCmul(v0[0],v1[3]),hipCmul(v2[2],v3[1])));

  // tmp=hipCadd(tmp,hipCmul(hipCmul(v1[0],v2[1]),hipCmul(v3[2],v0[3])));
  // tmp=hipCsub(tmp,hipCmul(hipCmul(v1[0],v2[1]),hipCmul(v3[3],v0[2])));
  // tmp=hipCsub(tmp,hipCmul(hipCmul(v1[0],v2[2]),hipCmul(v3[1],v0[3])));
  // tmp=hipCadd(tmp,hipCmul(hipCmul(v1[0],v2[2]),hipCmul(v3[3],v0[1])));
  // tmp=hipCadd(tmp,hipCmul(hipCmul(v1[0],v2[3]),hipCmul(v3[1],v0[2])));
  // tmp=hipCsub(tmp,hipCmul(hipCmul(v1[0],v2[3]),hipCmul(v3[2],v0[1])));

  // tmp=hipCadd(tmp,hipCmul(hipCmul(v2[0],v3[1]),hipCmul(v0[2],v1[3])));
  // tmp=hipCsub(tmp,hipCmul(hipCmul(v2[0],v3[1]),hipCmul(v0[3],v1[2])));
  // tmp=hipCsub(tmp,hipCmul(hipCmul(v2[0],v3[2]),hipCmul(v0[1],v1[3])));
  // tmp=hipCadd(tmp,hipCmul(hipCmul(v2[0],v3[2]),hipCmul(v0[3],v1[1])));
  // tmp=hipCadd(tmp,hipCmul(hipCmul(v2[0],v3[3]),hipCmul(v0[1],v1[2])));
  // tmp=hipCsub(tmp,hipCmul(hipCmul(v2[0],v3[3]),hipCmul(v0[2],v1[1])));			      

  // tmp=hipCadd(tmp,hipCmul(hipCmul(v3[0],v0[1]),hipCmul(v1[2],v2[3])));
  // tmp=hipCsub(tmp,hipCmul(hipCmul(v3[0],v0[1]),hipCmul(v1[3],v2[2])));
  // tmp=hipCsub(tmp,hipCmul(hipCmul(v3[0],v0[2]),hipCmul(v1[1],v2[3])));
  // tmp=hipCadd(tmp,hipCmul(hipCmul(v3[0],v0[2]),hipCmul(v1[3],v2[1])));
  // tmp=hipCadd(tmp,hipCmul(hipCmul(v3[0],v0[3]),hipCmul(v1[1],v2[2])));
  // tmp=hipCsub(tmp,hipCmul(hipCmul(v3[0],v0[3]),hipCmul(v1[2],v2[1])));
  
  for(int c0=0; c0<4; ++c0)
  for(int c1=0; c1<4; ++c1)
  for(int c2=0; c2<4; ++c2)
  for(int c3=0; c3<4; ++c3)
  {
    double eps=(c0-c1)*(c0-c2)*(c0-c3)*(c1-c2)*(c1-c3)*(c2-c3)/12;
    if(eps!=0)
    {
      hipDoubleComplex ecmplx = make_hipDoubleComplex(eps, 0.);
      tmp=hipCadd(tmp,hipCmul(hipCmul(hipCmul(v0[c0],v1[c1]),
				   hipCmul(v2[c2],v3[c3])), 
			    ecmplx));
    }}
      
  return tmp;
}
			    

__global__ void calc_baryonTfunc(hipDoubleComplex *baryonTfunc, 
				 hipDoubleComplex *evecs_ft,
				 int NVEC)
{
  // index for Nvec*Nvec*Nvec*Nvec
  // idx = l0*Nvec*Nvec*Nvec + l1*Nvec*Nvec + l2*Nvec + l3
  int idx=blockIdx.x*blockDim.x + threadIdx.x; 

  int l0,l1,l2,l3,tmp;
  l3=idx%NVEC;
  tmp=idx/NVEC;
  l2=tmp%NVEC;
  tmp=idx/NVEC;
  l1=tmp%NVEC;
  tmp=idx/NVEC;
  l0=tmp%NVEC;  
  
  // baryonTfunc[idx]=__baryonTfunc(&(evecs_ft[l0]),
  // 			       &(evecs_ft[l0]),
  // 			       &(evecs_ft[l0]),
  // 			       &(evecs_ft[l0]));
			       
  baryonTfunc[idx]=__baryonTfunc(evecs_ft+l0*4,
				 evecs_ft+l1*4,
				 evecs_ft+l2*4,
				 evecs_ft+l3*4);
			       
  // baryonTfunc(&baryonTfunc[idx], 
  // 	      evecs_ft+l0,
  // 	      evecs_ft+l1,
  // 	      evecs_ft+l2,
  // 	      evecs_ft+l3);

  
}


__global__ void ft_evecs_readphase(hipDoubleComplex *res, 
    hipDoubleComplex *v0, 
    hipDoubleComplex *phase)
{
  //spatial-index
  int idx=blockIdx.x*blockDim.x + threadIdx.x; 

  __shared__ hipDoubleComplex sdata[1024];
  sdata[threadIdx.x].x=0;
  sdata[threadIdx.x].y=0;

  // multiply phase
  sdata[threadIdx.x] = hipCmul(v0[idx],phase[idx]);

  // block reduce using atomic function
  for(int s=1; s<blockDim.x; s*=2)
  {
    int index = 2*s*threadIdx.x;
    if(index<blockDim.x)
    {
      sdata[index]=hipCadd(sdata[index],sdata[index+s]);  
    }

    __syncthreads();
  }

  if(threadIdx.x==0)
  {
      atomicAdd(&(res[0].x), sdata[0].x);
      atomicAdd(&(res[0].y), sdata[0].y);
  }

}

__global__ void tfunc_sink_spatialsum_print(hipDoubleComplex *res, 
    hipDoubleComplex *v0, hipDoubleComplex *v1, hipDoubleComplex *v2, hipDoubleComplex *v3,
    int px, int py, int pz, int NX)
{
  //spatial-index
  int idx=blockIdx.x*blockDim.x + threadIdx.x; 
  // Do I need memory bounds check?
  // in principle no but keeping for now
  
  __shared__ hipDoubleComplex sdata[1024];
  sdata[threadIdx.x].x=0;
  sdata[threadIdx.x].y=0;

  if(idx<NX*NX*NX)
  {
//    if(idx==9)
//      printf("idx=%d, v0=%f,%f  v1=%f,%f  v2=%f,%f  v3=%f,%f", idx,
//          v0[idx].x, v0[idx].y, v1[idx].x, v1[idx].y,
//          v2[idx].x, v2[idx].y, v3[idx].x, v3[idx].y);

    int z=idx%NX;
    int tmp=idx/NX;
    int y=tmp%NX;
    tmp=tmp/NX;
    int x=tmp%NX;
  

    hipDoubleComplex phase;
    sincos(-2.0*M_PI*(double(px*x+py*y+pz*z))/((double)NX), &phase.y, &phase.x);
    
   // printf("GPUid: block.x=%d, blockDim.x=%d, thread.x=%d, idx=%d\n", blockIdx.x, blockDim.x, threadIdx.x, idx);

    //hipDoubleComplex val = hipCmul(hipCmul(hipCmul(v0[idx],v1[idx]),hipCmul(v2[idx],v3[idx])),phase); 

    // Do reduction in the shared space
    // (v0*v1) * (v2*v3) * phase
    sdata[threadIdx.x] = hipCmul(hipCmul(hipCmul(v0[idx],v1[idx]),hipCmul(v2[idx],v3[idx])),phase); 
    
  /*  
    printf("GPUSum site=%d, v0=(%e,%e), v1=(%e,%e), v2=(%e,%e), v3=(%e,%e), phase=(%e,%e), val=(%e,%e), threadIdx.x=%d, blockIdx.x=%d \n", idx, 
        v0[idx].x ,v0[idx].y, 
        v1[idx].x, v1[idx].y, 
        v2[idx].x, v2[idx].y, 
        v3[idx].x, v3[idx].y,
        phase.x, phase.y, 
        sdata[threadIdx.x].x, sdata[threadIdx.x].y, threadIdx.x, blockIdx.x);
*/
    // sdata seems to have all the right values on the threads.
  }  
    
  __syncthreads();
  
  for(int s=1; s<blockDim.x; s*=2)
  {
    int index = 2*s*threadIdx.x;
    if(index<blockDim.x)
    {
      sdata[index]=hipCadd(sdata[index],sdata[index+s]);  
    }

    __syncthreads();
  }

  if(threadIdx.x==0)
  {
   // printf("attomic adding from blockIdx.x=%d, res[0]=(%e,%e), sdata[0]=(%e,%e)", blockIdx.x, res[0].x, res[0].y, sdata[0].x, sdata[0].y);
      atomicAdd(&(res[0].x), sdata[0].x);
      atomicAdd(&(res[0].y), sdata[0].y);
      //atomicAdd(&(res[blockIdx.x].x), sdata[0].x);
      //atomicAdd(&(res[blockIdx.x].y), sdata[0].y);

//    res[blockIdx.x]=sdata[0];
  }
}

__global__ void tfunc_sink_spatialsum(hipDoubleComplex *res, 
    hipDoubleComplex *v0, hipDoubleComplex *v1, hipDoubleComplex *v2, hipDoubleComplex *v3,
    int px, int py, int pz, int NX)
{
  //spatial-index
  int idx=blockIdx.x*blockDim.x + threadIdx.x; 
  // Do I need memory bounds check?
  // in principle no but keeping for now
  
  __shared__ hipDoubleComplex sdata[1024];
  sdata[threadIdx.x].x=0;
  sdata[threadIdx.x].y=0;

  if(idx<NX*NX*NX)
  {
//    if(idx==9)
//      printf("idx=%d, v0=%f,%f  v1=%f,%f  v2=%f,%f  v3=%f,%f", idx,
//          v0[idx].x, v0[idx].y, v1[idx].x, v1[idx].y,
//          v2[idx].x, v2[idx].y, v3[idx].x, v3[idx].y);

    int z=idx%NX;
    int tmp=idx/NX;
    int y=tmp%NX;
    tmp=tmp/NX;
    int x=tmp%NX;
  

    hipDoubleComplex phase;
    sincos(-2.0*M_PI*(double(px*x+py*y+pz*z))/((double)NX), &phase.y, &phase.x);
    
   // printf("GPUid: block.x=%d, blockDim.x=%d, thread.x=%d, idx=%d\n", blockIdx.x, blockDim.x, threadIdx.x, idx);

    //hipDoubleComplex val = hipCmul(hipCmul(hipCmul(v0[idx],v1[idx]),hipCmul(v2[idx],v3[idx])),phase); 

    // Do reduction in the shared space
    sdata[threadIdx.x] = hipCmul(hipCmul(hipCmul(v0[idx],v1[idx]),hipCmul(v2[idx],v3[idx])),phase); 
    // sdata seems to have all the right values on the threads.
  }  
    
  __syncthreads();
  
  for(int s=1; s<blockDim.x; s*=2)
  {
    int index = 2*s*threadIdx.x;
    if(index<blockDim.x)
    {
      sdata[index]=hipCadd(sdata[index],sdata[index+s]);  
    }

    __syncthreads();
  }

  if(threadIdx.x==0)
  {
      atomicAdd(&(res[0].x), sdata[0].x);
      atomicAdd(&(res[0].y), sdata[0].y);
      //atomicAdd(&(res[blockIdx.x].x), sdata[0].x);
      //atomicAdd(&(res[blockIdx.x].y), sdata[0].y);

//    res[blockIdx.x]=sdata[0];
  }
}


__global__ void tfunc_sink_spatialsum_readphase(hipDoubleComplex *res, 
    hipDoubleComplex *v0, hipDoubleComplex *v1, hipDoubleComplex *v2, hipDoubleComplex *v3, 
    hipDoubleComplex *phase,
    int px, int py, int pz, int NX)
{
  //spatial-index
  int idx=blockIdx.x*blockDim.x + threadIdx.x; 
  // Do I need memory bounds check?
  // in principle no but keeping for now
  
  __shared__ hipDoubleComplex sdata[1024];
  sdata[threadIdx.x].x=0;
  sdata[threadIdx.x].y=0;

  if(idx<NX*NX*NX)
  {

    int z=idx%NX;
    int tmp=idx/NX;
    int y=tmp%NX;
    tmp=tmp/NX;
    int x=tmp%NX;
  
    // Do reduction in the shared space
    sdata[threadIdx.x] = hipCmul(hipCmul(hipCmul(v0[idx],v1[idx]),hipCmul(v2[idx],v3[idx])),phase[idx]); 
    // sdata seems to have all the right values on the threads.
  }  
    
  __syncthreads();
  
  for(int s=1; s<blockDim.x; s*=2)
  {
    int index = 2*s*threadIdx.x;
    if(index<blockDim.x)
    {
      sdata[index]=hipCadd(sdata[index],sdata[index+s]);  
    }

    __syncthreads();
  }

  if(threadIdx.x==0)
  {
      atomicAdd(&(res[0].x), sdata[0].x);
      atomicAdd(&(res[0].y), sdata[0].y);
  }
}



__global__ void tfunc_sink_spatialsum_readphase_v2(hipDoubleComplex *res, 
    hipDoubleComplex *v0, hipDoubleComplex *v1, hipDoubleComplex *v2, hipDoubleComplex *v3, 
						   hipDoubleComplex *phase, int NX)
{
  //spatial-index
  int idx=blockIdx.x*blockDim.x + threadIdx.x; 
  // Do I need memory bounds check?
  // in principle no but keeping for now
  
  __shared__ hipDoubleComplex sdata[1024];
  sdata[threadIdx.x].x=0;
  sdata[threadIdx.x].y=0;

  if(idx<NX*NX*NX)
  {

  hipDoubleComplex tmp;
  
  // // Do reduction in the shared space
  // sdata[threadIdx.x] = hipCmul(hipCmul(hipCmul(v0[idx],v1[idx]),hipCmul(v2[idx],v3[idx])),phase[idx]); 
  // Do reduction in the shared space
  tmp=__baryonTfunc( v0 + idx*4,
  		     v1 + idx*4,
  		     v2 + idx*4,
  		     v3 + idx*4 );

  
  sdata[threadIdx.x] = hipCmul( tmp, phase[idx] ); 
  // sdata[threadIdx.x] = phase[idx]; 
  // sdata seems to have all the right values on the threads.
  }    
  __syncthreads();
  
  for(int s=1; s<blockDim.x; s*=2)
  {
    int index = 2*s*threadIdx.x;
    if(index<blockDim.x)
    {
      sdata[index]=hipCadd(sdata[index],sdata[index+s]);  
    }

    __syncthreads();
  }

  if(threadIdx.x==0)
  {
      atomicAdd(&(res[0].x), sdata[0].x);
      atomicAdd(&(res[0].y), sdata[0].y);
  }
}




__global__ void tfunc_source_spatialsum_readphase_v2(hipDoubleComplex *res, 
    hipDoubleComplex *v0, hipDoubleComplex *v1, hipDoubleComplex *v2, hipDoubleComplex *v3, 
						   hipDoubleComplex *phase, int NX)
{
  //spatial-index
  int idx=blockIdx.x*blockDim.x + threadIdx.x; 
  // Do I need memory bounds check?
  // in principle no but keeping for now
  
  __shared__ hipDoubleComplex sdata[1024];
  sdata[threadIdx.x].x=0;
  sdata[threadIdx.x].y=0;

  if(idx<NX*NX*NX)
  {

  hipDoubleComplex tmp;
  
  // // Do reduction in the shared space
  // sdata[threadIdx.x] = hipCmul(hipCmul(hipCmul(v0[idx],v1[idx]),hipCmul(v2[idx],v3[idx])),phase[idx]); 
  // Do reduction in the shared space
  tmp=__baryonTfunc( v0 + idx*4,
  		     v1 + idx*4,
  		     v2 + idx*4,
  		     v3 + idx*4 );

  
  sdata[threadIdx.x] = hipConj(hipCmul( tmp, phase[idx] )); 
  // sdata[threadIdx.x] = phase[idx]; 
  // sdata seems to have all the right values on the threads.
  }    
  __syncthreads();
  
  for(int s=1; s<blockDim.x; s*=2)
  {
    int index = 2*s*threadIdx.x;
    if(index<blockDim.x)
    {
      sdata[index]=hipCadd(sdata[index],sdata[index+s]);  
    }

    __syncthreads();
  }

  if(threadIdx.x==0)
  {
      atomicAdd(&(res[0].x), sdata[0].x);
      atomicAdd(&(res[0].y), sdata[0].y);
  }
}




__global__ void tfunc_source_spatialsum(hipDoubleComplex *res, 
    hipDoubleComplex *v0, hipDoubleComplex *v1, hipDoubleComplex *v2, hipDoubleComplex *v3,
    int px, int py, int pz, int NX)
{
  //spatial-index
  int idx=blockIdx.x*blockDim.x + threadIdx.x; 
  // Do I need memory bounds check?
  // in principle no but keeping for now

  
  __shared__ hipDoubleComplex sdata[1024];
  sdata[threadIdx.x].x=0;
  sdata[threadIdx.x].y=0;
  
  if(idx<NX*NX*NX)
  {
    
    int z=idx%NX;
    int tmp=idx/NX;
    int y=tmp%NX;
    tmp=tmp/NX;
    int x=tmp%NX;
  

    hipDoubleComplex phase;
    sincos(2.0*M_PI*(double(px*x+py*y+pz*z))/((double)NX), &phase.y, &phase.x);
    
    //printf("GPUid: block.x=%d, blockDim.x=%d, thread.x=%d, idx=%d\n", blockIdx.x, blockDim.x, threadIdx.x, idx);

    //hipDoubleComplex val = hipCmul(hipCmul(hipCmul(v0[idx],v1[idx]),hipCmul(v2[idx],v3[idx])),phase); 

    // Do reduction in the shared space
    sdata[threadIdx.x] = hipCmul(hipCmul(hipCmul(hipConj(v0[idx]),hipConj(v1[idx])),hipCmul(hipConj(v2[idx]),hipConj(v3[idx]))),phase); 
    
    
/*    printf("GPUSum site=%d: v0=%e%+ei, v1=%e%+ei, v2=%e%+ei, v3=%e%+ei, phase=%e%+ei, val=%e%+ei\n", idx, 
        v0[idx].x ,v0[idx].y, 
        v1[idx].x, v1[idx].y, 
        v2[idx].x, v2[idx].y, 
        v3[idx].x, v3[idx].y,
        phase.x, phase.y, 
        sdata[threadIdx.x].x, sdata[threadIdx.x].y);
  */  

    // sdata seems to have all the right values on the threads.
  }  
    
  __syncthreads();
  
  for(int s=1; s<blockDim.x; s*=2)
  {
    int index = 2*s*threadIdx.x;
    if(index<blockDim.x)
    {
      sdata[index]=hipCadd(sdata[index],sdata[index+s]);  
    }

    __syncthreads();
  }

  if(threadIdx.x==0)
  {
      atomicAdd(&(res[0].x), sdata[0].x);
      atomicAdd(&(res[0].y), sdata[0].y);
//    res[blockIdx.x]=sdata[0];
  }
}

/*
  // Do I need memory bounds check?
  // in principle no but keeping for now
  if(idx<NX*NX*NX)
  {

    int z=idx%NX;
    int tmp=idx/NX;
    int y=tmp%NX;
    tmp=tmp/NX;
    int x=tmp%NX;
  

    hipDoubleComplex phase;
    sincos(-2.0*M_PI*(double(px*x+py*y+pz*z))/((double)NX), &phase.y, &phase.x);
    
//    printf("GPUSum: v0=%f, v1=%f, v2=%f, v3=%f, phase=%f", v0[idx], v1[idx], v2[idx], v3[idx], phase);

    hipDoubleComplex val = hipCmul(hipCmul(hipCmul(v0[idx],v1[idx]),hipCmul(v2[idx],v3[idx])),phase); 


  
    atomicAdd(&(res[0].x), val.x);
    atomicAdd(&(res[0].y), val.y);
  }
*/
