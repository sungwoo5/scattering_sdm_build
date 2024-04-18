#include "utils.h"

#include <complex>
#include <cfloat>
#include <iostream>

using namespace std;
using cd = std::complex<double>;

#define ULP_N 4

bool nearlyequal2(std::complex<double> a, std::complex<double> b, double TOL)
{
 double diff = abs(a-b);
 double mag = (abs(a)+abs(b))/2.;
 
 return diff <= mag*TOL;
}

bool nearlyequal(std::complex<double> a, std::complex<double> b)
{
  double diff = abs(a-b);
  double mag = (abs(a)-abs(b))/2.;
  return diff <= (mag*FLT_EPSILON*(1ull << ULP_N));  
}

bool compare_Tfuncs(Eigen::Tensor<std::complex<double>,4> &A, Eigen::Tensor<std::complex<double>,4> &B, int N)
{
  bool result=true;
  for(int l0=0; l0<N; ++l0)
  for(int l1=0; l1<N; ++l1)
  for(int l2=0; l2<N; ++l2)
  for(int l3=0; l3<N; ++l3)
  {
    if( l0==l1 || l0==l2 || l0==l3 || l1==l2 || l1==l3 || l2==l3 )
    {
    
    }
    else
    {
      if(!nearlyequal2(A(l0,l1,l2,l3),B(l0,l1,l2,l3)))
      {
        std::cout << "Tensor difference at idx " << l0 << "," << l1 << "," << l2 << "," << l3 << " - values " << A(l0,l1,l2,l3) << "   " << B(l0,l1,l2,l3) << "  dif=" << A(l0,l1,l2,l3)-B(l0,l1,l2,l3) << std::endl;
        
        result=false;
      }
    }
  }
  return result;
}



int epsilon(const vector<int> &indices)
{
  ///make sure the epsilon tensor isn't zero.
  ///by finding the first instance of two indices being equal
  vector<int> idx=indices;
  if( idx[0]==idx[1] || idx[0]==idx[2] || idx[0]==idx[3] || idx[1]==idx[2] || idx[1]==idx[3] || idx[2]==idx[3] )
    return 0;

  return (idx[0]-idx[1])*(idx[0]-idx[2])*(idx[0]-idx[3])*(idx[1]-idx[2])*(idx[1]-idx[3])*(idx[2]-idx[3])/12;
}


