#ifndef EIGEN_BARYON_CONSTRUCTOR_H
#define EIGEN_BARYON_CONSTRUCTOR_H

#include "get_evecs.h"
#include "global_variables.h"
#include "timer.h"
#include "diagram_utils.h"
#include "utils.h"

#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Dense>

#include <vector>
#include <complex>
#include <omp.h>
#include <stdio.h>

using cd = std::complex<double>;

std::vector<int> position(int site)
{
  int z = site%Lz;
  site = site/Lz;
  int y = site%Ly;
  site = site/Ly;
  int x = site%Lx;

  return std::vector<int>{x,y,z};
}


void test_evec_storage(std::vector<Eigen::MatrixXcd> &evecs, int t)
{
  std::cout << "Testing evec storage" << std::endl;

  std::complex<double> *v0_slow = new std::complex<double>[Lx*Lx*Lx];
  std::complex<double> *v1_slow = new std::complex<double>[Lx*Lx*Lx];
  std::complex<double> *v2_slow = new std::complex<double>[Lx*Lx*Lx];
  std::complex<double> *v3_slow = new std::complex<double>[Lx*Lx*Lx];
  std::complex<double> *v0_fast = new std::complex<double>[Lx*Lx*Lx];
  std::complex<double> *v1_fast = new std::complex<double>[Lx*Lx*Lx];
  std::complex<double> *v2_fast = new std::complex<double>[Lx*Lx*Lx];
  std::complex<double> *v3_fast = new std::complex<double>[Lx*Lx*Lx];

  // dont want to loop pover sites and get all data onto GPU in one go
  std::complex<double> *all_evecs = new std::complex<double>[Nvec*Nc*Lx*Lx*Lx];
  for(int l=0; l<Nvec; ++l)
  {
    VectorXcd evec = evecs[t].row(l);
    memcpy(all_evecs+l*Nc*Lx*Lx*Lx, evec.data(), Nc*Lx*Lx*Lx*sizeof(std::complex<double>));
  }
  
  std::cout << "evecs[t].row(0)[0*Lx*Lx*Lx+0]=" << evecs[t].row(0)[0*Lx*Lx*Lx+0] << " " 
            << "all_evecs[0*Nc*Lx*Lx*Lx+1*Lx*Lx*Lx+0]=" << all_evecs[0*Nc*Lx*Lx*Lx+0*Lx*Lx*Lx+0] << std::endl;

  std::cout << "evecs[t].row(2)[1*Lx*Lx*Lx+1]=" << evecs[t].row(2)[1*Lx*Lx*Lx+1] << " " 
            << "all_evecs[2*Nc*Lx*Lx*Lx+1*Lx*Lx*Lx+1]=" << all_evecs[2*Nc*Lx*Lx*Lx+1*Lx*Lx*Lx+1] << std::endl;

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
        // lets fill the v's the original slow way.
        auto eVec0 = evecs[t].row(l0);
        auto eVec1 = evecs[t].row(l1);
        auto eVec2 = evecs[t].row(l2);
        auto eVec3 = evecs[t].row(l3);
        
        for(int site=0; site<Lx*Lx*Lx; ++site)
        {
          v0_slow[site]=eVec0[c0*Lx*Lx*Lx+site]; 
          v1_slow[site]=eVec1[c1*Lx*Lx*Lx+site]; 
          v2_slow[site]=eVec2[c2*Lx*Lx*Lx+site]; 
          v3_slow[site]=eVec3[c3*Lx*Lx*Lx+site]; 
        }   

        memcpy(v0_fast, all_evecs+l0*Nc*Lx*Lx*Lx+c0*Lx*Lx*Lx,Lx*Lx*Lx*sizeof(std::complex<double>));
        memcpy(v1_fast, all_evecs+l1*Nc*Lx*Lx*Lx+c1*Lx*Lx*Lx,Lx*Lx*Lx*sizeof(std::complex<double>));
        memcpy(v2_fast, all_evecs+l2*Nc*Lx*Lx*Lx+c2*Lx*Lx*Lx,Lx*Lx*Lx*sizeof(std::complex<double>));
        memcpy(v3_fast, all_evecs+l3*Nc*Lx*Lx*Lx+c3*Lx*Lx*Lx,Lx*Lx*Lx*sizeof(std::complex<double>));

        std::cout << "comparing vecs - l=(" 
                  << l0 << "," << l1 << "," << l2 << "," << l3 << ") c=("
                  << c0 << "," << c1 << "," << c2 << "," << c3 << ")" << std::endl;
        for(int i=0; i<Lx*Lx*Lx; ++i)
        {
          if(!nearlyequal2(v0_slow[i], v0_fast[i]))
          {
            std::cout << "v0 different - " << v0_slow[i] << "   " << v0_fast[i];
          }
          if(!nearlyequal2(v1_slow[i], v1_fast[i]))
          {
            std::cout << "v1 different - " << v1_slow[i] << "   " << v1_fast[i];
          }
          if(!nearlyequal2(v2_slow[i], v2_fast[i]))
          {
            std::cout << "v2 different - " << v2_slow[i] << "   " << v2_fast[i];
          }
          if(!nearlyequal2(v3_slow[i], v3_fast[i]))
          {
            std::cout << "v3 different - " << v3_slow[i] << "   " << v3_fast[i];
          }

       //   std::cout << v0_slow[i] << " " << v0_fast[i] << " | "
       //             << v1_slow[i] << " " << v1_fast[i] << " | "
       //             << v2_slow[i] << " " << v2_fast[i] << " | "
       //             << v3_slow[i] << " " << v3_fast[i] << std::endl;
        }
      }
      }
    }
  } 
}



Eigen::Tensor<std::complex<double>,4> create_Tsink_serial(std::vector<Eigen::MatrixXcd> &evectors, int t, int px, int py, int pz)
{
  Eigen::Tensor<std::complex<double>,4> baryonTFunc(Nvec,Nvec,Nvec,Nvec);
  baryonTFunc.setConstant(std::complex<double>(0.,0.));
  for(int c0=0; c0<Nc; ++c0)
  for(int c1=0; c1<Nc; ++c1)
  for(int c2=0; c2<Nc; ++c2)
  for(int c3=0; c3<Nc; ++c3)
  {
    auto eps = epsilon(vector<int>{c0,c1,c2,c3});
    if(eps!=0)
    {
      auto e = std::complex<double>((double)eps,0.);
      for(int l0=0; l0<Nvec; ++l0)
      for(int l1=0; l1<Nvec; ++l1)
      for(int l2=0; l2<Nvec; ++l2)
      for(int l3=0; l3<Nvec; ++l3)
      {
        auto eVec0 = evectors[t].row(l0);
        auto eVec1 = evectors[t].row(l1);
        auto eVec2 = evectors[t].row(l2);
        auto eVec3 = evectors[t].row(l3);
        /*
        if(l0==0 && l1==1 && l2==2 && l3==5)
        { 
        std::cout << "starting cpu-site-sum|"
                  << c0 << "," 
                  << c1 << "," 
                  << c2 << "," 
                  << c3 << "," 
                  << l0 << "," 
                  << l1 << "," 
                  << l2 << "," 
                  << l3 << "," 
                  << "|" << std::endl;
        }
       */         

        //std::cout << "evec0-serial = ";
        //for(int site=0; site<Lx*Lx*Lx; ++site)
        //{
        //  std::cout << eVec0(v_ind(c0,site)) << " ";
        //}
        //std::cout << std::endl;
        cd tmp(0,0);
        for(int site=0; site<Lx*Lx*Lx; ++site)
        {
            auto p = position(site);

            auto phase = exp(cd(0.,-1.)*2.0*M_PI*(p[0]*px/((double)Lx)+p[1]*py/((double)Ly)+p[2]*pz/((double)Lz) )); // momentum zero

            //printf("CPUSum: site=%d, v0=%f, v1=%f, v2=%f, v3=%f, phase=%f, val=%f\n", site, eVec0(v_ind(c0,site)), eVec1(v_ind(c1,site)),eVec2(v_ind(c2,site)),eVec3(v_ind(c3,site)),phase,
            /*
        if(l0==0 && l1==1 && l2==2 && l3==5)
        { 
            std::cout << "CPUSum site=" << site  
                      << ", v0=" << eVec0(v_ind(c0,site))
                      << ", v1=" << eVec1(v_ind(c1,site))
                      << ", v2=" << eVec2(v_ind(c2,site))
                      << ", v3=" << eVec3(v_ind(c3,site))
                      << ", phase=" << phase 
                      << ", val=" << phase*eVec0(v_ind(c0,site))*eVec1(v_ind(c1,site))*eVec2(v_ind(c2,site))*eVec3(v_ind(c3,site)) 
                      << std::endl;
        }            
            */
            tmp += phase*eVec0(v_ind(c0,site))*eVec1(v_ind(c1,site))*
                                       eVec2(v_ind(c2,site))*eVec3(v_ind(c3,site)); 
     

            baryonTFunc(l0,l1,l2,l3) += phase*e*
                                       eVec0(v_ind(c0,site))*eVec1(v_ind(c1,site))*
                                       eVec2(v_ind(c2,site))*eVec3(v_ind(c3,site)); 
        } //end sites loop
       /* 
        if(l0==0 && l1==1 && l2==2 && l3==5)
        { 
        std::cout << "cpu-site-sum|"
                  << c0 << "," 
                  << c1 << "," 
                  << c2 << "," 
                  << c3 << "," 
                  << l0 << "," 
                  << l1 << "," 
                  << l2 << "," 
                  << l3 << "," 
                  << "|" << tmp << std::endl;
        }  
       */
      } // end evec loop
    } // end eps if
  } // end color loop

  return baryonTFunc;
}


Eigen::Tensor<std::complex<double>,4> create_Tsink_parallel(std::vector<Eigen::MatrixXcd> &evectors, int t, int px, int py, int pz)
{
  Eigen::Tensor<std::complex<double>,4> baryonTFunc(Nvec,Nvec,Nvec,Nvec);
  baryonTFunc.setConstant(std::complex<double>(0.,0.));
  for(int c0=0; c0<Nc; ++c0)
  for(int c1=0; c1<Nc; ++c1)
  for(int c2=0; c2<Nc; ++c2)
  for(int c3=0; c3<Nc; ++c3)
  {
    auto eps = epsilon(vector<int>{c0,c1,c2,c3});
    if(eps!=0)
    {
      auto e = std::complex<double>((double)eps,0.);
      for(int l0=0; l0<Nvec; ++l0)
      for(int l1=0; l1<Nvec; ++l1)
      for(int l2=0; l2<Nvec; ++l2)
      {
        auto eVec0 = evectors[t].row(l0);
        auto eVec1 = evectors[t].row(l1);
        auto eVec2 = evectors[t].row(l2);
#pragma omp parallel for shared(baryonTFunc, px,py,pz, l0,l1,l2), schedule(static)
      for(int l3=0; l3<Nvec; ++l3)
      {
        auto eVec3 = evectors[t].row(l3);
       // QDP::sum 
//#pragma omp parallel for reduction(+:sum), shared(baryonTFunc,px,py,pz,l0,l1,l2,l3), schedule(static) 
//
        //Timer<> ompTimer("spatial sum in omp thread");
        for(int site=0; site<Lx*Lx*Lx; ++site)
        {
            auto p = position(site);

            auto phase = exp(cd(0.,-1.)*2.0*M_PI*(p[0]*px/((double)Lx)+p[1]*py/((double)Ly)+p[2]*pz/((double)Lz) )); // momentum zero
            baryonTFunc(l0,l1,l2,l3) = baryonTFunc(l0,l1,l2,l3) + phase*e*
                                       eVec0(v_ind(c0,site))*eVec1(v_ind(c1,site))*
                                       eVec2(v_ind(c2,site))*eVec3(v_ind(c3,site)); 
        } //end sites loop
        //ompTimer.stop<std::chrono::microseconds>("us");
      } // end evec loop
      }
    } // end eps if
  } // end color loop

  return baryonTFunc;
}


Eigen::Tensor<std::complex<double>,4> create_Tsource_serial(std::vector<Eigen::MatrixXcd> &evectors, int t, int px, int py, int pz)
{
  Eigen::Tensor<std::complex<double>,4> baryonTFunc(Nvec,Nvec,Nvec,Nvec);
  baryonTFunc.setConstant(std::complex<double>(0.,0.));
  for(int c0=0; c0<Nc; ++c0)
  for(int c1=0; c1<Nc; ++c1)
  for(int c2=0; c2<Nc; ++c2)
  for(int c3=0; c3<Nc; ++c3)
  {
    auto eps = epsilon(vector<int>{c0,c1,c2,c3});
    if(eps!=0)
    {
      auto e = std::complex<double>((double)eps,0.);
      for(int l0=0; l0<Nvec; ++l0)
      for(int l1=0; l1<Nvec; ++l1)
      for(int l2=0; l2<Nvec; ++l2)
      for(int l3=0; l3<Nvec; ++l3)
      {
        auto eVec0 = evectors[t].row(l0);
        auto eVec1 = evectors[t].row(l1);
        auto eVec2 = evectors[t].row(l2);
        auto eVec3 = evectors[t].row(l3);

        for(int site=0; site<Lx*Lx*Lx; ++site)
        {
            auto p = position(site);

            auto phase = exp(cd(0.,1.)*2.0*M_PI*(p[0]*px/((double)Lx)+p[1]*py/((double)Ly)+p[2]*pz/((double)Lz)) ); // momentum zero
            baryonTFunc(l0,l1,l2,l3) += phase*e*
                                       conj(eVec0(v_ind(c0,site)))*conj(eVec1(v_ind(c1,site)))*
                                       conj(eVec2(v_ind(c2,site)))*conj(eVec3(v_ind(c3,site)));
        } //end sites loop
      } // end evec loop
    } // end eps loop
  }

  return baryonTFunc;
}


Eigen::Tensor<std::complex<double>,4> create_Tsource_parallel(std::vector<Eigen::MatrixXcd> &evectors, int t, int px, int py, int pz)
{
  Eigen::Tensor<std::complex<double>,4> baryonTFunc(Nvec,Nvec,Nvec,Nvec);
  baryonTFunc.setConstant(std::complex<double>(0.,0.));
  for(int c0=0; c0<Nc; ++c0)
  for(int c1=0; c1<Nc; ++c1)
  for(int c2=0; c2<Nc; ++c2)
  for(int c3=0; c3<Nc; ++c3)
  {
    auto eps = epsilon(vector<int>{c0,c1,c2,c3});
    if(eps!=0)
    {
      auto e = std::complex<double>((double)eps,0.);
      for(int l0=0; l0<Nvec; ++l0)
      for(int l1=0; l1<Nvec; ++l1)
      for(int l2=0; l2<Nvec; ++l2)
      {  
        auto eVec0 = evectors[t].row(l0);
        auto eVec1 = evectors[t].row(l1);
        auto eVec2 = evectors[t].row(l2);
      
#pragma omp parallel for shared(baryonTFunc, px,py,pz, l0,l1,l2), schedule(static)
      for(int l3=0; l3<Nvec; ++l3)
      {
        auto eVec3 = evectors[t].row(l3);
      
//  QDP::sum is not a vaild identifier.

//#pragma omp parallel for reduction(+:sum), shared(baryonTFunc, px,py,pz, l0,l1,l2), schedule(static)
        for(int site=0; site<Lx*Lx*Lx; ++site)
        {
            auto p = position(site);

            auto phase = exp(cd(0.,1.)*2.0*M_PI*(p[0]*px/((double)Lx)+p[1]*py/((double)Ly)+p[2]*pz/((double)Lz)) ); // momentum zero
            baryonTFunc(l0,l1,l2,l3) += phase*e*
                                       conj(eVec0(v_ind(c0,site)))*conj(eVec1(v_ind(c1,site)))*
                                       conj(eVec2(v_ind(c2,site)))*conj(eVec3(v_ind(c3,site)));
        } //end sites loop
      } // end evec loop
      }
    } // end eps loop
  }

  return baryonTFunc;
}

Eigen::Tensor<std::complex<double>,4> create_baryonSink(
    const Eigen::Tensor<std::complex<double>,4> &tFunc, 
    const Eigen::Tensor<std::complex<double>,4> &GAMMA)
{
    Eigen::Tensor<std::complex<double>,4> res(Ns*Nvec,Ns*Nvec,Ns*Nvec,Ns*Nvec);
    res.setConstant(std::complex<double>(0.,0.));
    
    for(int l0=0; l0<Nvec; ++l0)
    for(int l1=0; l1<Nvec; ++l1)
    for(int l2=0; l2<Nvec; ++l2)
    for(int l3=0; l3<Nvec; ++l3)
    for(int s0=0; s0<Ns; ++s0)
    for(int s1=0; s1<Ns; ++s1)
    for(int s2=0; s2<Ns; ++s2)
    for(int s3=0; s3<Ns; ++s3)
    {
      res(s0*Nvec+l0,s1*Nvec+l1,s2*Nvec+l2,s3*Nvec+l3)=GAMMA(s0,s1,s2,s3)*tFunc(l0,l1,l2,l3);
    }
    
    return res;
}

Eigen::Tensor<std::complex<double>,4> create_baryonSource(
    const Eigen::Tensor<std::complex<double>,4> &tFunc, 
    const Eigen::Tensor<std::complex<double>,4> &GAMMA)
{
    Eigen::Tensor<std::complex<double>,4> res(Ns*Nvec,Ns*Nvec,Ns*Nvec,Ns*Nvec);
    res.setConstant(std::complex<double>(0.,0.));
    
    for(int l0=0; l0<Nvec; ++l0)
    for(int l1=0; l1<Nvec; ++l1)
    for(int l2=0; l2<Nvec; ++l2)
    for(int l3=0; l3<Nvec; ++l3)
    for(int s0=0; s0<Ns; ++s0)
    for(int s1=0; s1<Ns; ++s1)
    for(int s2=0; s2<Ns; ++s2)
    for(int s3=0; s3<Ns; ++s3)
    {
      res(s0*Nvec+l0,s1*Nvec+l1,s2*Nvec+l2,s3*Nvec+l3)=conj(GAMMA(s0,s1,s2,s3))*tFunc(l0,l1,l2,l3);
    }
    
    return res;
}












void bblocks_checks(
    const std::vector<Eigen::Tensor<std::complex<double>,4>> &B, 
    const std::vector<Eigen::Tensor<std::complex<double>,4>> &Bstar, 
    const std::vector<MatrixXcd> &tau, 
    int t,
    int tf
    )
{
  std::complex<double> d0(0,0),d1(0,0),d2(0,0),d3(0,0);

  for(int s0=0; s0<Ns; ++s0) //alpha
  for(int s1=0; s1<Ns; ++s1) //sigma
  for(int s2=0; s2<Ns; ++s2) //beta
  for(int s3=0; s3<Ns; ++s3) //delta
  {
    for(int s4=0; s4<Ns; ++s4) //delta'
    for(int s5=0; s5<Ns; ++s5) //beta'
    for(int s6=0; s6<Ns; ++s6) //sigma'
    for(int s7=0; s7<Ns; ++s7) //alpha'
    {
      for(int l0=0; l0<Nvec; ++l0) 
      for(int l1=0; l1<Nvec; ++l1) 
      for(int l2=0; l2<Nvec; ++l2) 
      for(int l3=0; l3<Nvec; ++l3) 
      for(int l4=0; l4<Nvec; ++l4) 
      for(int l5=0; l5<Nvec; ++l5) 
      for(int l6=0; l6<Nvec; ++l6) 
      for(int l7=0; l7<Nvec; ++l7) 
      {
        std::complex<double> Bval=Bstar[tf](s0*Nvec+l0,s1*Nvec+l1,s2*Nvec+l2,s3*Nvec+l3)
                                  *B[t](s7*Nvec+l7,s6*Nvec+l6,s5*Nvec+l5,s4*Nvec+l4);
        d0 += Bval
             *tau[tau_ind(tf,t,s3,s7)](l3,l7)
             *tau[tau_ind(tf,t,s0,s6)](l0,l6)
             *tau[tau_ind(tf,t,s1,s5)](l1,l5)
             *tau[tau_ind(tf,t,s2,s4)](l2,l4);
        d1 += Bval
             *tau[tau_ind(tf,t,s1,s7)](l1,l7)
             *tau[tau_ind(tf,t,s0,s6)](l0,l6)
             *tau[tau_ind(tf,t,s3,s5)](l3,l5)
             *tau[tau_ind(tf,t,s2,s4)](l2,l4);
        d2 += Bval
             *tau[tau_ind(tf,t,s3,s7)](l3,l7)
             *tau[tau_ind(tf,t,s2,s6)](l2,l6)
             *tau[tau_ind(tf,t,s1,s5)](l1,l5)
             *tau[tau_ind(tf,t,s0,s4)](l0,l4);
        d3 += Bval
             *tau[tau_ind(tf,t,s1,s7)](l1,l7)
             *tau[tau_ind(tf,t,s2,s6)](l2,l6)
             *tau[tau_ind(tf,t,s3,s5)](l3,l5)
             *tau[tau_ind(tf,t,s0,s4)](l0,l4);

      }
    }
  }
  std::cout << "bblock_checks" << std::endl;
  std::cout << "d0=" << d0 << "  ";
  std::cout << "d1=" << d1 << "  ";
  std::cout << "d2=" << d2 << "  ";
  std::cout << "d3=" << d3 << "  \n";
  std::cout << "c=" << -d0 + d1 + d2 - d3 << std::endl;
}

void label_swap_checks(
    const std::vector<Eigen::Tensor<std::complex<double>,4>> &T, 
    const std::vector<Eigen::Tensor<std::complex<double>,4>> &Tstar, 
    const std::vector<MatrixXcd> &tau, 
    const Eigen::Tensor<std::complex<double>,4> &GAMMA,   
    int t,
    int tf
    )
{
  std::complex<double> d0(0,0),d1(0,0),d2(0,0),d3(0,0);

  for(int a=0; a<Ns; ++a) //alpha
  for(int s=0; s<Ns; ++s) //sigma
  for(int b=0; b<Ns; ++b) //beta
  for(int d=0; d<Ns; ++d) //delta
  {
  if( GAMMA(a,s,b,d).real()!=0 || GAMMA(a,s,b,d).imag()!=0 )
  { 
    for(int dp=0; dp<Ns; ++dp) //delta'
    for(int bp=0; bp<Ns; ++bp) //beta'
    for(int sp=0; sp<Ns; ++sp) //sigma'
    for(int ap=0; ap<Ns; ++ap) //alpha'
    {
    if( GAMMA(dp,bp,sp,ap).real()!=0 || GAMMA(dp,bp,sp,ap).imag()!=0 )
    {
      std::complex<double> Gval = GAMMA(a,s,b,d)*GAMMA(dp,bp,sp,ap);
      for(int l0=0; l0<Nvec; ++l0) 
      for(int l1=0; l1<Nvec; ++l1) 
      for(int l2=0; l2<Nvec; ++l2) 
      for(int l3=0; l3<Nvec; ++l3) 
      for(int l4=0; l4<Nvec; ++l4) 
      for(int l5=0; l5<Nvec; ++l5) 
      for(int l6=0; l6<Nvec; ++l6) 
      for(int l7=0; l7<Nvec; ++l7) 
      {
        std::complex<double> Tval=Tstar[tf](l2,l6,l4,l0)*T[t](l7,l3,l5,l1);
        d0 += Tval*Gval
             *tau[tau_ind(tf,t,d,dp)](l0,l1)
             *tau[tau_ind(tf,t,a,bp)](l2,l3)
             *tau[tau_ind(tf,t,s,sp)](l4,l5)
             *tau[tau_ind(tf,t,b,ap)](l6,l7);
        d1 += Tval*Gval
             *tau[tau_ind(tf,t,s,dp)](l4,l1)
             *tau[tau_ind(tf,t,a,bp)](l2,l3)
             *tau[tau_ind(tf,t,d,sp)](l0,l5)
             *tau[tau_ind(tf,t,b,ap)](l6,l7);
        d2 += Tval*Gval
             *tau[tau_ind(tf,t,d,dp)](l0,l1)
             *tau[tau_ind(tf,t,b,bp)](l6,l3)
             *tau[tau_ind(tf,t,s,sp)](l4,l5)
             *tau[tau_ind(tf,t,a,ap)](l2,l7);
        d3 += Tval*Gval
             *tau[tau_ind(tf,t,s,dp)](l4,l1)
             *tau[tau_ind(tf,t,b,bp)](l6,l3)
             *tau[tau_ind(tf,t,d,sp)](l0,l5)
             *tau[tau_ind(tf,t,a,ap)](l2,l7);

      }
    }
    }
  }
  }
  std::cout << "label_swap_checks" << std::endl;
  std::cout << "d0=" << d0 << "  ";
  std::cout << "d1=" << d1 << "  ";
  std::cout << "d2=" << d2 << "  ";
  std::cout << "d3=" << d3 << "  \n";
  std::cout << "c=" << -d0 + d1 + d2 - d3 << std::endl;
}

void kimmy_checks(
    const std::vector<Eigen::Tensor<std::complex<double>,4>> &T, 
    const std::vector<Eigen::Tensor<std::complex<double>,4>> &Tstar, 
    const std::vector<MatrixXcd> &tau, 
    const Eigen::Tensor<std::complex<double>,4> &GAMMA,   
    int t,
    int tf
    )
{
  std::complex<double> d0(0,0),d1(0,0),d2(0,0),d3(0,0);

  for(int a=0; a<Ns; ++a) //alpha
  for(int s=0; s<Ns; ++s) //sigma
  for(int b=0; b<Ns; ++b) //beta
  for(int d=0; d<Ns; ++d) //delta
  {
  if( GAMMA(a,s,b,d).real()!=0 || GAMMA(a,s,b,d).imag()!=0 )
  { 
    for(int dp=0; dp<Ns; ++dp) //delta'
    for(int bp=0; bp<Ns; ++bp) //beta'
    for(int sp=0; sp<Ns; ++sp) //sigma'
    for(int ap=0; ap<Ns; ++ap) //alpha'
    {
    if( GAMMA(dp,bp,sp,ap).real()!=0 || GAMMA(dp,bp,sp,ap).imag()!=0 )
    {
      std::complex<double> Gval = GAMMA(a,s,b,d)*GAMMA(dp,bp,sp,ap);
      for(int l0=0; l0<Nvec; ++l0) 
      for(int l1=0; l1<Nvec; ++l1) 
      for(int l2=0; l2<Nvec; ++l2) 
      for(int l3=0; l3<Nvec; ++l3) 
      for(int l4=0; l4<Nvec; ++l4) 
      for(int l5=0; l5<Nvec; ++l5) 
      for(int l6=0; l6<Nvec; ++l6) 
      for(int l7=0; l7<Nvec; ++l7) 
      {
        d0 += Tstar[tf](l2,l6,l4,l0)*T[t](l7,l3,l5,l1)*Gval
             *tau[tau_ind(tf,t,d,dp)](l0,l1)
             *tau[tau_ind(tf,t,a,bp)](l2,l3)
             *tau[tau_ind(tf,t,s,sp)](l4,l5)
             *tau[tau_ind(tf,t,b,ap)](l6,l7);
        d1 += Tstar[tf](l2,l6,l0,l4)*T[t](l7,l3,l5,l1)*Gval
             *tau[tau_ind(tf,t,s,dp)](l0,l1)
             *tau[tau_ind(tf,t,a,bp)](l2,l3)
             *tau[tau_ind(tf,t,d,sp)](l4,l5)
             *tau[tau_ind(tf,t,b,ap)](l6,l7);
        d2 += Tstar[tf](l6,l2,l4,l0)*T[t](l7,l3,l5,l1)*Gval
             *tau[tau_ind(tf,t,d,dp)](l0,l1)
             *tau[tau_ind(tf,t,b,bp)](l2,l3)
             *tau[tau_ind(tf,t,s,sp)](l4,l5)
             *tau[tau_ind(tf,t,a,ap)](l6,l7);
        d3 += Tstar[tf](l6,l2,l0,l4)*T[t](l7,l3,l5,l1)*Gval
             *tau[tau_ind(tf,t,s,dp)](l0,l1)
             *tau[tau_ind(tf,t,b,bp)](l2,l3)
             *tau[tau_ind(tf,t,d,sp)](l4,l5)
             *tau[tau_ind(tf,t,a,ap)](l6,l7);

      }
    }
    }
  }
  }
  std::cout << "kimmy_checks" << std::endl;

  std::cout << "d0=" << d0 << "  ";
  std::cout << "d1=" << d1 << "  ";
  std::cout << "d2=" << d2 << "  ";
  std::cout << "d3=" << d3 << "  \n";
  std::cout << "c=" << -d0 + d1 + d2 - d3 << std::endl;
}






Eigen::Tensor<std::complex<double>,4> create_baryonT(std::vector<Eigen::MatrixXcd> &evectors, int t, int px, int py, int pz)
{

  Eigen::Tensor<std::complex<double>,4> baryonTFunc(Nvec,Nvec,Nvec,Nvec);
  baryonTFunc.setConstant(std::complex<double>(0.,0.));
  for(int a=0; a<Nc; ++a)
  for(int b=0; b<Nc; ++b)
  for(int c=0; c<Nc; ++c)
  for(int d=0; d<Nc; ++d)
  {
    auto eps = epsilon(vector<int>{a,b,c,d});
    if(eps!=0)
    {
      auto e = std::complex<double>((double)eps,0.);
      for(int l0=0; l0<Nvec; ++l0)
      for(int l1=0; l1<Nvec; ++l1)
      for(int l2=0; l2<Nvec; ++l2)
      for(int l3=0; l3<Nvec; ++l3)
      {
        auto eVec0 = evectors[t].row(l0);
        auto eVec1 = evectors[t].row(l1);
        auto eVec2 = evectors[t].row(l2);
        auto eVec3 = evectors[t].row(l3);
        
#pragma omp parallel for shared(baryonTFunc, eVec0, eVec1, eVec2, eVec3, a,b,c,d,e, l0,l1,l2,l3), schedule(static) 
        for(int site=0; site<Lx*Lx*Lx; ++site)
        {
            auto p = position(site);

            auto phase = exp(cd(0.,1.)*2.0*M_PI*(p[0]*px/((double)Lx)+p[1]*py/((double)Ly)+p[2]*pz/((double)Lz) )); // momentum zero
            
            baryonTFunc(l0,l1,l2,l3) += phase*e*
                                       eVec0(v_ind(a,site))*eVec1(v_ind(b,site))*
                                       eVec2(v_ind(c,site))*eVec3(v_ind(d,site)); 
        } //end sites loop
      } // end evec loop
    } // end eps if
  } // end color loop

  return baryonTFunc;
}

Eigen::Tensor<std::complex<double>,4> create_baryonTDag(std::vector<Eigen::MatrixXcd> &evectors, int t, int px, int py, int pz)
{
  Eigen::Tensor<std::complex<double>,4> baryonTFunc(Nvec,Nvec,Nvec,Nvec);
  baryonTFunc.setConstant(std::complex<double>(0.,0.));
  for(int a=0; a<Nc; ++a)
  for(int b=0; b<Nc; ++b)
  for(int c=0; c<Nc; ++c)
  for(int d=0; d<Nc; ++d)
  {
    auto eps = epsilon(vector<int>{a,b,c,d});
    if(eps!=0)
    {
      auto e = std::complex<double>((double)eps,0.);
      for(int l0=0; l0<Nvec; ++l0)
      for(int l1=0; l1<Nvec; ++l1)
      for(int l2=0; l2<Nvec; ++l2)
      for(int l3=0; l3<Nvec; ++l3)
      {
        auto eVec0 = evectors[t].row(l0);
        auto eVec1 = evectors[t].row(l1);
        auto eVec2 = evectors[t].row(l2);
        auto eVec3 = evectors[t].row(l3);
        
//Timer<> baryonT_omp_timer("site loop"); 
#pragma omp parallel for shared(baryonTFunc, eVec0, eVec1, eVec2, eVec3, a,b,c,d,e, l0,l1,l2,l3), schedule(static)
   // array with size num of threads
   // each elem is partial sum of each thread
   // then reduce array into baryonTFunc
        for(int site=0; site<Lx*Lx*Lx; ++site)
        {
            auto p = position(site);

            auto phase = exp(cd(0.,-1.)*2.0*M_PI*(p[0]*px/((double)Lx)+p[1]*py/((double)Ly)+p[2]*pz/((double)Lz)) ); // momentum zero
            baryonTFunc(l0,l1,l2,l3) += phase*e*
                                       conj(eVec0(v_ind(a,site)))*conj(eVec1(v_ind(b,site)))*
                                       conj(eVec2(v_ind(c,site)))*conj(eVec3(v_ind(d,site)));
        } //end sites loop


//      baryonT_omp_timer.stop<std::chrono::microseconds>("us");
      } // end evec loop
    } // end eps loop
  }

  return baryonTFunc;
}









#endif

