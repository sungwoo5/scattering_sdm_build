#ifndef GET_PERAMS_H
#define GET_PERAMS_H

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include <complex>
#include <vector> 
#include <set>
#include <map>
#include <unordered_set>
#include <string>
#include "Eigen/Dense"
#include <Eigen/StdVector>
#include "util/ferm/key_prop_matelem.h"
using namespace Eigen;
using namespace std; 
using std::vector;
using std::ifstream;
using namespace QDP; 
using namespace FILEDB;
using namespace Chroma;


// get the perambulator matrix (with distillation indices) from timeslices and dirac indices
// with the convention here, t_sink varies slowest, can be changed if causes a slow-down
int tau_ind(int t_sink, int t_source, int alpha_sink, int alpha_source){ 
   return t_sink  * Nt * Nd * Nd
        + t_source     * Nd * Nd 
        + alpha_sink        * Nd  
        + alpha_source;
}


void get_tau(string filename, std::vector<MatrixXcd> &tau){
    // for a given t source/sink, alpha sources/sink tau is a Nvec x Nvec matrix of complex doubles
    int num_matrices = Nt * Nt* Nd * Nd;
    
    BinaryStoreDB<SerialDBKey<KeyPropElementalOperator_t>, SerialDBData<ValPropElementalOperator_t>> qdp_db;
    qdp_db.open(filename, O_RDONLY, 0664); // 0600,0644,etc  = identifier for permissions on file 
    std::vector< SerialDBKey<KeyPropElementalOperator_t> > keys;
    qdp_db.keys(keys);
    for(int i=0; i<keys.size(); i++){ // start loop over t and Dirac indices -- only as many timeslices as there are
      SerialDBData<ValPropElementalOperator_t> tmp;
      auto id = qdp_db.get(keys[i], tmp); // store the value from key into tmp 
      
      int t_source       = keys[i].key().t_source;
      int t_sink         = keys[i].key().t_slice;
      int alpha_source   = keys[i].key().spin_src;
      int alpha_sink     = keys[i].key().spin_snk;
    
      //std::cout << "Reading tau from " << filename << " filling mat for t_source=" << t_source << " t_sink=" << t_sink << std::endl;
      // whatever order the tsource and tsink are calculated in the perambulator file, 
      // we can save them to our perambulator matrix as outlined in the tau_ind function
      int index = tau_ind(t_sink, t_source, alpha_sink, alpha_source);
      for(int j=0; j<Nvec; j++){
        for(int k=0; k<Nvec; k++){
          double value_real = tmp.data().mat(j,k).elem().elem().elem().real(); // scalar indices can be popped
          double value_imag = tmp.data().mat(j,k).elem().elem().elem().imag(); // out of the lattice with elems
          tau[index](j,k) = complex<double>(value_real, value_imag);
        } // end k 
      } // end j
    } // end loop over keys (t and Dirac indices)
}




#endif
