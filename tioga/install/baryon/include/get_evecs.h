#ifndef GET_EVECS_H
#define GET_EVECS_H

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
using namespace Eigen;
using namespace std; 
using std::vector;
using std::ifstream;
#include "global_variables.h"
using namespace QDP;
using namespace FILEDB;
// using namespace Chroma;


//---------------------------------------------------------------------
//! Some struct to use
struct EvecDBKey_t
{
  int t_slice;
  int colorvec;
};

//! Reader
void read(BinaryReader& bin, EvecDBKey_t& param)
{
  read(bin, param.t_slice);
  read(bin, param.colorvec);
}

//! Writer
void write(BinaryWriter& bin, const EvecDBKey_t& param)
{
  write(bin, param.t_slice);
  write(bin, param.colorvec);
}


//---------------------------------------------------------------------
//! Some struct to use
struct EvecDBData_t
{
  std::vector<std::complex<double>> data = std::vector<std::complex<double>>(Nc*Lx*Lx*Lx); 
};

//! Reader
void read(BinaryReader& bin, EvecDBData_t& param)
{
  read(bin, param.data);
}

//! Writer
void write(BinaryWriter& bin, const EvecDBData_t& param)
{
  write(bin, param.data);
}


//---------------------------------------------------------------------
// Simple concrete key class
template<typename K>
class EvecDBKey : public DBKey // DBKey defined in FILEDB
{
public:
  //! Default constructor
  EvecDBKey() {} 

  //! Constructor from data
  EvecDBKey(const K& k) : key_(k) {}

  //! Setter
  K& key() {return key_;}

  //! Getter
  const K& key() const {return key_;}

  // Part of Serializable
  const unsigned short serialID (void) const {return 456;}

  void writeObject (std::string& output) const throw (SerializeException) {
    BinaryBufferWriter bin;
    write(bin, key());
    output = bin.str();
  }

  void readObject (const std::string& input) throw (SerializeException) {
    BinaryBufferReader bin(input);
    read(bin, key());
  }

  // Part of DBKey
  int hasHashFunc (void) const {return 0;}
  int hasCompareFunc (void) const {return 0;}

  /**
    * Empty hash and compare functions. We are using default functions.
    */
  static unsigned int hash (const void* bytes, unsigned int len) {return 0;}
  static int compare (const FFDB_DBT* k1, const FFDB_DBT* k2) {return 0;}
  
private:
  K  key_;
};


//---------------------------------------------------------------------
// Simple concrete data class
template<typename D>
class EvecDBData : public DBData // DBData defined in FILEDB
{
public:
  //! Default constructor
  EvecDBData() {} 

  //! Constructor from data
  EvecDBData(const D& d) : data_(d) {}

  //! Setter
  D& data() {return data_;}

  //! Getter
  const D& data() const {return data_;}

  // Part of Serializable
  const unsigned short serialID (void) const {return 123;}

  void writeObject (std::string& output) const throw (SerializeException) {
    BinaryBufferWriter bin;
    write(bin, data());
    output = bin.str();
  }

  void readObject (const std::string& input) throw (SerializeException) {
    BinaryBufferReader bin(input);
    read(bin, data());
  }

private:
  D  data_;
};

//****************************************************************************
//! Prop operator
struct KeyTimeSliceColorVec_t
{
  KeyTimeSliceColorVec_t() {}
  KeyTimeSliceColorVec_t(int t_slice_, int colorvec_) : t_slice(t_slice_), colorvec(colorvec_) {}

  int        t_slice;       /*!< Source time slice */
  int        colorvec;      /*!< Colorstd::vector index */
};


//----------------------------------------------------------------------------
// KeyPropColorVec read
void read(BinaryReader& bin, KeyTimeSliceColorVec_t& param)
{
  read(bin, param.t_slice);
  read(bin, param.colorvec);
}

// KeyPropColorVec write
void write(BinaryWriter& bin, const KeyTimeSliceColorVec_t& param)
{
  write(bin, param.t_slice);
  write(bin, param.colorvec);
}



int v_ind(int color, int site){
    // color * space + position 
    return color*Lx*Lx*Lx + site;
}

//void get_evecs(vector<string> &filenames, vector<MatrixXcd> &evectors, MatrixXcd<MatrixXcd> &evectors_spatial, vector<int> &t_all_list){ 
void get_evecs(vector<string> &filenames, vector<MatrixXcd> &evectors, vector<int> &t_all_list){ 
    for(auto t: t_all_list){// t!=Nt_all; t++){
      evectors.push_back(MatrixXcd::Zero(Nvec, Lx*Lx*Lx*Nc));
      //for(int i; i<Nvec; i++){
      //  evectors_spatial(t,i) = MatrixXcd::Zero(Lx*Lx*Lx, Nc); 
      //}
    }
    
    
    // loop over t
    for(auto t: t_all_list){// t!=Nt_all; t++){
      typedef BinaryStoreDB< EvecDBKey<EvecDBKey_t>, EvecDBData<EvecDBData_t> > DBType_t_read;
      DBType_t_read db_read;
      const std::string dbase_read = filenames[t];
      db_read.open(dbase_read, O_RDONLY, 0664);

      // loop over distill vecs, find key value (t, distill_vec)
      for(int vec_num=0; vec_num< Nvec; ++vec_num){
        // ************ get values from .db file ************ //
        EvecDBData<EvecDBData_t> testDBData_read;
        //testDBData_read.data().data.at(7) = std::complex<double>(1.2, 3.4);
        
        EvecDBKey<EvecDBKey_t> testDBKey_read;
         
        testDBKey_read.key().t_slice = t;       // key to read 
        testDBKey_read.key().colorvec = vec_num;// key to read
      
        // get value from key, store in testDBData_read 
        db_read.get(testDBKey_read, testDBData_read);
        
        //************** fill CPP array *************//
        // loop over spatial points
        //MatrixXcd &spatial_color_matrix = evectors_spatial(t, vec_num); // reference to this matrix 
        for(int site=0; site < Lx*Lx*Lx; site++){
          // loop over color 
          for(int color=0; color < Nc; color++){
            std::complex<double> tmp = testDBData_read.data().data.at(site*Nc + color);
            evectors[t](vec_num, v_ind(color, site)) = tmp * complex<double>(1.0,0.0);
          } // end loop over color 
        } // end loop over sites
      } // end loop over Dist evecs 
    } // end loop over timeslices 
}


int v_ind_v2(int color, int site){
    // position * Nc + color 
    return site*Nc + color;
}

//void get_evecs(vector<string> &filenames, vector<MatrixXcd> &evectors, MatrixXcd<MatrixXcd> &evectors_spatial, vector<int> &t_all_list){ 
void get_evecs_v2(vector<string> &filenames, vector<MatrixXcd> &evectors, vector<int> &t_all_list){ 
    for(auto t: t_all_list){// t!=Nt_all; t++){
      evectors.push_back(MatrixXcd::Zero(Nvec, Lx*Lx*Lx*Nc));
      //for(int i; i<Nvec; i++){
      //  evectors_spatial(t,i) = MatrixXcd::Zero(Lx*Lx*Lx, Nc); 
      //}
    }
    
    
    // loop over t
    for(auto t: t_all_list){// t!=Nt_all; t++){
      typedef BinaryStoreDB< EvecDBKey<EvecDBKey_t>, EvecDBData<EvecDBData_t> > DBType_t_read;
      DBType_t_read db_read;
      const std::string dbase_read = filenames[t];
      db_read.open(dbase_read, O_RDONLY, 0664);

      // loop over distill vecs, find key value (t, distill_vec)
      for(int vec_num=0; vec_num< Nvec; ++vec_num){
        // ************ get values from .db file ************ //
        EvecDBData<EvecDBData_t> testDBData_read;
        //testDBData_read.data().data.at(7) = std::complex<double>(1.2, 3.4);
        
        EvecDBKey<EvecDBKey_t> testDBKey_read;
         
        testDBKey_read.key().t_slice = t;       // key to read 
        testDBKey_read.key().colorvec = vec_num;// key to read
      
        // get value from key, store in testDBData_read 
        db_read.get(testDBKey_read, testDBData_read);
        
        //************** fill CPP array *************//
	// loop over color 
	for(int color=0; color < Nc; color++){
	  // loop over spatial points
	  //MatrixXcd &spatial_color_matrix = evectors_spatial(t, vec_num); // reference to this matrix 
	  for(int site=0; site < Lx*Lx*Lx; site++){
            std::complex<double> tmp = testDBData_read.data().data.at(site*Nc + color);
            evectors[t](vec_num, v_ind_v2(color, site)) = tmp * complex<double>(1.0,0.0);
	  } // end loop over sites
	} // end loop over color 
      } // end loop over Dist evecs 
    } // end loop over timeslices 
}


#endif
