#ifndef READ_SPIN_WICK_H
#define READ_SPIN_WICK_H

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream> // reads and writes files 
#include <sstream> // get sub strings 
#include <iterator>
#include <algorithm>
#include <complex>
#include <vector> 
#include <set>
#include <map>
#include <unordered_set>
#include <string>
#include "Eigen/Dense"
#include <unsupported/Eigen/CXX11/Tensor>
// #include "get_perams.h"
// #include "get_evecs.h"
#include "spin_sparse.h"	/* Spin4 */

// simple splitting funtion for " " delimeter 
vector<string> split(string &line) {
  istringstream is_line(line);
  vector<string> words;
  copy(istream_iterator<string>(is_line),
       istream_iterator<string>(),
       back_inserter(words));
  return words;
}


// SPARSE SPIN MATRIX
// read text file into sparse rank-4 spin tensor 
vector<Spin4> spintensor_from_file_sparse(string spin_matrix_file){
  
  cout << "Reading sparse spin matrix from file " << spin_matrix_file << std::endl;

  // initialize spin matrix output
  vector<Spin4> matrix_sp;

  // Create a text string, which is used to output the text file
  string line;  
  // Read from the text file
  ifstream MyReadFile(spin_matrix_file);

  // // Use a while loop together with the getline() function to read the file line by line
  while (getline (MyReadFile, line)) {
    // split line into words
    string delim(" ");
    vector<string> words = split(line); 
    
    // convert each word to stringstream 
    stringstream alpha_s(words[0]);
    stringstream beta_s(words[1]);
    stringstream sigma_s(words[2]);
    stringstream delta_s(words[3]);
    stringstream value_real_s(words[4]);
    stringstream value_imag_s(words[5]);
    
    // convert each stringstream to desired type 
    int alpha;
    int beta;
    int sigma;
    int delta;
    double value_real, value_imag;
    alpha_s >> alpha;
    beta_s >> beta;
    sigma_s >> sigma;
    delta_s >> delta;
    value_real_s >> value_real;
    value_imag_s >> value_imag;
    // assume real value 
    
    
    Spin4 elem;
    elem.s[0]=alpha;
    elem.s[1]=beta;
    elem.s[2]=sigma;
    elem.s[3]=delta;
    elem.val=cd(value_real, value_imag);
    matrix_sp.push_back(elem);
  }
  MyReadFile.close();
  
  
  return matrix_sp;
}


// SPIN MATRIX
// read text file into 4d Eigen tensor 
Eigen::Tensor<cd,4> spin_matrix_from_file(string spin_matrix_file){
  
  cout << "Reading spin matrix from file " << spin_matrix_file << std::endl;

  // initialize spin matrix output
  //Nc = 4 // 4 indices due to 4 colors (4 quarks over which to sum spins)
  //Nd = 4 // number of dimension (Dirac spinors)
  Eigen::Tensor<cd,Nc> matrix(Nd,Nd,Nd,Nd);
  matrix.setZero();

  // Create a text string, which is used to output the text file
  string line;  
  // Read from the text file
  ifstream MyReadFile(spin_matrix_file);

  // // Use a while loop together with the getline() function to read the file line by line
  while (getline (MyReadFile, line)) {
    // split line into words
    string delim(" ");
    vector<string> words = split(line); 
    
    // convert each word to stringstream 
    stringstream alpha_s(words[0]);
    stringstream beta_s(words[1]);
    stringstream sigma_s(words[2]);
    stringstream delta_s(words[3]);
    stringstream value_real_s(words[4]);
    stringstream value_imag_s(words[5]);
    
    // convert each stringstream to desired type 
    int alpha;
    int beta;
    int sigma;
    int delta;
    double value_real, value_imag;
    alpha_s >> alpha;
    beta_s >> beta;
    sigma_s >> sigma;
    delta_s >> delta;
    value_real_s >> value_real;
    value_imag_s >> value_imag;
    // assume real value 
    
    
    cd value = cd(value_real, value_imag);
    
    matrix(alpha,beta,sigma,delta) = value;
  }
  MyReadFile.close();
  
  
  return matrix;
}

#endif
