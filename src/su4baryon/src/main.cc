// We are specializing this to mesons.

#include <fstream>
#include <iostream>
#include <cstdio>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <iomanip>
#include <complex>
#include <map>
#include "qdp.h" 

#include "global_variables.h" // written and updated by config_loop.py
#include "get_perams.h"
#include "get_evecs.h"
#include "compute_t_gpu.h"
#include "define_diagrams.h"
#include "eigen_baryon_constructor.h"
#include "compute_Bprop_gpu.h"
#include "read_spin_wick.h"
// #include "spin_sparse.h"
#include "timer.h"
#include "omp.h"
#include "utils.h"

using namespace QDP; 
using namespace FILEDB;
using namespace Eigen;
// using namespace Chroma;
using namespace std;
using cd = complex<double>;
using mat = MatrixXcd;
using Tensor2 = Eigen::Tensor<std::complex<double>,2>;
using Tensor4 = Eigen::Tensor<std::complex<double>,4>;


// global variables 
int Lx, Ly, Lz, Nt, Nvec;

template<typename Out> void split(const string &s, char delim, Out result)
{
  stringstream ss(s);
  string item;
  while(getline(ss, item, delim))
  {
    *(result++) = item;
  }
}

vector<string> split(const string&s, char delim)
{
  vector<string> elems;
  split(s, delim, back_inserter(elems));
  return elems;
}


std::map<std::string, std::string> load_input(string input_filename);

int main(int argc, char *argv[]){
    omp_set_dynamic(0); 
    START_CODE();
    // Put the machine into a known state
    QDP_initialize(&argc, &argv);
	  
    std::vector<mat> gamma(7);
    std::complex<double> I(0,1);
    ///anti-milc basis
    gamma[1].resize(4,4); gamma[1] << 0, 0, 0, -I, 0, 0, -I, 0, 0, I, 0, 0, I, 0, 0, 0;
    gamma[2].resize(4,4); gamma[2] << 0, 0, 0, 1, 0, 0, -1, 0, 0, -1, 0, 0, 1, 0, 0, 0;
    gamma[3].resize(4,4); gamma[3] << 0, 0, -I, 0, 0, 0, 0, I, I, 0, 0, 0, 0, -I, 0, 0;
    gamma[4].resize(4,4); gamma[4] << 0, 0, -1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, -1, 0, 0;
    gamma[5].resize(4,4); gamma[5] = gamma[1]*gamma[2]*gamma[3]*gamma[4];
    gamma[6].resize(4,4); gamma[6] = mat::Identity(4,4);

    mat C = complex<double>(0.0,1.0)*gamma[2]*gamma[4];
    mat cg5 = C*gamma[5];

    ofstream gammaFile;
    gammaFile.open("GAMMAS/buchoff_gamma.txt");
    for(size_t s0=0; s0<Ns; ++s0)
    for(size_t s1=0; s1<Ns; ++s1)
    for(size_t s2=0; s2<Ns; ++s2)
    for(size_t s3=0; s3<Ns; ++s3)
    {
      complex<double> val = cg5(s0,s1)*cg5(s2,s3);
      gammaFile << s0 << " " << s1 << " " << s2 << " " << s3 << " " << val.real() << " " << val.imag() << std::endl;
    }
    gammaFile.close();

    std::cout << "   S T A R T   C O D E   " << std::endl;
    // std::cout << "   Nvec   " << Nvec << std::endl;
   


    auto input = load_input(argv[1]);
    auto perambulator_filebase = input["PerambulatorFileBase"];
    input.erase("PerambulatorFileBase");
    auto fnames_base = input["EvecFileBase"];
    input.erase("EvecFileBase");
    auto cfg = stoi(input["cfg"]);
    input.erase("cfg");
    auto NDIAGS = stoi(input["NDIAGS"]);
    input.erase("NDIAGS");
    auto diagram_names_filename = input["DiagramNamesFilename"];
    input.erase("CORRELATION_MATRIX_BASIS_STRING");
    


    Lx = stoi(input["Lx"]);
    Ly = stoi(input["Ly"]);
    Lz = stoi(input["Lz"]);
    Nt = stoi(input["Nt"]);
    Nvec = stoi(input["Nvec"]);
    
    // perambulator Nvec at source and sink
    int Nvec_src, Nvec_snk;
    Nvec_src=stoi(input["Nvec_src"]);
    Nvec_snk=stoi(input["Nvec_snk"]);

    std::cout << "Read from input:" << std::endl;
    std::cout << "  Lx Ly Lz Nt Nvec = " << Lx << " " << Ly << " " << Lz << " " << Nt << " " << Nvec << " " << std::endl;
    std::cout << "  (perambulator cutoff) Nvec_snk Nvec_src = " << Nvec_snk << " " << Nvec_src << " " << std::endl;


    std::vector<int> unique_dts, unique_ts, peram_sources;
    int N_dts, N_ts;		// number of dts, number of ts
    if(input["dts"]=="ALL")
    {
      for(size_t i=0; i<Nt; ++i)
        unique_dts.push_back(i);
    }
    else
    {
      auto dts = split(input["dts"], ',');
      for(auto &s : dts){
        std::cout << s << std::endl;
        unique_dts.push_back(stoi(s));
      }
    }
    input.erase("dts");
    
    // test if the sink time slices include 0 as first element, dts[0]==0
    if(unique_dts[0]!=0)
      {
	std::cerr<< "" << endl;
	QDP::QDP_abort(10);
      }
    N_dts=unique_dts.size();

    if(input["ts"]=="ALL")
    {
      for(size_t i=0; i<Nt; ++i)
        unique_ts.push_back(i);
    }
    else
    {
      auto ts = split(input["ts"], ',');
      for(auto &s : ts)
        unique_ts.push_back(stoi(s));
    }
    input.erase("ts");
    N_ts=unique_ts.size();
    
    
    if(input["peram_sources"]=="ALL")
    {
      for(size_t i=0; i<Nt; ++i)
        peram_sources.push_back(i);
    }
    else
    {
      auto sources = split(input["peram_sources"], ',');
      for(auto &s : sources)
        peram_sources.push_back(stoi(s));
    }


    auto all_mom = split(input["mom"], ',');
    input.erase("mom");
    vector< vector<int> > unique_mom(all_mom.size());
    for(size_t i=0; i<all_mom.size(); ++i)
    {
      unique_mom[i].resize(3);
      std::stringstream ss(all_mom[i]);
      ss >> unique_mom[i][0] >> unique_mom[i][1] >> unique_mom[i][2];
    } 

 
    // comma separated list of files with rank 4 tensors.
    auto all_gammas = split(input["gammas"], ',');
    input.erase("gammas"); 
    vector<Tensor4> unique_gammas;
    for(size_t i=0; i<all_gammas.size(); ++i)
    {
      std::stringstream ss(all_gammas[i]);
      std::string tmp(ss.str());
  	  unique_gammas.push_back(spin_matrix_from_file(tmp));
  	}

    vector< vector<Spin4> > unique_gammas_sparse;
    for(size_t i=0; i<all_gammas.size(); ++i)
    {
      std::stringstream ss(all_gammas[i]);
      std::string tmp(ss.str());
  	  unique_gammas_sparse.push_back(spintensor_from_file_sparse(tmp));
  	}
    

    std::ifstream dnames_file(diagram_names_filename);
    std::string s;
    std::vector<std::string> diagram_names;
    while(std::getline(dnames_file, s))
    {
      if(s.size()>0)
        diagram_names.push_back(s);
    }
    dnames_file.close();
    
//    std::string perambulator_filename = argv[1];
//    std::string fnames_base = argv[2];
//    cout << "\n\n" << fnames_base << "\n\n"  << endl;
   
    //TODO read from input 
    vector< vector<int> > unique_disp(1);
    unique_disp[0] = vector<int>{0};
    std::string output_prefix = "";
    std::stringstream ss;

    int ngamma=unique_gammas.size();
    int ndisp=unique_disp.size();
    int nmom=unique_mom.size();
    int NQL=ngamma*ndisp*nmom; 

    // load perambulators i
    vector<string> taufnames(Nt);
    std::vector<MatrixXcd> tau(Nt*Nt*Nd*Nd,MatrixXcd::Zero(Nvec,Nvec));
    if (perambulator_filebase=="DUMMY")
      {
	std::cout << "Use dummy values for perambulators" << std::endl; 
      }
    else
      for(const auto &t: peram_sources){
	std::ostringstream oss;
	oss << perambulator_filebase << t << "_cfg" << cfg << ".sdb";
	get_tau( oss.str(), tau);
      }
  
    //--------------------------------------------------------------
    // Compute omega, 
    // note indices are same as tau, source on right, sink on left
    // t0=source is most varying index
    Timer<> omega_timer("Compute omega");
    std::vector<Eigen::Tensor<std::complex<double>,2>> omegaTilde(Nt*Nt);
    for(size_t t=0; t<Nt; ++t)	// sink time
    for(size_t t0=0; t0<Nt; ++t0) // source time
    {
      Eigen::Tensor<std::complex<double>,2> omega(Nvec*Ns,Nvec*Ns);
      omega.setConstant(std::complex<double>(0.,0.));
      for(size_t s0=0; s0<Ns; ++s0) // sink
      for(size_t s1=0; s1<Ns; ++s1) // source
      for(size_t l0=0; l0<Nvec_snk; ++l0) // sink
      for(size_t l1=0; l1<Nvec_src; ++l1) // source
      {
        for(size_t s2=0; s2<Ns; ++s2)
        {
          //omega(t,t0)
          omega(s0*Nvec+l0, s1*Nvec+l1) += tau[tau_ind(t,t0,s0,s2)](l0,l1)*gamma[5](s2,s1);  
        }
      }
      omegaTilde[t*Nt+t0]=omega;
    }
    omega_timer.stop<std::chrono::milliseconds>("ms");

    //--------------------------------------------------------------
    std::cout << "Kimmy edit - previously loading/computing for all t, only want dts. But my fix is hardwired for t0=0" << std::endl; 
    // load all Nt eigen vectors 
    Timer<> load_eigen_timer("Load eigenvectors");
    vector<string> fnames(Nt); 
    for(size_t t=0; t<Nt; ++t){  
      std::ostringstream oss;
      oss <<  fnames_base + "t" << t << "_evecs.db";
      fnames[t] = oss.str();
    }
    
    vector<int> t_all_list;
    for(auto dts: unique_dts){
      t_all_list.push_back(dts);
    }

    vector<MatrixXcd> evectors;	   // [t][i][ color*vol + site ] ordering
    if (fnames_base=="DUMMY")
      {
	std::cout << "Use dummy values for eigenvectors" << std::endl; 
	dummy_evecs(fnames, evectors, t_all_list);
      }
    else
      {
	get_evecs(fnames, evectors, t_all_list);
      }
    // vector<MatrixXcd> evectors_v2; // [t][i][ site*Nc + color  ] ordering
    // get_evecs_v2(fnames, evectors_v2, t_all_list);
    // t_all_list = unique_dts;

    std::cout << "dts = "; 
    for(auto dts: unique_dts)
      std::cout << dts << " ";
    std::cout << std::endl;


    load_eigen_timer.stop<std::chrono::milliseconds>("ms");

 
    //-----------------------------
    // GPU T Function
    // source and sink for all Nt 
    // Kimmy edit to t_all_list affects this 
    vector<vector<Tensor4>> Tsink(ndisp*nmom), Tsource(ndisp*nmom);
    for(size_t d=0; d<unique_disp.size(); ++d)
    for(size_t p=0; p<unique_mom.size(); ++p)
    {
      for(auto t: t_all_list){
	std::cout << "\nStart computing T for timeslice " << t << std::endl; 
	// sink
	// Timer<> baryonT_gpu_timer("Constructing T's with GPU");
	// Tensor4 _Tsinkv2=create_Tsink_gpu_v2(evectors_v2,t, unique_mom[p][0], unique_mom[p][1], unique_mom[p][2]);
	// baryonT_gpu_timer.stop<std::chrono::milliseconds>("ms");

	Timer<> baryonT_cublas_timer("Constructing T's with cublas");
	// Tensor4 _Tsink(Nvec,Nvec,Nvec,Nvec);
	Tensor4 _Tsink=create_Tsink_cublas(evectors, t, unique_mom[p][0], unique_mom[p][1], unique_mom[p][2]);
        baryonT_cublas_timer.stop<std::chrono::milliseconds>("ms");

	Tsink[d*nmom+p].push_back(_Tsink);    

	// source: complex conjugate of the sink
        // Tsource[d*nmom+p].push_back(create_Tsource_gpu_v2(evectors_v2,t, unique_mom[p][0], unique_mom[p][1], unique_mom[p][2]));	
	Timer<> baryonT_source_timer("Constructing T's at source");

	Tensor4 _Tsource(Nvec,Nvec,Nvec,Nvec);
	cd* _Tsink_1d=_Tsink.data();	// 1d access to Tensor4
	cd* _Tsource_1d=_Tsource.data();
	// copy comp conj of Tsink (independent of the layout, RowMajor of ColMajor)
#pragma omp parallel for	// not useful for lassen power9?
	for(size_t ii=0; ii<Nvec*Nvec*Nvec*Nvec; ++ii)
	  {
	    _Tsource_1d[ii]=conj(_Tsink_1d[ii]);
	  }
        Tsource[d*nmom+p].push_back(_Tsource);	

        baryonT_source_timer.stop<std::chrono::milliseconds>("ms");
       // end loop over time
      }
    }



    //---------------
    // diagram array 
    std::vector<std::vector<std::vector<std::complex<double>>>> diagram_values(NDIAGS);
    for(size_t i=0; i<NDIAGS; ++i)
      diagram_values[i].resize(unique_dts.size());
    for(size_t i=0; i<NDIAGS; ++i)
      for(size_t dt=0; dt<diagram_values[i].size(); ++dt)
    	diagram_values[i][dt].resize(unique_ts.size());
    for(size_t i=0; i<NDIAGS; ++i)
      for(size_t dt=0; dt<diagram_values[i].size(); ++dt)
    	for(size_t t=0; t<diagram_values[i][dt].size(); ++t)
    	  diagram_values[i][dt][t]=0;




    //=======================================================================================
    // Baryon routines for "t" (a each sink time slice) one by one
    //=======================================================================================
    for(auto t0: unique_ts)
    // for(auto t: t_all_list)
    for(auto dt: unique_dts)
      {

    	// currently assume t0=0 only
    	if (t0!=0) {
    	  std::cerr << "Currently support t0=0 single source time case only" << std::endl;
    	  exit(10);
    	}
    	int t=t0+dt;

    	std::cout << "\n\n======================================" << std::endl; 
    	std::cout << "Start baryon routines for t_sink=" << t << std::endl; 

    	//---------------------------------
    	// Compute baryon (sink) functions with sparse spin tensor
    	// = (delta)^{s0',l0'; s0,l0}
    	// * (delta)^{s1',l1'; s1,l1} 
    	// * (delta)^{s2',l2'; s2,l2} 
    	// * (delta)^{s3',l3'; s3,l3}
    	// * (Tsink;d,p,t)^{l0,l1,l2,l3} * unique_gammas(g)^{s0,s1,s2,s3}
    	// = (delta)^{s0',l0'; s0,l0}
    	// * (delta)^{s1',l1'; s1,l1} 
    	// * (delta)^{s2',l2'; s2,l2} 
    	// * (delta)^{s3',l3'; s3,l3}
    	// * (Tsink;d,p,t)^{l0,l1,l2,l3} * unique_gammas(g)^{s0,s1,s2,s3}
    	Timer<> baryonFunc_1gpu_timer("Constructing B's with cublas");
    	// vector<vector<Tensor4>> baryonSink(ngamma*ndisp*nmom), baryonSource(ngamma*ndisp*nmom);

    	// baryon sink given at t
    	vector<Tensor4> baryonSink(ngamma*ndisp*nmom);

    	for(size_t g=0; g<unique_gammas.size(); ++g)
    	for(size_t d=0; d<unique_disp.size(); ++d)
    	for(size_t p=0; p<unique_mom.size(); ++p)
    	  {
    	    vector<Spin4> spintensor = unique_gammas_sparse[g];


    	    // calculate Bsink at (t)
		
    	    Tensor4 Gamma=unique_gammas[g];
    	    baryonSink[g*ndisp*nmom + d*nmom + p]=create_baryonSink(Tsink[d*nmom+p][t],Gamma);	    


    	    //=============
    	    // TODO: Following gpu routine is not working yet...
    	    //=============
    	    // Tensor4 Bsink_t=compute_Bsink_gpu(spintensor, Tsink[d*nmom+p][t], Ns, Nvec);
    	    // baryonSink[g*ndisp*nmom + d*nmom + p]=Bsink_t;

    	  }
      
    	baryonFunc_1gpu_timer.stop<std::chrono::milliseconds>("ms");





    

    	//--------------------------------------------------------------
    	// Compute Bprop with sparse spin tensor
    	// = (omega;t,t0)^{s0',l0'; s0,l0}
    	// * (omega;t,t0)^{s1',l1'; s1,l1} 
    	// * (omega;t,t0)^{s2',l2'; s2,l2} 
    	// * (omega;t,t0)^{s3',l3'; s3,l3} 
    	// * (Tsource;d,p,t0)^{l0,l1,l2,l3} * unique_gammas(g)^{s0,s1,s2,s3}
    	Timer<> baryonProp_sparse_1gpu_timer("Constructing Bprops's using sparse spin contraction with GPU (cuBLAS)");

    	std::cout << "Eigen thread num = " << Eigen::nbThreads() << std::endl;
    	// vector<vector<Tensor4>> baryonProp(ngamma*ndisp*nmom);
    	vector<Tensor4> baryonProp(ngamma*ndisp*nmom);
    	for(size_t g=0; g<unique_gammas_sparse.size(); ++g) 
    	for(size_t d=0; d<unique_disp.size(); ++d)
    	for(size_t p=0; p<unique_mom.size(); ++p)
    	  {
	    
    	    // run only for the specific source and sink slices
    	    int t1=(t0+dt)%Nt;
		  
    	    // std::cout << "g:" << g << ", t0:" << t0  << ", t1:" << t1  << std::endl;
    	    Tensor2 omega=omegaTilde[t1*Nt+t0];
    	    Tensor4 tsource=Tsource[d*nmom+p][t0]; // [Nvec,Nvec,Nvec,Nvec]

    	    vector<Spin4> spintensor = unique_gammas_sparse[g];

    	    // calculate Bprop at (t0,t1)
    	    // std::cout << "run compute_Bprop_gpu() for t1=" << t1 << ", t0=" << t0 << std::endl;
    	    Tensor4 Bprop_t0t1=compute_Bprop_gpu(spintensor, omega, tsource, Ns, Nvec);

    	    // baryonProp[g*ndisp*nmom + d*nmom + p][t1*Nt+t0]
    	    baryonProp[g*ndisp*nmom + d*nmom + p]=Bprop_t0t1;
    	  }
    	baryonProp_sparse_1gpu_timer.stop<std::chrono::milliseconds>("ms");



    	//--------------------------------------------------------------
    	// Contract diagrams
    	Timer<> contract_timer("Contracting diagrams");

    	std::cout << "compute_diagrams for t0 = " << t0 << ", dt = " << dt  << std::endl;
	  
    	std::vector<complex<double>> diagrams(NDIAGS,0.);
    	compute_diagrams(diagrams, baryonProp, baryonSink);
    	for(size_t i=0; i<NDIAGS; ++i)
    	  diagram_values[i][dt][t0]=diagrams[i];

    	contract_timer.stop<std::chrono::milliseconds>("ms");
      }

    //--------------------------------------
    // print diagrams 
    Timer<> print_timer("Print diagrams");
    std::ofstream diagram_value_file("DIAGRAMS/diagrams_cfg"+to_string(cfg)+".txt");
    for(int i=0; i<NDIAGS; ++i)
    {
      diagram_value_file << diagram_names[i] << "\n";
      for(int dt=0; dt<unique_dts.size(); ++dt)
        for(int t=0; t<unique_ts.size(); ++t)
        {
          diagram_value_file << std::fixed << std::setprecision(0) << unique_dts[dt] << " " << unique_ts[t] << " ";
          diagram_value_file << std::scientific << std::setprecision(10) << diagram_values[i][dt][t].real() << " " << diagram_values[i][dt][t].imag() << "\n";   
        }
    }
    diagram_value_file.close();
    print_timer.stop<std::chrono::milliseconds>("ms");

    std::cout << "   E N D  C O D E   ---   S U C C E S S !" << std::endl;
    return 0;
}



std::map<std::string, std::string> load_input(std::string input_filename)
{
  ifstream input(input_filename);
	map<string, string> name_value;

	string line;
	while( getline(input, line) )
	{
		auto data = split(line,'=');
		name_value[ data[0] ] = data[1];
	}

  return name_value;
}




















