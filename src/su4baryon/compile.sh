#!/bin/bash -l

# source ../../env.sh

include=./include

Nc=4
op="original_Sungwoo"
define_diagrams_gpu_correlation_matrix_specific="define_diagrams_gpu_${op}"  # .cpp added in compile line 
echo ${define_diagrams_gpu_correlation_matrix_specific}
mkdir -p bin
gpuexe="bin/main_Nc${Nc}_${op}_gpu.exe"
rm ${gpuexe}
#rm src/*.o


MAIN_GPU=main.cpp

echo "C O M P I L I N G"

CHROMA=chroma_Nc4
CONFIG=../${CHROMA}/bin/chroma-config
CXX=$(${CONFIG} --cxx)
CXXFLAGS=$(${CONFIG} --cxxflags)
LDFLAGS=$(${CONFIG} --ldflags)
LIBS=$(${CONFIG} --libs)

Ieigen=../../src/eigen

echo CXX=${CXX}
echo CXXFLAGS=${CXXFLAGS}
echo LDFLAGS=${LDFLAGS}
echo LIBS=${LIBS}
${CXX} --offload-arch=gfx90a -I${include} -I${Ieigen} ${CXXFLAGS} ${LDFLAGS} ${LIBS} -lamdhip64 -lhipblas -lrocblas -c src/compute_t_gpu.hip -o src/compute_t_gpu.o

${CXX} --offload-arch=gfx90a -I${include} -I${Ieigen} ${CXXFLAGS} ${LDFLAGS} ${LIBS} -lamdhip64 -lhipblas -lrocblas -c src/tfunc.hip -o src/tfunc.o
${CXX} --offload-arch=gfx90a -I${include} -I${Ieigen} ${CXXFLAGS} ${LDFLAGS} ${LIBS} -lamdhip64 -lhipblas -lrocblas -c src/compute_Bprop_gpu.hip -o src/compute_Bprop_gpu.o
${CXX} --offload-arch=gfx90a -I${include} -I${Ieigen} ${CXXFLAGS} ${LDFLAGS} ${LIBS} -lamdhip64 -lhipblas -lrocblas -c src/diagram_utils.hip -o src/diagram_utils.o

${CXX} -I${include} -I${Ieigen} ${CXXFLAGS} ${LDFLAGS} ${LIBS} -c src/utils.cpp -o src/utils.o
${CXX} -I${include} -I${Ieigen} ${CXXFLAGS} ${LDFLAGS} ${LIBS} -c src/${define_diagrams_gpu_correlation_matrix_specific}.cpp -o src/define_diagrams_gpu.o
${CXX} -I${include} -I${Ieigen} ${CXXFLAGS} ${LDFLAGS} ${LIBS} -c src/main.cc -o src/main.o

${CXX} --offload-arch=gfx90a -I${include} -I${Ieigen} ${CXXFLAGS} ${LDFLAGS} ${LIBS} -lamdhip64 -lhipblas -lrocblas src/utils.o src/define_diagrams_gpu.o src/tfunc.o src/compute_t_gpu.o src/compute_Bprop_gpu.o src/diagram_utils.o src/main.o  -o ${gpuexe} 
