#!/bin/bash
source ./env.sh

CHROMA=chroma_Nc4


pushd ${BUILDDIR}

if [ -d ./build_${CHROMA} ];
then
  rm -rf ./build_${CHROMA}
fi

mkdir  ./build_${CHROMA}
cd ./build_${CHROMA}


# Configure using the relevant variables
${SRCDIR}/chroma/configure --prefix=${INSTALLDIR}/${CHROMA} \
    --with-qdp=${INSTALLDIR}/qdpxx_lapH \
    --with-qmp=${BASE_INSTALLDIR}/qmp \
    CC="${CC}"  CXX="${MPICXX}" \
    CXXFLAGS="${MPI_CFLAGS} ${CXXFLAGS}" \
    CFLAGS="${MPI_CFLAGS} ${CFLAGS}" \
    LDFLAGS="-Wl,-zmuldefs" \
    --host=x86_64-linux-gnu --build=none \
    --enable-clover \
    ${OMPENABLE} \
    # --with-libxml2=${BASE_INSTALLDIR}/libxml2/lib64 \
    #--with-quda=${INSTALLDIR}/quda \
    #--with-cuda=${CUDA_HOME} \
    # --enable-layout=cb2 
    # --enable-static-packed-gauge \
    # --enable-fused-clover-deriv-loops
    ##--with-mg-proto=${BASE_INSTALLDIR}/mg_proto

make -j 14

make install

popd
popd
