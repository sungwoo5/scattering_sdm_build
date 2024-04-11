#!/bin/bash
#
#################
# BUILD QDPXX
#################
source ./env.sh


QDP=qdpxx_lapH

pushd ${BUILDDIR}

if [ -d ./build_${QDP} ];
then
  rm -rf ./build_${QDP}
fi

mkdir  ./build_${QDP}
cd ./build_${QDP}

${SRCDIR}/${QDP}/configure \
	--prefix=${INSTALLDIR}/${QDP} \
        --enable-parallel-arch=parscalar \
	--enable-precision=double \
        --enable-Nd=3 \
        --enable-Nc=4 \
	--enable-filedb \
	--enable-openmp \
	--disable-generics \
        --enable-largefile \
        --enable-parallel-io \
        --enable-alignment=64 \
	--with-qmp=${INSTALLDIR}/qmp \
	CXXFLAGS="${MPI_CFLAGS} ${CXXFLAGS} -fpermissive " \
	CFLAGS="${MPI_CFLAGS} ${CFLAGS}" \
	CXX="${MPICXX}" \
	CC="${CC}" \
	--host=x86_64-linux-gnu --build=none


	# --with-libxml2=${BASE_INSTALLDIR}/libxml2 \

make -j 14
make install

popd
