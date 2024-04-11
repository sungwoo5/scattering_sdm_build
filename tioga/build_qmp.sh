#!/bin/bash

source ./env.sh

QMP=qmp
pushd ${SRCDIR}/${QMP}
autoreconf
popd

pushd ${BUILDDIR}

if [ -d ./build_${QMP} ];
then
  rm -rf ./build_${QMP}
fi

mkdir  ./build_${QMP}
cd ./build_${QMP}

${SRCDIR}/qmp/configure --prefix=${INSTALLDIR}/qmp \
    --with-qmp-comms-type=MPI \
    --with-qmp-comms-cflags="${MPI_CFLAGS}" \
    --with-qmp-comms-ldflags="${MPI_LDFLAGS}" \
    --with-qmp-comms-libs="${MPI_LDFLAGS}" \
    CC="${CC}" \
    CFLAGS="${CFLAGS}"

# \
#     # --host=x86_64-linux-gnu \
#     --build=none

# cmake ${SRCDIR}/${QMP} -DQMP_MPI=ON \
# 	-DCMAKE_C_COMPILER=${CC} \
# 	-DCMAKE_INSTALL_PREFIX=${INSTALLDIR}/${QMP}  \
# 	-DCMAKE_C_FLAGS="${MPI_CFLAGS}" \
# 	-DCMAKE_EXE_LINKER_FLAGS="${MPI_LDFLAGS}" \
# 	-DCMAKE_SHARED_LINKER_FLAGS="${MPI_LDFLAGS}" \
#   -DBUILD_SHARED_LIBS=ON \
# 	-DCMAKE_C_STANDARD=99 \
# 	-DCMAKE_C_EXTENSIONS=OFF

# 	# -DCMAKE_BUILD_TYPE=${PK_BUILD_TYPE} \
# 	# -DCMAKE_C_COMPILER_WORKS=1 \

# cmake --build . -j 32  -v
# cmake --install .


make -j 14
make install

popd

