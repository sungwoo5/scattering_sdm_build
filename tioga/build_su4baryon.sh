#!/bin/bash
source env.sh

pushd ${BUILDDIR}

BARYON=su4baryon

if [ -d ./build_${BARYON} ];
then
    rm -rf ./build_${BARYON}
fi
if [ -d ../install/${BARYON} ];
then
    rm -rf ../install/${BARYON}
fi
if [ ! -d ../install/chroma_Nc4 ];
then
    echo "Need install/chroma_Nc4"
    exit 1
fi

cp -a ../src/${BARYON} build_${BARYON}
cd ./build_${BARYON}

make
make install

popd
