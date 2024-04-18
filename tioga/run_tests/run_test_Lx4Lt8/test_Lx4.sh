#!/bin/bash
source ../../env.sh

# export OMP_NUM_THREADS=1

# mkdir -p DIAGRAMS
exe=../../install/su4baryon/bin/su4baryon
${exe} input_singlesource_Nvec4.in



# module load omniperf
# OMNIPERF="rocprof --stats"
# ${OMNIPERF} ${exe} ./input_singlesource_Nvec4_dummy.in

# OMNIPERF="omniperf profile -n peak_profile_Nvec4 --roof-only --sort kernels --device 0 --kernel-names -- "
# ${OMNIPERF} ${exe} ./input_singlesource_Nvec4_dummy.in
