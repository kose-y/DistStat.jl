export COMPILERVARS_ARCHITECTURE=intel64
export MKLVARS_INTERFACE=ilp64

. /shared/intel/mkl/bin/mklvars.sh intel64 ilp64
export PATH=/shared/julia-1.2.0/bin:
export LD_LIBRARY_PATH=/shared/julia-1.2.0/lib:
export JULIA_DEPOT_PATH=/shared/julia_pkgs
