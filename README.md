# DistStat

[![Build Status](https://travis-ci.com/kose-y/DistStat.jl.svg?branch=master)](https://travis-ci.com/kose-y/DistStat.jl)

Distributed statistical computing


### Installation Information

Dependencies:
- Julia >= 1.1 (recommended) >=0.7 (required. Version 0.7 and 1.0 throws a lengthy warning with CUDA-related packages)
- MPI installation (MPI.jl is tested on OpenMPI, MPICH, and Intel MPI)
- MPI.jl >= 0.10.0 (and its dependencies: defined in `Manifest.toml`. MPI.jl now supports CUDA-aware MPI)
- CustomUnitRanges.jl
- See `Project.toml`

For CUDA support:
- CUDA >= 9.0
- CuArrays.jl, CUDAnative.jl, CUDAdrv.jl, GPUArrays.jl (and their dependencies: defined in `Manifest.toml`)
- CUDA-aware MPI installation (of OpenMPI, MPICH, and Intel MPI, only OpenMPI supports CUDA)
