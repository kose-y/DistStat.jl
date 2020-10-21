# DistStat

[![Build Status](https://travis-ci.com/kose-y/DistStat.jl.svg?branch=master)](https://travis-ci.com/kose-y/DistStat.jl)

Distributed statistical computing


### Installation Information

Dependencies:
- Julia >= 1.4
- An MPI installation (tested on OpenMPI, MPICH, and Intel MPI)
- MPI.jl >= 0.15.0 (and its dependencies)
- CustomUnitRanges.jl
- See `Project.toml`

For CUDA support:
- CUDA >= 9.0
- CUDA.jl, GPUArrays.jl (and their dependencies)
- CUDA-aware MPI installation (of OpenMPI, MPICH, and Intel MPI, only OpenMPI supports CUDA)

To install the package, run the following code in Julia.

```julia
using Pkg
pkg"add https://github.com/kose-y/DistStat.jl"
```
