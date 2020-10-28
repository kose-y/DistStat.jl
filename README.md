# DistStat.jl

[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://kose-y.github.io/DistStat.jl/dev)
[![codecov](https://codecov.io/gh/kose-y/DistStat.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/kose-y/DistStat.jl)
[![build Actions Status](https://github.com/kose-y/DistStat.jl/workflows/build/badge.svg)](https://github.com/kose-y/DistStat.jl/actions)

<!--[![Build Status](https://travis-ci.com/kose-y/DistStat.jl.svg?branch=master)](https://travis-ci.com/kose-y/DistStat.jl)-->

DistStat.jl: Towards unified programming for high-performance statistical computing environments in Julia


### Installation Information

Dependencies:
- Julia >= 1.4
- An MPI installation (tested on OpenMPI, MPICH, and Intel MPI)
- MPI.jl >= 0.15.0 (and its dependencies)
    - Select proper MPI backend when building MPI.jl, as described in [this page](https://juliaparallel.github.io/MPI.jl/stable/configuration/))
- CustomUnitRanges.jl
- See `Project.toml`

For CUDA support:
- CUDA >= 9.0
- CUDA.jl, GPUArrays.jl (and their dependencies)
- CUDA-aware MPI installation (of OpenMPI, MPICH, and Intel MPI, only OpenMPI supports CUDA)
- MPI.jl should be built with the environment variable `JULIA_MPI_BINARY=system`; see [this page](https://juliaparallel.github.io/MPI.jl/stable/configuration/)).

To install the package, run the following code in Julia.

```julia
using Pkg
pkg"add https://github.com/kose-y/DistStat.jl"
```

### Examples

Examples of nonnegative matrix factorization, multidimensional scaling, and l1-regularized Cox regression is provided in the directory `examples/`. Settings for multi-gpu experiments and multi-instance cloud experiments are also provided.
