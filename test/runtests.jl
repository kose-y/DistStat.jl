# Most part of this code is from the corresponding file from JuliaParallel/MPI.jl, which is under the UNLICENSE (no conditions whatsoever to the public domain)
using Test, MPI, DistStat

if get(ENV,"JULIA_MPI_TEST_ARRAYTYPE","") == "CuArray"
    import CUDA
    ArrayType = CUDA.CuArray
else
    ArrayType = Array
end


nprocs_str = get(ENV, "JULIA_MPI_TEST_NPROCS","")
nprocs = nprocs_str == "" ? clamp(Sys.CPU_THREADS, 2, 4) : parse(Int, nprocs_str)
testdir = @__DIR__
istest(f) = endswith(f, ".jl") && startswith(f, "test_")
testfiles = sort(filter(istest, readdir(testdir)))

@info "Running DistStat tests" ArrayType nprocs

@testset "$f" for f in testfiles
    mpiexec() do cmd
        run(`$cmd -n $nprocs $(Base.julia_cmd()) $(joinpath(testdir, f))`)
        @test true
    end
end
