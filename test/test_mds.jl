using DistStat, Random, LinearAlgebra, Test

if get(ENV,"JULIA_MPI_TEST_ARRAYTYPE","") == "CuArray"
    using CuArrays
    A = CuArray
else
    A = Array
end

type=[Float64,Float32]

include("mds_fun.jl")
include("cmdline.jl")

variables = parse_commandline_mds()

m = 10
n = 10
r = 4
interval = variables["step"]
init_opt = variables["init_from_master"]
seed = variables["seed"]
eval_obj = variables["eval_obj"]

for T in type
    X1=MPIMatrix{T, A}(undef, m, n)
    rand!(X1; common_init=init_opt, seed=0)
    Y1=MPIMatrix{T, A}(undef, n, n)
    DistStat.euclidean_distance!(Y1, X1)

    X2=MPIMatrix{T,A}(undef,m,n)
    rand!(X2; common_init=init_opt, seed=0)
    Y2=MPIMatrix{T,A}(undef,n,n)
    DistStat.euclidean_distance!(Y2, X2)

    iter1=1000; iter2=1100
    u1 = MDSUpdate(;maxiter=iter1, step=interval, verbose=true)
    u2 = MDSUpdate(;maxiter=iter2, step=interval, verbose=true)
    v1 = MDSVariables(Y1, r; eval_obj=eval_obj, seed=seed)
    v2 = MDSVariables(Y1, r; eval_obj=eval_obj, seed=seed)
    reset!(v1; seed=seed)
    reset!(v2; seed=seed)

    tol=1e-06

    result1=mds!(Y1,u1,v1)
    result2=mds!(Y2,u2,v2)

    @test abs(result1-result2)<tol

end
