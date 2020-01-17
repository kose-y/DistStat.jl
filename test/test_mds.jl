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

m = 10
n = 10
r = 4
init_opt = true
seed = 777
eval_obj = true

for T in type
    X=MPIMatrix{T, A}(undef, m, n)
    rand!(X; common_init=init_opt, seed=0)
    Y=MPIMatrix{T, A}(undef, n, n)
    DistStat.euclidean_distance!(Y, X)

    iter=15; interval=1

    u = MDSUpdate(;maxiter=iter, step=interval, verbose=true)
    v = MDSVariables(Y, r; eval_obj=eval_obj, seed=seed)
    reset!(v; seed=seed)

    ans=3.646210055153943
    result=mds!(Y,u,v)

    @test isapprox(ans,result)

end
