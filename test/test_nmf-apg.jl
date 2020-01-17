using DistStat, Random, LinearAlgebra, Test

if get(ENV,"JULIA_MPI_TEST_ARRAYTYPE","") == "CuArray"
    using CuArrays
    A = CuArray
else
    A = Array
end

type=[Float64,Float32]

include("nmf-apg_fun.jl")
include("cmdline.jl")

m = 10
n = 10
r = 4
init_opt = true
seed = 777
eval_obj = true

for T in type
    X=MPIMatrix{T,A}(undef,m,n)
    rand!(X; common_init=init_opt, seed=0)

    iter=20; interval=1

    u = APGUpdate(;maxiter=iter, step=interval, verbose=true)
    v = NMFVariables(X, r; eval_obj=eval_obj, seed=seed)

    ans=4.161165558120161
    result=nmf!(X,u,v)

    @test isapprox(ans,result)

end
