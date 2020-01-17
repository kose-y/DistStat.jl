using DistStat, Random, LinearAlgebra, Test, CuArrays, CUDAnative

if get(ENV,"JULIA_MPI_TEST_ARRAYTYPE","") == "CuArray"
    using CuArrays
    A = CuArray
else
    A = Array
end

type=[Float64,Float32]

include("cox-memoryefficient_fun.jl")
include("cmdline.jl")

variables = parse_commandline_cox()
m = 10
n = 10
init_opt = true
seed = variables["seed"]
eval_obj = variables["eval_obj"]
censor_rate = variables["censor_rate"]
lambda = variables["lambda"]

for T in type
    X = MPIMatrix{T, A}(undef, m, n)
    rand!(X; common_init=init_opt, seed=seed)
    δ = convert(A{T}, rand(m) .> censor_rate)
    DistStat.Bcast!(δ)

    iter=20; interval=1

    u = COXUpdate(;maxiter=iter, step=interval, verbose=true)
    t = convert(A{T}, collect(reverse(1:size(X,1))))
    v = COXVariables(X, δ, lambda, t; eval_obj=eval_obj)

    reset!(v; seed=seed)

    result=cox!(X,u,v)
    ans=(0.016952382618763417,10.0)

    @test isapprox(ans[1],result[1])
    @test isapprox(ans[2],result[2])

end
