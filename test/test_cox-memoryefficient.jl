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

function loop!(X::MPIArray, u, iterfun, evalfun, args...)
    converged = false
    t = 0
    result=nothing
    while !converged && t < u.maxiter
        t += 1
        iterfun(X, u, args...)
        if t % u.step == 0
            converged, result = evalfun(X, u, args...)
        end
    end
    return result
end

function cox!(X::MPIArray, u::COXUpdate, v::COXVariables)
    loop!(X, u, cox_one_iter!, get_objective!, v)
end

variables = parse_commandline_cox()
m = variables["rows"]
n = variables["cols"]
interval = variables["step"]
init_opt = variables["init_from_master"]
seed = variables["seed"]
eval_obj = variables["eval_obj"]
censor_rate = variables["censor_rate"]
lambda = variables["lambda"]

for T in type
    X = MPIMatrix{T, A}(undef, m, n)
    rand!(X; common_init=init_opt, seed=seed)
    δ = convert(A{T}, rand(m) .> censor_rate)
    DistStat.Bcast!(δ)

    iter1=variables["iter"]
    iter2=variables["iter"]+100

    u1 = COXUpdate(;maxiter=iter1, step=interval, verbose=true)
    u2 = COXUpdate(;maxiter=iter2, step=interval, verbose=true)
    t = convert(A{T}, collect(reverse(1:size(X,1))))
    v = COXVariables(X, δ, lambda, t; eval_obj=eval_obj)

    reset!(v; seed=seed)

    result1=cox!(X,u1,v)
    result2=cox!(X,u2,v)
    tol1=1e-05
    tol2=1e-06

    @test result1[1]<tol1
    @test result2[1]<tol1

    @test abs(result1[1]-result2[1])<tol2

end
