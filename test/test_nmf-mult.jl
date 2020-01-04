using DistStat, Random, LinearAlgebra, Test

if get(ENV,"JULIA_MPI_TEST_ARRAYTYPE","") == "CuArray"
    using CuArrays
    A = CuArray
else
    A = Array
end

type=[Float64,Float32]

include("nmf-mult_fun.jl")
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

function nmf!(X::MPIArray, u::MultUpdate, v::NMFVariables)
    loop!(X, u, nmf_mult_one_iter!, nmf_get_objective!, v)
end

variables=parse_commandline_nmf()
m = 10
n = 10
r = 4
interval = variables["step"]
init_opt = variables["init_from_master"]
seed = variables["seed"]
eval_obj = variables["eval_obj"]

for T in type
    X=MPIMatrix{T,A}(undef,m,n)
    rand!(X; common_init=init_opt, seed=0)

    iter1=20000
    iter2=21000

    u1 = MultUpdate(;maxiter=iter1, step=interval, verbose=true)
    u2 = MultUpdate(;maxiter=iter2, step=interval, verbose=true)
    v = NMFVariables(X, r; eval_obj=eval_obj, seed=seed)

    reset!(v; seed=seed)

    result1=nmf!(X,u1,v)
    result2=nmf!(X,u2,v)
    tol=1e-05

    @test abs(result1[1]-result2[1])<tol
    @test abs(result1[2]-result2[2])<tol

end
