using DistStat, Random, LinearAlgebra


mutable struct MDSUpdate
    maxiter::Int
    step::Int
    verbose::Bool
    tol::Real
    function MDSUpdate(; maxiter::Int=100, step::Int=10, verbose::Bool=false, tol::Real=1e-10)
        maxiter > 0 || throw(ArgumentError("maxiter must be greater than 0."))
        tol > 0 || throw(ArgumentError("tol must be positive."))
        new(maxiter, step, verbose, tol)
    end
end

mutable struct MDSVariables{T, A}
    # MDS variables corresponding to distance matrix of  n x [n], X is external. m: original dimension, n: number of subjects
    # m::Int # rows, original dimension
    n::Int # cols, number of subjects
    r::Int # dimension of new space
    X::MPIMatrix{T,A} # n x [n], weighted distance
    θ::MPIMatrix{T,A} # r x [n]
    θ_prev::MPIMatrix{T,A} # r x [n]
    W::Union{T,MPIMatrix{T,A}} # weights, a singleton or an n x [n] MPIMatrix (a singleton is used for large-scale experiments)
    W_sums::Union{T,A} # sum of weights (by sample), a singleton or an MPIMatrix of [n] x 1.
    tmp_rn::MPIMatrix{T,A}
    tmp_rn_local::A
    tmp_nn::MPIMatrix{T,A}
    tmp_1n::MPIMatrix{T,A}
    tmp_n_local::A
    eval_obj::Bool
    function MDSVariables(Y::MPIMatrix{T,A}, r::Int, W=one(T); eval_obj=false, seed=nothing) where {T,A}
        n    = size(Y, 1)
        @assert m == n

        X = MPIMatrix{T,A}(undef, n, n)
        X .= W .* Y

        θ      = MPIMatrix{T,A}(undef, r, n)
        θ_prev = MPIMatrix{T,A}(undef, r, n)
        rand!(θ; seed=seed, common_init=true)
        θ .= 2θ .- 1
        if typeof(W) <: MPIMatrix
            fill_diag!(zero(T), W)
            W_sums = sum(weights; dims=2) # length-n vector
        elseif typeof(W) == T
            W_sums = convert(T, W * (n - 1))
        else
            error("type of W mismatch")
        end

        tmp_rn = MPIMatrix{T,A}(undef, r, n)
        tmp_rn_local = A{T}(undef, r, n)
        tmp_nn = MPIMatrix{T,A}(undef, n, n)
        tmp_1n = MPIMatrix{T,A}(undef, 1, n)
        tmp_n_local = A{T}(undef, n)
        new{T,A}(n, r, X, θ, θ_prev, W, W_sums, tmp_rn, tmp_rn_local,
            tmp_nn, tmp_1n, tmp_n_local, eval_obj)
    end
end
    
function reset!(v::MDSVariables{T,A}; seed=nothing) where {T,A}
    rand!(v.θ; seed=seed, common_init=true)
    v.θ .= 2v.θ .- 1
end

function update!(X::MPIArray, u::MDSUpdate, v::MDSVariables{T,A}) where {T,A}
    d = mul!(v.tmp_nn, transpose(v.θ), v.θ; tmp=v.tmp_rn_local)
    diag!(v.tmp_1n, d) # dist. row vector
    diag!(v.tmp_n_local, d) # local col vector

    d .= -2d .+ v.tmp_1n .+ v.tmp_n_local

    fill_diag!(d, Inf)

    v.tmp_nn .= v.X ./ d # Z
    v.tmp_1n .= sum(v.tmp_nn; dims=1) # Z sums, length-n dist. vector
    v.tmp_nn .= v.W .- v.tmp_nn # W - Z

    if typeof(v.W) == T
        fill_diag!(v.tmp_nn, zero(T))
    end

    θWmZ = mul!(v.tmp_rn, v.θ, (v.tmp_nn); tmp=v.tmp_rn_local)
    v.θ .= (v.θ .* (v.tmp_1n .+ v.W_sums) .+ θWmZ) ./ 2v.W_sums
end


function get_objective!(Y::MPIArray, u::MDSUpdate, v::MDSVariables{T,A}) where {T,A}
    if v.eval_obj
        DistStat.euclidean_distance!(v.tmp_nn, v.θ)
        return false, sum((Y .- v.tmp_nn).^2 .* v.W) / convert(T, 2.0)
    else
        v.tmp_rn .= abs.(v.θ_prev .- v.θ)
        return false, maximum(v.tmp_rn)
    end
end

function mds_one_iter!(Y::MPIArray, u::MDSUpdate, v::MDSVariables)
    copyto!(v.θ_prev, v.θ)
    update!(Y, u, v)
end

function loop!(Y::MPIArray, u, iterfun, evalfun, args...)
    converged = false
    t = 0
    while !converged && t < u.maxiter
        t += 1
        iterfun(Y, u, args...)
        if t % u.step == 0
            converged, monitor = evalfun(Y, u, args...)
            if DistStat.Rank() == 0
                println(t, ' ', monitor)
            end
        end
    end
end

function mds!(Y::MPIArray, u::MDSUpdate, v::MDSVariables)
    loop!(Y, u, mds_one_iter!, get_objective!, v)
end


include("cmdline.jl")
opts = parse_commandline_mds()
if DistStat.Rank() == 0
    println("world size: ", DistStat.Size())
    println(opts)
end

m = opts["rows"]
n = opts["cols"]
r = opts["r"]
iter = opts["iter"]
interval = opts["step"]
T = Float64
A = Array
if opts["gpu"]
    using CuArrays
    A = CuArray
end
if opts["Float32"]
    T = Float32
end
init_opt = opts["init_from_master"]
seed = opts["seed"]
eval_obj = opts["eval_obj"]

X = MPIMatrix{T, A}(undef, m, n)
rand!(X; common_init=init_opt, seed=0)
Y = MPIMatrix{T,A}(undef, n,n)
DistStat.euclidean_distance!(Y, X)
uquick = MDSUpdate(;maxiter=2, step=1, verbose=true)
u = MDSUpdate(;maxiter=iter, step=interval, verbose=true)
v = MDSVariables(Y, r; eval_obj=eval_obj, seed=seed)
mds!(Y, uquick, v)
reset!(v; seed=seed)
if DistStat.Rank() == 0
    @time mds!(Y, u, v)
else
    mds!(Y, u, v)
end
