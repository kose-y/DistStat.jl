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
    W_sums::Union{T,MPIMatrix{T,A}} # sum of weights (by sample), a singleton or an MPIMatrix of [n] x 1.
    tmp_rr::A # broadcasted
    tmp_rn1::MPIMatrix{T,A}
    tmp_rn2::MPIMatrix{T,A}
    tmp_nn::Union{MPIMatrix, Nothing}
    tmp_rn_local::A
    tmp_n::MPIVector{T,A}
    tmp_n_local::A
    function MDSVariables(Y::MPIMatrix{T,A}, r::Int, W=one(T); eval_obj=false, seed=nothing) where {T,A}
        m, n    = size(Y, 1)
        @assert m == n

        X = MPIMatrix{T,A}(undef, n, n)
        X .= W .* Y

        θ      = MPIMatrix{T,A}(undef, r, n)
        θ_prev = MPIMatrix{T,A}(undef, r, n)
        rand!(θ; seed=seed, common_init=true)
        θ .= 2θ - 1
        if typeof(W) <: MPIMatrix
            fill_diag!(zero(T), W)
            W_sums = sum(weights; dims=2)
        elseif typeof(W) == T
            W_sums = convert(T, W * (n - 1))
        else
            error("type of W mismatch")
        end

        tmp_rr  = A{T}(undef, r, r)
        tmp_rn1 = MPIMatrix{T,A}(undef, r, n)
        tmp_rn2 = MPIMatrix{T,A}(undef, r, n)
        if eval_obj
            tmp_nn = MPIMatrix{T,A}(undef, m, n)
        else
            tmp_nn = nothing
        end
        tmp_rn_local = A{T}(undef, r, n)
        tmp_1n = MPIMatrix{T,A}(undef, 1, n)
        tmp_n_local = A{T}(undef, n)
        new{T,A}(n, r, X, θ, θ_prev, W, W_sums, tmp_rr, tmp_rn1, tmp_rn2, tmp_nn, tmp_rn_local, tmp_1n, tmp_n_local)
    end
end
    
function reset!(v::MDSVariables{T,A}; seed=nothing) where {T,A}
    rand!(v.θ; seed=seed, common_init=true)
    v.θ .= 2v.θ - 1
end

function update!(X::MPIArray, u::APGUpdate, v::MDSVariables{T,A}) where {T,A}
    d = mul!(TODO, transpose(v.θ), v.θ)
    v.tmp_1n      .= reshape(diag(d; dist=false), 1, n) # dist. row vector
    v.tmp_n_local .= diag(d; dist=true) # local col vector
    d .*= convert(T, -2.0)
    d .+= v.tmp_1n .+ v.tmp_n_local

    v.Z .= v.X ./ d
    Z_sums .= sum(Z; dims=TODO) # length-n dist. vector
    WmZ .= v.W .- v.Z

    if typeof(self.W) == T
        fill_diag!(WmZ, zero(T))
    end

    θWmZ = mul!(TODO, v.θ, WmZ)
    v.θ .= (v.θ .* (v.W_sums .+ Z_sums) + θWmZ) ./ (convert(T, 2.0) .* v.W_sums)
end


function get_objective!(Y::MPIArray, u::MDSUpdate, v::MDSVariables{T,A}) where {T,A}
    if v.tmp_nn2 != nothing
        v.tmp_nn2 .= euclidean_disntances(v.θ)
        return false, sum(((Y .- v.tmp_nn) .* v.W) .^ 2) / convert(T, 2.0)
    else
        v.θ_prev .= abs.(v.θ_prev .- v.θ)
        return false, maximum(v.θ_prev)
    end
end

function mds_one_iter!(Y::MPIArray, u::APGUpdate, v::NMFVariables)
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

function mds!(Y::MPIArray, u::APGUpdate, v::NMFVariables)
    loop!(X, u, mds_one_iter!, get_objective!, v)
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
uquick = APGUpdate(;maxiter=2, step=1, verbose=true)
u = APGUpdate(;maxiter=iter, step=interval, verbose=true)
v = NMFVariables(X, r; eval_obj=eval_obj, seed=seed)
nmf!(X, uquick, v)
reset!(v; seed=seed)
if DistStat.Rank() == 0
    @time nmf!(X, u, v)
else
    nmf!(X, u, v)
end
