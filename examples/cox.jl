using DistStat, Random, LinearAlgebra


mutable struct COXUpdate
    maxiter::Int
    step::Int
    verbose::Bool
    tol::Real
    function COXUpdate(; maxiter::Int=100, step::Int=10, verbose::Bool=false, tol::Real=1e-10)
        maxiter > 0 || throw(ArgumentError("maxiter must be greater than 0."))
        tol > 0 || throw(ArgumentError("tol must be positive."))
        new(maxiter, step, verbose, tol)
    end
end

mutable struct COXVariables{T, A}
    # COX variables corresponding to an m × [n] data matrix X (external). m: number of subjects, n: number of predictors
    m::Int # rows, number of subjects
    n::Int # cols, number of predictors
    β::MPIVector{T,A} # [n]
    β_prev::MPIVector{T,A} # r x [n]
    δ::A # indicator for right censoring (0 if censored)
    λ::T # regularization parameter
    t::AbstractVector{T} # a vector containing timestamps, should be in descending order. 
    σ::T # step size. 1/(2 * opnorm(X)^2) for guaranteed convergence
    π_ind::MPIMatrix{T,A}
    tmp_n::MPIVector{T,A}
    tmp_1m::MPIMatrix{T,A}
    tmp_mm::MPIMatrix{T,A}
    tmp_m_local1::A
    tmp_m_local2::A
    eval_obj::Bool
    function COXVariables(X::MPIMatrix{T,A}, δ::A, λ::AbstractFloat,
                            t::A;
                            σ::T=1/(2*opnorm(X; verbose=true)^2), 
                            eval_obj=false) where {T,A}
        m, n = size(X)
        β = MPIVector{T,A}(undef, n)
        β_prev = MPIVector{T,A}(undef, n)
        fill!(β, zero(T))
        fill!(β_prev, zero(T))

        δ = convert(Vector{T}, δ)

        π_ind = MPIMatrix{T,A}(undef, m, m)
        t_dist = distribute(reshape(t, 1, :))
        fill!(π_ind, one(T))
        π_ind .= ((π_ind .* t_dist) .- t) .<= 0

        tmp_n = MPIVector{T,A}(undef, n)
        tmp_1m = MPIMatrix{T,A}(undef, 1, m)
        tmp_mm = MPIMatrix{T,A}(undef, m, m)
        tmp_m_local1 = A{T}(undef, m)
        tmp_m_local2 = A{T}(undef, m)

        new{T,A}(m, n, β, β_prev, δ, λ, t, σ, π_ind, tmp_n, tmp_1m, tmp_mm, tmp_m_local1, tmp_m_local2, eval_obj)
    end
end
    
function reset!(v::COXVariables{T,A}; seed=nothing) where {T,A}
    fill!(v.β, zero(T))
    fill!(v.β_prev, zero(T))
end

function soft_threshold(x::T, λ::T) ::T where T <: AbstractFloat
    x > λ && return (x - λ)
    x < -λ && return (x + λ)
    return zero(T)
end

function update!(X::MPIArray, u::COXUpdate, v::COXVariables{T,A}) where {T,A}
    mul!(v.tmp_m_local1, X, v.β) # {m} = {m x [n]} * {[n]}.
    v.tmp_m_local1 .= exp.(v.tmp_m_local1) # w
    cumsum!(v.tmp_m_local2, v.tmp_m_local1) # W. TODO: deal with ties.
    v.tmp_1m .= distribute(reshape(v.tmp_m_local2, 1, :)) # W_dist: distribute W.
    v.tmp_mm .= v.π_ind .* v.tmp_m_local1 ./ v.tmp_1m # (π_ind .* w) ./ W_dist. computation order is determined for CuArray safety. 
    pd = mul!(v.tmp_m_local1, v.tmp_mm, v.δ) # {m} = {m x [m]} * {m}.
    v.tmp_m_local2 .= v.δ .- pd # {m}. 
    grad = mul!(v.tmp_n, transpose(X), v.tmp_m_local2) # {[n]} = {[n] x m} * {m}. 
    v.β .= soft_threshold.(v.β .+ v.σ .* grad , v.λ) # {[n]}.
end


function get_objective!(X::MPIArray, u::COXUpdate, v::COXVariables{T,A}) where {T,A}
    v.tmp_n .= (v.β .== 0)
    sparsity = sum(v.tmp_n)/size(v.tmp_n, 1)
    if v.eval_obj
        v.tmp_m_local1 .= exp.(mul!(v.tmp_m_local1, X, v.β))
        cumsum!(v.tmp_m_local1, v.tmp_m_local1)
        obj = dot(v.δ, mul!(v.tmp_m_local2, X, v.β) .- log.(v.tmp_m_local1)) .- v.λ .* sum(abs.(v.β)) # TODO: deal with ties.
        
        return false, (obj, sparsity)
    else
        v.tmp_n .= abs.(v.β_prev .- v.β)
        return false, (maximum(v.tmp_n), sparsity)
    end
end

function cox_one_iter!(X::MPIArray, u::COXUpdate, v::COXVariables)
    copyto!(v.β_prev, v.β)
    update!(X, u, v)
end

function loop!(X::MPIArray, u, iterfun, evalfun, args...)
    converged = false
    t = 0
    while !converged && t < u.maxiter
        t += 1
        iterfun(X, u, args...)
        if t % u.step == 0
            converged, monitor = evalfun(X, u, args...)
            if DistStat.Rank() == 0
                println(t, ' ', monitor)
            end
        end
    end
end

function cox!(X::MPIArray, u::COXUpdate, v::COXVariables)
    loop!(X, u, cox_one_iter!, get_objective!, v)
end


include("cmdline.jl")
opts = parse_commandline_cox()
if DistStat.Rank() == 0
    println("world size: ", DistStat.Size())
    println(opts)
end

m = opts["rows"]
n = opts["cols"]
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
censor_rate = opts["censor_rate"]
lambda = opts["lambda"]

X = MPIMatrix{T, A}(undef, m, n)
rand!(X; common_init=init_opt, seed=seed)
δ = convert(A{T}, rand(m) .> censor_rate)
DistStat.Bcast!(δ) # synchronize the choice for δ.

uquick = COXUpdate(;maxiter=2, step=1, verbose=true)
u = COXUpdate(;maxiter=iter, step=interval, verbose=true)
# for simulation run, we just assume that the data are in reversed order of survivial time
t = convert(A{T}, collect(reverse(1:size(X,1))))
v = COXVariables(X, δ, lambda, t; eval_obj=eval_obj) 
cox!(X, uquick, v)
reset!(v; seed=seed)
if DistStat.Rank() == 0
    @time cox!(X, u, v)
else
    cox!(X, u, v)
end
