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

"""
    breslow_ind(x)
    
returns indexes of result of cumulutive sum corresponding to "W". `x` is assumed to be nonincreasing.
"""
function breslow_ind(x::AbstractVector)
    uniq = unique(x)
    lastinds = findlast.(isequal.(uniq), [x])
    invinds = findfirst.(isequal.(x), [uniq])
    lastinds[invinds]
end

mutable struct COXVariables{T, A}
    m::Int # rows, number of subjects
    n::Int # cols, number of predictors
    β::MPIVector{T,A} # [n]
    β_prev::MPIVector{T,A} # r x [n]
    δ::A # indicator for right censoring (0 if censored)
    λ::T # regularization parameter
    t::AbstractVector{T} # a vector containing timestamps, should be in descending order. 
    breslow::A
    σ::T # step size. 1/(2 * opnorm(X)^2) for guaranteed convergence
    grad::MPIVector{T,A}
    w::A
    W::A
    W_dist::MPIMatrix{T,A}
    q::A # (1-π)δ, distributed
    eval_obj::Bool
    function COXVariables(X::MPIMatrix{T,A}, δ::A, λ::AbstractFloat,
                        t::A;
                        σ::T=convert(T, 1/(2*opnorm(X; verbose=true)^2)), 
                        eval_obj=false) where {T,A} 
        m, n = size(X)
        β = MPIVector{T,A}(undef, n)
        β_prev = MPIVector{T,A}(undef, n)
        fill!(β, zero(T))
        fill!(β_prev, zero(T))

        δ = convert(A{T}, δ)
        breslow = convert(A{Int}, breslow_ind(convert(Array, t)))

        grad = MPIVector{T,A}(undef, n)
        w = A{T}(undef, m)
        W = A{T}(undef, m)
        W_dist = MPIMatrix{T,A}(undef, 1, m)
        q = A{T}(undef, m)

        new{T,A}(m, n, β, β_prev, δ, λ, t, breslow, σ, grad, w, W, W_dist, q, eval_obj)
    end
end
    
function reset!(v::COXVariables{T,A}; seed=nothing) where {T,A}
    fill!(v.β, zero(T))
    fill!(v.β_prev, zero(T))
end

function soft_threshold(x::T, λ::T) ::T where T <: AbstractFloat
    @assert λ >= 0 "Argument λ must be greater than or equal to zero."
    x > λ && return (x - λ)
    x < -λ && return (x + λ)
    return zero(T)
end

function π_δ!(out, w, W_dist, δ, breslow, W_range)
    # fill `out` with zeros beforehand. 
    m = length(δ)
    W_base = minimum(W_range) - 1
    Threads.@threads for i in 1:m
        for j in W_range
            @inbounds if breslow[i] <= breslow[j] 
                out[i] +=  δ[j] * w[i]/ W_dist.localarray[j - W_base]
            end
        end
    end
    DistStat.Allreduce!(out)
    return out
end




function get_breslow!(out, cumsum_w, bind)
    out .= cumsum_w[bind]
    out
end



function cox_grad!(out, w, W, W_dist, t, q, X, β, δ, bind)
    T = eltype(β)
    m, n = size(X)
    mul!(w, X, β)
    w .= exp.(w) 
    cumsum!(q, w) # q is used as a dummy variable
    get_breslow!(W, q, bind)
    W_dist .= distribute(reshape(W, 1, :))
    fill!(q, zero(eltype(q)))
    π_δ!(q, w, W_dist, δ, bind, W_dist.partitioning[DistStat.Rank()+1][2])
    q .= δ .- q
    mul!(out, transpose(X), q) # ([n]) = (n x [m]) x (m)
    out
end

cox_grad!(v::COXVariables{T,A}, X) where {T,A} = cox_grad!(v.grad, v.w, v.W, v.W_dist, v.t, v.q, X, v.β, v.δ, v.breslow)

function update!(X::MPIArray, u::COXUpdate, v::COXVariables{T,A}) where {T,A}
    #mul!(v.tmp_m_local1, X, v.β) # {m} = {m x [n]} * {[n]}.
    #v.tmp_m_local1 .= exp.(v.tmp_m_local1) # w
    #cumsum!(v.tmp_m_local2, v.tmp_m_local1) # W. TODO: deal with ties.
    #v.tmp_1m .= distribute(reshape(v.tmp_m_local2, 1, :)) # W_dist: distribute W.
    #v.tmp_mm .= v.π_ind .* v.tmp_m_local1 ./ v.tmp_1m # (π_ind .* w) ./ W_dist. computation order is determined for CuArray safety. 
    #pd = mul!(v.tmp_m_local1, v.tmp_mm, v.δ) # {m} = {m x [m]} * {m}.
    #v.tmp_m_local2 .= v.δ .- pd # {m}. 
    #grad = mul!(v.tmp_n, transpose(X), v.tmp_m_local2) # {[n]} = {[n] x m} * {m}.
    cox_grad!(v, X)
    v.β .= soft_threshold.(v.β .+ v.σ .* v.grad , v.λ) # {[n]}.
end

function get_objective!(X::MPIArray, u::COXUpdate, v::COXVariables{T,A}) where {T,A}
    v.grad .= (v.β .!= 0) # grad is dummy
    nnz = sum(v.grad)

    if v.eval_obj
        v.w .= exp.(mul!(v.w, X, v.β))
        cumsum!(v.q, v.w) # q is dummy
        get_breslow!(v.W, v.q, v.breslow)
        obj = dot(v.δ, mul!(v.q, X, v.β) .- log.(v.W)) .- v.λ .* sum(abs.(v.β))
        return false, (obj, nnz)
    else
        v.grad .= abs.(v.β_prev .- v.β)
        return false, (maximum(v.grad), nnz)
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
using CuArrays, CUDAnative
if opts["gpu"]
    A = CuArray

    # All the GPU-related functions here
    function π_δ_kernel!(out, w, W_dist, δ, breslow, W_range)
        # fill `out` with zeros beforehand.
        idx_x = (blockIdx().x-1) * blockDim().x + threadIdx().x
        stride_x = blockDim().x * gridDim().x
        W_base = minimum(W_range) - 1
        for i in idx_x:stride_x:length(out)
            for j in W_range
                @inbounds if breslow[i] <= breslow[j]
                    out[i] += δ[j] * w[i] / W_dist[j - W_base]
                end
            end
        end  
    end

    function π_δ!(out::CuArray, w::CuArray, W_dist, δ, breslow, W_range)
        numblocks = ceil(Int, length(w)/256)
        CuArrays.@sync begin
            @cuda threads=256 blocks=numblocks π_δ_kernel!(out, w, W_dist.localarray, δ, breslow, W_range)
        end
        DistStat.Allreduce!(out)
        out
    end

    function breslow_kernel!(out, cumsum_w, bind)
        idx_x = (blockIdx().x-1) * blockDim().x + threadIdx().x
        stride_x = blockDim().x * gridDim().x
        for i = idx_x: stride_x:length(out)
            out[i]=cumsum_w[bind[i]]
        end
    end

    function get_breslow!(out::CuArray, cumsum_w::CuArray, bind)
        numblocks = ceil(Int, length(out)/256)
        CuArrays.@sync begin
            @cuda threads=256 blocks=numblocks breslow_kernel!(out, cumsum_w, bind)
        end
        out
    end
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
