using DistStat, Random, LinearAlgebra


mutable struct COXUpdate
    maxiter::Int
    step::Int
    verbose::Bool
    tol::Real
    function COXUpdate(; maxiter::Int=100, step::Int=10, verbose::Bool=false, tol::Real=1e-6)
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
    # COX variables corresponding to an m × [n] data matrix X (external). m: number of subjects, n: number of predictors
    m::Int # rows, number of subjects
    n::Int # cols, number of predictors
    β::MPIVector{T,A} # [n]
    β_prev::MPIVector{T,A} # r x [n]
    δ::A # indicator for right censoring (0 if censored)
    λ::MPIVector{T,A} # regularization parameter
    t::AbstractVector{T} # a vector containing timestamps, should be in descending order. 
    breslow::A
    σ::T # step size. 1/(2 * opnorm(X)^2) for guaranteed convergence
    π_ind::MPIMatrix{T,A}
    tmp_n::MPIVector{T,A}
    tmp_1m::MPIMatrix{T,A}
    tmp_mm::MPIMatrix{T,A}
    tmp_m_local1::A
    tmp_m_local2::A
    eval_obj::Bool
    obj_prev::T
    function COXVariables(X::MPIMatrix{T,A}, δ::A, λ::MPIVector{T,A},
                            t::A;
                            σ::T=convert(T, 1/(2*opnorm(X; verbose=true)^2)), 
                            eval_obj=false) where {T,A}
        m, n = size(X)
        β = MPIVector{T,A}(undef, n)
        β_prev = MPIVector{T,A}(undef, n)
        fill!(β, zero(T))
        fill!(β_prev, zero(T))

        δ = convert(Vector{T}, δ)
        breslow = convert(A{Int}, breslow_ind(convert(Array, t)))

        π_ind = MPIMatrix{T,A}(undef, m, m)
        t_dist = distribute(reshape(t, 1, :))
        fill!(π_ind, one(T))
        π_ind .= ((π_ind .* t_dist) .- t) .<= 0

        tmp_n = MPIVector{T,A}(undef, n)
        tmp_1m = MPIMatrix{T,A}(undef, 1, m)
        tmp_mm = MPIMatrix{T,A}(undef, m, m)
        tmp_m_local1 = A{T}(undef, m)
        tmp_m_local2 = A{T}(undef, m)

        obj_prev = -Inf

        new{T,A}(m, n, β, β_prev, δ, λ, t, breslow, σ, π_ind, tmp_n, tmp_1m, tmp_mm, tmp_m_local1, tmp_m_local2, eval_obj, obj_prev)
    end
end
    
function reset!(v::COXVariables{T,A}; seed=nothing) where {T,A}
    fill!(v.β, zero(T))
    fill!(v.β_prev, zero(T))
    v.obj_prev = -Inf
end

function soft_threshold(x::T, λ::T) ::T where T <: AbstractFloat
    x > λ && return (x - λ)
    x < -λ && return (x + λ)
    return zero(T)
end

function get_breslow!(out, cumsum_w, bind)
    out .= cumsum_w[bind]
    out
end

function update!(X::MPIArray, u::COXUpdate, v::COXVariables{T,A}) where {T,A}
    mul!(v.tmp_m_local1, X, v.β) # {m} = {m x [n]} * {[n]}.
    v.tmp_m_local1 .= exp.(v.tmp_m_local1) # w
    cumsum!(v.tmp_m_local2, v.tmp_m_local1) # W. TODO: deal with ties.
    get_breslow!(v.tmp_m_local2, v.tmp_m_local2, v.breslow)
    v.tmp_1m .= distribute(reshape(v.tmp_m_local2, 1, :)) # W_dist: distribute W.
    v.tmp_mm .= v.π_ind .* v.tmp_m_local1 ./ v.tmp_1m # (π_ind .* w) ./ W_dist. computation order is determined for CuArray safety. 
    pd = mul!(v.tmp_m_local1, v.tmp_mm, v.δ) # {m} = {m x [m]} * {m}.
    v.tmp_m_local2 .= v.δ .- pd # {m}. 
    grad = mul!(v.tmp_n, transpose(X), v.tmp_m_local2) # {[n]} = {[n] x m} * {m}. 
    v.β .= soft_threshold.(v.β .+ v.σ .* grad , v.λ) # {[n]}.
end


function get_objective!(X::MPIArray, u::COXUpdate, v::COXVariables{T,A}) where {T,A}
    v.tmp_n .= (v.β .!= 0)
    nonzeros = sum(v.tmp_n)
    if v.eval_obj
        v.tmp_m_local1 .= exp.(mul!(v.tmp_m_local1, X, v.β))
        cumsum!(v.tmp_m_local1, v.tmp_m_local1)
        get_breslow!(v.tmp_m_local1, v.tmp_m_local1, v.breslow)
        obj = dot(v.δ, mul!(v.tmp_m_local2, X, v.β) .- log.(v.tmp_m_local1)) .- sum(abs.(v.λ .* v.β))
        reldiff = abs(obj - v.obj_prev) / (abs(obj) * u.step)
        conv = reldiff < u.tol
        v.obj_prev = obj
        return conv, (obj, reldiff, nonzeros)
    else
        v.tmp_n .= abs.(v.β_prev .- v.β)
        return false, (maximum(v.tmp_n), nonzeros)
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
opts = parse_commandline_cox_ukbk()
if DistStat.Rank() == 0
    println("world size: ", DistStat.Size())
    println(opts)
end

iter = opts["iter"]
interval = opts["step"]
prefix = opts["prefix"]
T = Float64
A = Array
using CuArrays, CUDAnative
if opts["gpu"]
    A = CuArray

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
eval_obj = opts["eval_obj"]

using SnpArrays, CSV

pheno = CSV.read("/home/kose/project/dist-stat-julia/ukb_short.filtered.tab"; delim="\t")
δ = convert(A{T}, pheno[1:end, :T2D])
t = convert(A{T}, pheno[1:end, :AgeAtOnset])
print(size(δ), size(t))
dat = SnpData("/home/kose/project/dist-stat-julia/filtered_merged")

m, n = size(dat.snparray)

n += 11 # 11 unpenalized variables

lambda = MPIVector{T,A}(undef, n)
fill!(lambda, opts["lambda"])
if DistStat.Rank() == DistStat.Size() - 1
    lambda.localarray[end-10:end] .= zero(T) # 11 unpenalized variables
end

X = MPIMatrix{T,A}(undef, m, n)

# copyto! X
idx1, idx2 = X.partitioning[DistStat.Rank() + 1]
if DistStat.Rank() == DistStat.Size() - 1
    tmp = A{T}(undef, m, size(X.localarray, 2) - 11)
    Base.copyto!(tmp, @view(dat.snparray[1:m, idx2[1:end-11]]); impute=true)
    X.localarray[:, 1:end-11] .= tmp
    X.localarray[:, end-10:end] .= convert(A{T}, pheno[1:m, 2:12])
    tmp = nothing
    GC.gc()
else
    Base.copyto!(X.localarray, @view(dat.snparray[1:m, idx2]); impute=true)
end


uquick = COXUpdate(;maxiter=2, step=1, verbose=true)
u = COXUpdate(;maxiter=iter, step=interval, verbose=true)
v = COXVariables(X, δ, lambda, t; eval_obj=eval_obj) 
cox!(X, uquick, v)
reset!(v; seed=seed)
if DistStat.Rank() == 0
    @time cox!(X, u, v)
else
    cox!(X, u, v)
end

nonzero_idxs = v.β.partitioning[DistStat.Rank()+1][1][findall(!iszero, v.β.localarray)]
nonzero_vals = v.β.localarray[findall(!iszero, v.β.localarray)]


using Printf
filename = @sprintf("ukbk_julia_%s_%03d.txt", prefix, DistStat.Rank()+1)
open(filename, "w") do io
    for (idx, val) in zip(nonzero_idxs, nonzero_vals)
        write(io, "$idx\t$val\n")
    end
end

