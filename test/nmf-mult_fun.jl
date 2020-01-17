using DistStat, Random, LinearAlgebra

mutable struct MultUpdate
    maxiter::Int
    step::Int
    verbose::Bool
    tol::Real
    lambda::Real
    function MultUpdate(; maxiter::Int=100, step::Int=10, verbose::Bool=false, tol::Real=1e-10, lambda::Real=2e-38)
        maxiter > 0 || throw(ArgumentError("maxiter must be greater than 0."))
        tol > 0 || throw(ArgumentError("tol must be positive."))
        lambda >= 0 || throw(ArgumentError("lambda must be nonnegative."))
        new(maxiter, step, verbose, tol, lambda)
    end
end

mutable struct NMFVariables{T, A}
    # NMF variables corresponding to X::MPIMatrix{T,A} # m x [n], X is external.
    m::Int # rows
    n::Int # cols
    r::Int # intermediate
    Vt::MPIMatrix{T,A} # r x [m]
    Vt_prev::MPIMatrix{T,A} # r x [m]
    W::MPIMatrix{T,A} # r x [n]
    W_prev::MPIMatrix{T,A} # r x [n]
    tmp_rr::A # broadcasted
    tmp_rm1::MPIMatrix{T,A}
    tmp_rm2::MPIMatrix{T,A}
    tmp_rn1::MPIMatrix{T,A}
    tmp_rn2::MPIMatrix{T,A}
    tmp_mn::Union{MPIMatrix, Nothing}
    tmp_rm_local::A
    function NMFVariables(X::MPIMatrix{T,A}, r; eval_obj=false, seed=nothing) where {T,A}
        m, n    = size(X)
        Vt      = MPIMatrix{T,A}(undef, r, m)
        Vt_prev = MPIMatrix{T,A}(undef, r, m)
        W       = MPIMatrix{T,A}(undef, r, n)
        W_prev  = MPIMatrix{T,A}(undef, r, n)
        rand!(Vt; seed=seed, common_init=true)
        if seed ≠ nothing
            seed += 1
        end
        rand!(W; seed=seed, common_init=true)
        tmp_rr  = A{T}(undef, r, r)
        tmp_rm1 = MPIMatrix{T,A}(undef, r, m)
        tmp_rm2 = MPIMatrix{T,A}(undef, r, m)
        tmp_rn1 = MPIMatrix{T,A}(undef, r, n)
        tmp_rn2 = MPIMatrix{T,A}(undef, r, n)
        if eval_obj
            tmp_mn = MPIMatrix{T,A}(undef, m, n)
        else
            tmp_mn = nothing
        end
        tmp_rm_local = A{T}(undef, r, m)
        new{T,A}(m, n, r, Vt, Vt_prev, W, W_prev, tmp_rr, tmp_rm1, tmp_rm2, tmp_rn1, tmp_rn2, tmp_mn, tmp_rm_local)
    end
end

function reset!(v::NMFVariables{T,A}; seed=nothing) where {T,A}
    rand!(v.Vt; seed=seed, common_init=true)
    if seed ≠ nothing
        seed += 1
    end
    rand!(v.W; seed=seed, common_init=true)
end

function update_V!(X::MPIArray, u::MultUpdate, v::NMFVariables)
    WXt = mul!(v.tmp_rm1, v.W, transpose(X); tmp=v.tmp_rm_local)
    WWt = mul!(v.tmp_rr, v.W, transpose(v.W))
    WWtVt = mul!(v.tmp_rm2, WWt, v.Vt)
    v.Vt .= v.Vt .* WXt ./ (WWtVt .+ u.lambda)
end

function update_W!(X::MPIArray, u::MultUpdate, v::NMFVariables)
    VtX = mul!(v.tmp_rn1, v.Vt, X; tmp=v.tmp_rm_local)
    VtV = mul!(v.tmp_rr, v.Vt, transpose(v.Vt))
    VtVW = mul!(v.tmp_rn2, VtV, v.W)
    v.W .= v.W .* VtX ./ (VtVW .+ u.lambda)
end

function nmf_get_objective!(X::MPIArray, u::MultUpdate, v::NMFVariables{T,A}) where {T,A}
    if v.tmp_mn != nothing
        mul!(v.tmp_mn, transpose(v.Vt), v.W; tmp=v.tmp_rm_local) # TODO: improve: print amount of update, etc.
        v.tmp_mn .= (v.tmp_mn .- X).^ 2
        return false, (sum(v.tmp_mn))
    else
        v.Vt_prev .= abs.(v.Vt_prev .- v.Vt)
        v.W_prev  .= abs.(v.W_prev  .- v.W)
        return false, (maximum(v.Vt_prev), maximum(v.W_prev))
    end
end

function nmf_mult_one_iter!(X::MPIArray, u::MultUpdate, v::NMFVariables)
    copyto!(v.W_prev, v.W)
    copyto!(v.Vt_prev, v.Vt)
    update_V!(X, u, v)
    update_W!(X, u, v)
end

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
