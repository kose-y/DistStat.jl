using NPZ, LinearAlgebra, SparseArrays, DistStat, MPI

mutable struct PETUpdate_l1
    maxiter::Int
    step::Int
    verbose::Bool
    tol::Real
    function PETUpdate_l1(;maxiter::Int=100, step::Int=10, verbose::Bool=false, tol::Real=1e-10)
        maxiter > 0 || throw(ArgumentError("maxiter must be greater than 0."))
        tol > 0 || throw(ArgumentError("tol must be positive."))
        new(maxiter, step, verbose, tol)
    end
end

mutable struct PETVariables_l1{T,A}
    m::Int # num of detector pairs
    n::Int # num of pixels
    y::A # count vector
    E::MPIMatrix{T,A} # "E" matrix, m x [n].
    rho::Real # l1 penalty
    eps::Real # for numerical stability on division
    tau::Real
    sigma::Real
    D::AbstractSparseMatrix # difference matrix, d x n.
    lambda::MPIMatrix{T,A} # length-n.
    lambda_prev::MPIMatrix{T,A} # length-n.
    z::A # dual variable, length-m.
    w::A # dual variable, length-d.
    Et1::MPIMatrix{T,A}
    tmp_n::MPIVector{T,A}
    tmp_1n_1::MPIMatrix{T,A}
    tmp_1n_2::MPIMatrix{T,A}
    tmp_m_local::A
    tmp_d_local::A
    eval_obj::Bool
    function PETVariables_l1(y::AbstractArray{T}, E::MPIMatrix{T,A},
                                D::AbstractSparseMatrix,
                                rho::Real; tau=1/3, sigma=1/3, eval_obj=false, eps::Real=1e-16) where {T,A}
        m, n = size(E)
        d = size(D, 1)

        tmp_n = MPIVector{T,A}(undef, n)
        tmp_1n_1 = MPIMatrix{T,A}(undef, 1, n)
        fill!(tmp_1n_1, one(T))
        tmp_1n_2 = MPIMatrix{T,A}(undef, 1, n)
        tmp_m_local = A{T}(undef, m)
        tmp_d_local = A{T}(undef, d)

        z = A{T}(undef, m)
        fill!(z, -one(T))
        w = A{T}(undef, d)
        fill!(w, zero(T))

        Et1 = MPIMatrix{T,A}(undef, 1, n)
        mul!(tmp_n, transpose(E), -z)
        Et1.localarray .= transpose(tmp_n.localarray)
        # undefined in CuArrays: mul!(::Transpose{Float64,CuArray{Float64,2}}, ::Transpose{Float64,CuArray{Float64,2}}, ::CuArray{Float64,1})



        lambda = MPIMatrix{T,A}(undef, 1, n)
        fill!(lambda, one(T))
        lambda_prev = MPIMatrix{T,A}(undef, 1, n)
        fill!(lambda_prev, one(T))

        new{T,A}(m, n, y, E, rho, eps, tau, sigma, D, lambda, lambda_prev,
            z, w, Et1, tmp_n, tmp_1n_1, tmp_1n_2, tmp_m_local, tmp_d_local, eval_obj)
    end
end

function reset!(v::PETVariables_l1{T, A}) where {T, A<:AbstractArray}
    fill!(v.lambda, one(T))
    fill!(v.lambda_prev, one(T))
    fill!(v.z, -one(T))
    fill!(v.w, zero(T))
end


function update!(u::PETUpdate_l1, v::PETVariables_l1{T,A}) where {T,A<:AbstractArray}
    mul!(v.tmp_n, transpose(v.E), v.z) # Etz
    v.tmp_1n_1.localarray .= transpose(v.tmp_n.localarray)
    mul!(transpose(v.tmp_1n_2), transpose(v.D), v.w) # Dtw

    # update lambda
    v.lambda .= max.(v.lambda .- v.tau .* (v.tmp_1n_1 .+ v.tmp_1n_2 .+ v.Et1), zero(T))
    v.tmp_1n_1 .= 2v.lambda .- v.lambda_prev # lambda_tilde

    # update z
    mul!(v.tmp_m_local, v.E, transpose(v.tmp_1n_1)) # el
    v.z .+= v.sigma .* v.tmp_m_local
    v.tmp_m_local .= sqrt.(v.z.^2 .+ 4v.sigma .* v.y)
    v.z .= (v.z .- v.tmp_m_local)/2one(T)

    # update w
    mul!(v.tmp_d_local, v.D, transpose(v.tmp_1n_1)) # dl
    v.w .= min.(max.(v.w .+ v.sigma .* v.tmp_d_local, -v.rho), v.rho)
end

function get_objective!(::Nothing, u::PETUpdate_l1, v::PETVariables_l1)
    if v.eval_obj
        el = mul!(v.tmp_m_local, v.E, transpose(v.lambda))
        v.tmp_m_local .= v.y .* log.(el .+ v.eps) .- el
        likelihood = sum(v.tmp_m_local)
        mul!(v.tmp_d_local, v.D, transpose(v.lambda))
        v.tmp_d_local .= abs.(v.tmp_d_local)
        penalty = - v.rho * sum(v.tmp_d_local)
        return false, likelihood + penalty
    else
        v.tmp_1n_1 .= abs.(v.lambda_prev .- v.lambda)
        return false, maximum(v.tmp_1n_1)
    end
end

function pet_l1_one_iter!(::Nothing, u::PETUpdate_l1, v::PETVariables_l1{T,A}) where {T, A<:AbstractArray}
    copyto!(v.lambda_prev, v.lambda)
    update!(u, v)
end

function loop!(Y, u, iterfun, evalfun, args...)
    converged = false
    t = 0
    result = nothing
    while !converged && t < u.maxiter
        t += 1
        iterfun(Y, u, args...)
        if t % u.step == 0
            converged, result = evalfun(Y, u, args...)
        end
    end
    return result
end

function pet_l1!(u::PETUpdate_l1, v::PETVariables_l1{T,A}) where {T,A<:AbstractArray}
    loop!(nothing, u, pet_l1_one_iter!, get_objective!, v)
end
