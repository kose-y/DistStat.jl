using NPZ, LinearAlgebra, SparseArrays, DistStat

mutable struct PETUpdate_l2
    maxiter::Int
    step::Int
    verbose::Bool
    tol::Real
    function PETUpdate_l2(;maxiter::Int=100, step::Int=10, verbose::Bool=false, tol::Real=1e-10)
        maxiter > 0 || throw(ArgumentError("maxiter must be greater than 0."))
        tol > 0 || throw(ArgumentError("tol must be positive."))
        new(maxiter, step, verbose, tol)
    end
end

mutable struct PETVariables_l2{T,A}
    m::Int # num of detector pairs 
    n::Int # num of pixels
    y::A # count vector
    E::AbstractSparseMatrix # "E" matrix, m x n. 
    E_partial::AbstractSparseMatrix # a subset of "E" matrix. 
    # This will be considered as an m x [n] matrix with only elementwise operation.
    # NOTE: Or...should we implement MPIMatrix{T, <:AbstractSparseMatrix}?

    mu::Real # l2 penalty
    eps::Real # for numerical stability on division
    G::AbstractSparseMatrix # adjacency matrix, n x n.
    D::Union{Nothing, AbstractSparseMatrix} # difference matrix, d x n. 
    N::MPIVector # length-n.
    a::MPIVector{T,A} # length-n. 
    lambda::MPIVector{T,A} # length-n.
    lambda_prev::MPIVector{T,A} # length-n.
    tmp_n1::MPIVector{T,A}
    tmp_n2::MPIVector{T,A}
    tmp_m_local::A
    tmp_d::Union{Nothing, MPIVector{T,A}}
    eval_obj::Bool
    function PETVariables_l2{A}(y::AbstractArray{T}, E::AbstractSparseMatrix, 
                                E_partial::AbstractSparseMatrix,
                                G::AbstractSparseMatrix, D::AbstractSparseMatrix, 
                                mu::Real; eval_obj=false, eps::Real=1e-20) where {T,A}
        m, n = size(E)

        tmp_n1 = MPIVector{T,A}(undef, n)
        fill!(tmp_n1, one(T))
        tmp_n2 = MPIVector{T,A}(undef, n)
        tmp_m_local = A{T}(undef, m)

        #E_partial = E[:, tmp_n1.partitioning[DistStat.Rank()+1][1]]

        # compute |N_j| = G * 1
        N = MPIVector{T,A}(undef, n)
        mul!(N, G, tmp_n1)
        a = MPIVector{T,A}(undef, n)
        a .= -2mu .* N
        lambda = MPIVector{T,A}(undef, n)
        fill!(lambda, one(T))
        lambda_prev = MPIVector{T,A}(undef, n)
        fill!(lambda_prev, one(T))
        if eval_obj
            tmp_d = MPIVector{T,A}
        else
            tmp_d = nothing
            D = nothing
        end

        new{T,A}(m, n, y, E, E_partial, mu, eps, G, D, N, a, lambda, lambda_prev, 
            tmp_n1, tmp_n2, tmp_m_local)
    end
end

function reset!(v::PETVariables_l2)
    fill!(v.lambda, one(T))
    fill!(v.lambda_prev, one(T))
end

function update!(u::PETUpdate_l2, v::PETVariables_l2)
    el = mul!(v.tmp_m_local, v.E, v.lambda)
    gl = mul!(v.tmp_n1, v.G, v.lambda)
    z = transpose(v.E_partial) .* transpose(v.y) .* v.lambda.localarray ./ (transpose(el) .+ v.eps) 
    # We should consider z as a temporary [n] x m SparseArray. 

    v.tmp_n1 .= v.mu .* (v.N .* v.lambda .+ gl) .- one(T) # b.
    v.tmp_n2.localarray .= reshape(sum(z; dims=2), :) # c.
    if v.mu != 0
        v.lambda .= (-v.tmp_n1 .- (v.tmp_n1.^2 - 4 .* v.a .* v.tmp_n2)) ./ (2v.a .+ v.eps)
    else
        v.lambda .= -v.tmp_n2 ./ (v.tmp_n1 .+ v.eps)
    end
end

function get_objective!(::Nothing, u::PETUpdate_l2, v::PETVariables_l2)
    if v.eval_obj
        el = mul!(tmp_m_local, v.E, v.lambda)
        v.tmp_m_local .= v.y .* log.(el .+ v.eps) .- el
        likelihood = sum(tmp_m_local)
        mul!(v.tmp_d, v.D, v.lambda)
        v.tmp_d .^= 2
        penalty = - v.mu / 2.0 * sum(v.tmp_d)
        return false, likelihood + penalty
    else
        v.tmp_n1 .= abs.(v.lambda_prev .- v.lambda)
        return false, maximum(v.tmp_n1)
    end
end

function pet_l2_one_iter!(::Nothing, u::PETUpdate_l2, v::PETVariables_l2)
    copyto!(v.lambda_prev, v.lambda)
    update!(u, v)
end

function loop!(Y, u, iterfun, evalfun, args...)
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

function pet_l2!(u::PETUpdate_l2, v::PETVariables_l2)
    loop!(nothing, u, pet_l2_one_iter!, get_objective!, v)
end

include("cmdline.jl")
opts = parse_commandline_pet()
if DistStat.Rank() == 0
    println("world size: ", DistStat.Size())
    println(opts)
end

iter = opts["iter"]
interval = opts["step"]
T = Float64
A = Array
SA = SparseMatrixCSC
if opts["gpu"]
    using CuArrays
    using CuArrays.CUSPARSE
    A = CuArray
    SA = CuSparseMatrixCSR
end
if opts["Float32"]
    T = Float32
end



eval_obj = opts["eval_obj"]

datafile = opts["data"]
mu = opts["mu"]

dat = npzread(datafile)

n_x = dat["n_x"]
n_t = dat["n_t"]
e_indices = dat["e_indices"] .+ 1 # 2 x nnz
e_values = convert(Array{T}, dat["e_values"]) # length-nnz

m = n_t * (n_t - 1) ÷ 2
n = n_x ^ 2

D_indices = dat["D_indices"] .+ 1 # 2 x nnz
D_values  = convert(Array{T}, dat["D_values"]) # length-nnz
D_shape   = dat["D_shape"]

G_indices = dat["G_indices"] .+ 1 # 2 x nnz
G_values  = convert(Array{T}, dat["G_values"]) # length-nnz
G_shape   = dat["G_shape"]

y = dat["counts"] # length-n

E = sparse(e_indices[1,:], e_indices[2,:], e_values, m, n)
G = sparse(G_indices[1,:], G_indices[2,:], G_values, G_shape[1], G_shape[2])
D = sparse(D_indices[1,:], D_indices[2,:], D_values, D_shape[1], D_shape[2])
t = MPIVector{T,A}(undef, n)
E_partial = E[:, t.partitioning[DistStat.Rank()+1][1]]

y, E, E_partial, G, D = convert(A{T}, y), SA(E), SA(E_partial), SA(G), SA(D)


uquick = PETUpdate_l2(;maxiter=2, step=1, verbose=true)
u = PETUpdate_l2(;maxiter=iter, step=interval, verbose=true)
v = PETVariables_l2{A}(y, E, E_partial, G, D, mu; eval_obj=eval_obj)
pet_l2!(uquick, v)
reset!(v)
if DistStat.Rank() == 0
    @time pet_l2!(u, v)
else
    pet_l2!(u, v)
end