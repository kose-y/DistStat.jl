import MPI
import MPI: COMM_WORLD
using Random, SparseArrays, LinearAlgebra, ArgParse, Printf, Dates, Format, Folds, NPZ
using Base.Threads

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--input", "-i"
            help = "input npy file"
            default = nothing
        "--out", "-o"
            help = "name of output file"
            default = nothing
        "--l1", "-l"
            help = "lambda penalty"
            arg_type = Float64
            default = 0.1
        "--tau", "-t"
            help = "starting step size"
            arg_type = Float64
            default = 1.0
        "--tau_min", "-m"
            help = "minimum step size"
            arg_type = Float64
            default = 0.0
        "--epsilon", "-e"
            help = "stopping criterion"
            arg_type = Float64
            default = 1e-5
        "--max_outer"
            help = "number of maximum outer iterations"
            arg_type = Int
            default = 100
        "--max_inner"
            help = "number of maximum inner iterations"
            arg_type = Int
            default = 10
        "--threads", "-c"
            help = "number of threads used per process in BLAS"
            arg_type = Int
            default = 4
        "--block_size", "-b"
            help = "size of distributed block, default uses maximum range"
            arg_type = Int
            default = 0
        "--offset", "-k"
            help = "index for the distributed block"
            arg_type = Int
            default = 0
        "--mkl"
            help = "use mkl"
            action = :store_true
    end
    return parse_args(s)
end

include("../src/distdirectives.jl")
include("../src/distarray.jl")
include("../src/splinalg.jl")
include("../src/arrayfunctions.jl")

MPI.Initialized() || MPI.Init()

@inline function Size()
    MPI.Comm_size(COMM_WORLD)
end

@inline function Rank()
    MPI.Comm_rank(COMM_WORLD)
end

mutable struct ACCORDUpdate
    out_iter::Int
    inner_iter::Int
    tau_start::Real
    tau_min::Real
    tol::Real
    function ACCORDUpdate(out_iter, inner_iter, tau, tau_min, tol)
        out_iter > 0 || throw(ArgumentError("iter must be greater than 0."))
        inner_iter > 0 || throw(ArgumentError("iter must be greater than 0."))
        tau > 0 || throw(ArgumentError("step size must be greater than 0."))
        tol > 0 || throw(ArgumentError("tolerance must be positive."))
        new(out_iter, inner_iter, tau, tau_min, tol)
    end
end

function block_distribute(start_ind::Integer, end_ind::Integer, parts::Integer)
    dist_range = []
    nb_elems = (end_ind - start_ind) + 1
    local_len = nb_elems ÷ parts
    remainder = nb_elems % parts
    nb_elems_dist = [k <= remainder ? local_len+1 : local_len for k in 1:parts]

    current_start = start_ind
    current_end = start_ind - 1
    for k in 1:parts
        current_end += nb_elems_dist[k]
        push!(dist_range, UnitRange(current_start, current_end))
        current_start += nb_elems_dist[k]
    end
    return dist_range
end

mutable struct ACCORDvariables{T}
    n::Int
    p::Int
    lambda::Real
    X::Matrix{T}
    Y::Matrix{T}
    GT::Matrix{T}
    o_tilde::Matrix{T}
    threshold::Matrix{Bool}
    OmegaT::SparseMatrixCSC{T,Int}
    OmegaT_old::SparseMatrixCSC{T,Int}
    diag_indx::Int #for diagonal coordinate
    function ACCORDvariables(X::Matrix{T}, lambda::Real, OmegaT::SparseMatrixCSC{T,Int}, start_ind::Integer) where {T} 
        lambda >= 0 || throw(ArgumentError("penalty lambda must be nonnegative."))
        n, p = size(X)
        diag_indx = 1 - start_ind

        @assert size(OmegaT, 1) == p
        OmegaT_old = deepcopy(OmegaT)
        println(size(OmegaT))

        Y = Matrix{T}(undef, n, size(OmegaT, 2))
        GT = Matrix{T}(undef, p, size(OmegaT, 2))
        o_tilde = Matrix{T}(undef, p, size(OmegaT, 2))
        threshold = Matrix{Bool}(undef, p, size(OmegaT, 2))
        new{T}(n, p, lambda, X, Y, GT, o_tilde, threshold, OmegaT, OmegaT_old, diag_indx)
    end
    function ACCORDvariables(X::Matrix{T}, lambda::Real, start_ind::Integer, end_ind::Integer) where {T}
        # start with default identity
        n, p = size(X)
        parts = block_distribute(start_ind, end_ind, Size())
        OmegaT = SparseMatrixCSC{T, Int}(I, p, p)[1:p, parts[Rank() + 1]]

        #end_ind is implicitly determined by size of OmegaT
        return ACCORDvariables(X, lambda, OmegaT, parts[Rank() + 1][1])
    end
end

# TODO need to change when using replication
function compute_g!(v::ACCORDvariables{T}) where {T}
    # compute Y = X * Omega^T 
    # and return partial computation of g (smooth part of loss function)
    # LinearAlgebra.mul!(v.Y, v.X, v.OmegaT)
    dspmm!(v.Y, v.X, v.OmegaT)
    return 0.5 * Folds.mapreduce(x -> x^2, +, v.Y) / v.n 
end

function compute_grad!(v::ACCORDvariables{T}) where {T}
    # compute G^T = X^T * Y / n, gradient for g(Omega)
    LinearAlgebra.mul!(v.GT, transpose(v.X), v.Y / v.n)
    return
end

function compute_Omega!(v::ACCORDvariables{T}, tau::Real) where {T}
    # apply proximal update and update omega
    dgrad_update!(v.o_tilde, v.OmegaT_old, v.GT, tau)
    c = tau * v.lambda
    diag_entries = Folds.map(x -> 0.5 * (x + sqrt(x^2 + 4*tau)), diag(v.o_tilde, v.diag_indx))

    # construct updated v.Omega
    Folds.map!(x -> x > c ? true : (x < -c ? true : false), v.threshold, v.o_tilde)
    v.threshold[diagind(v.threshold, v.diag_indx)] .= true
    p,k = size(v.o_tilde)
    nnzs = sum(v.threshold; dims = 1)
    nnz_count = sum(nnzs)

    colptr = vec(accumulate(+, hcat(1, nnzs); dims = 2))
    rowval = Vector{Int}(undef, nnz_count)
    nzval = Vector{T}(undef, nnz_count)

    Threads.@threads for j in axes(v.o_tilde, 2)
        col_count = 0
        for i in (1:p)[v.threshold[:,j]]
            rowval[colptr[j] + col_count] = i
            if j - i == v.diag_indx
                nzval[colptr[j] + col_count] = diag_entries[j]
            else    
                nzval[colptr[j] + col_count] = v.o_tilde[i,j] > 0 ? v.o_tilde[i,j] - c : v.o_tilde[i,j] + c
            end
            col_count += 1
        end
    end
    v.OmegaT = SparseMatrixCSC{T, Int}(p, k, colptr, rowval, nzval)
end

function compute_Q(v::ACCORDvariables{T}, tau::Real) where {T}
    # compute Q function for backtracking, also return maximum difference for stopping criterion
    D = v.OmegaT - v.OmegaT_old

    # compute D_dot_G + D_F^2/(2*tau)
    partial_Q = Folds.mapreduce(x -> x[3] * v.GT[x[1],x[2]], +, zip(findnz(D)...)) + Folds.mapreduce(x -> x^2, +, D.nzval) / (2.0 * tau)
    partial_maxdiff = Folds.mapreduce(x -> abs(x), max, D.nzval)

    return partial_Q, partial_maxdiff
end

function update!(u::ACCORDUpdate, v::ACCORDvariables{T}, g_old::Real, i_outer::Integer, start_time::DateTime) where {T}
    tau = u.tau_start
    partial_maxdiff = 0.0
    g = 0.0
    nnz_count = 0.0
    compute_grad!(v) # need to execute compute_g! before
    for i_inner in 1:u.inner_iter
        compute_Omega!(v, tau) # update OmegaT with OmegaT_old
        g = compute_g!(v)
        Q, partial_maxdiff = compute_Q(v, tau)
        nnz_count = nnz(v.OmegaT)

        # temp = [partial_g, partial_q, partial_nnz_count]
        # MPI.Allreduce!(temp, MPI.SUM, MPI.COMM_WORLD)
        Q += g_old
        nnz_ratio = nnz_count * 100.0 / (v.p ^ 2)
        
        if Rank() == 0 
            @printf("Round %03d.%02d [%10.4lf]: tau = %10.4lf, g = %10.4lf, Q = %10.4lf, %%nnz = %9.6lf, nnz = %d\n", 
                i_outer, i_inner, Dates.value(now() - start_time) * 0.001, tau, g, Q, nnz_ratio, nnz_count)
        end
        if tau <= u.tau_min || g <= Q
            break
        end
        tau /= 2.0
    end
    return partial_maxdiff, g, nnz_count
end

function accord!(u::ACCORDUpdate, v::ACCORDvariables{T}, start_time::DateTime) where {T}
    g_omega = compute_g!(v)
    omega_nnz = 0
    if Rank() == 0
        @printf("g_0 = %lf\n", g_omega)
    end
    for i_outer in 1:u.out_iter
        partial_maxdiff, g_omega, omega_nnz = update!(u, v, g_omega, i_outer, start_time)
        v.OmegaT, v.OmegaT_old = v.OmegaT_old, v.OmegaT
        if partial_maxdiff <= u.tol
            break
        end
    end
    return omega_nnz
end

if Rank() != 0
    redirect_stdout(devnull)
end

start_time = Dates.now()
opts = parse_commandline()

X = npzread(opts["input"])
output_dir = opts["out"]
lambda = opts["l1"]
tau_start = opts["tau"]
tau_min = opts["tau_min"]
tol = opts["epsilon"]
max_outer = opts["max_outer"]
max_inner = opts["max_inner"]
if opts["mkl"]
    using MKL
end
block_size = opts["block_size"]
offset = opts["offset"]

println("Available Threads: ", Threads.nthreads())
BLAS.set_num_threads(opts["threads"])

start_ind = block_size * offset + 1
end_ind = block_size * (offset + 1)

@assert start_ind + Size() - 1 <= size(X,2) # at least one dimension per each array
if block_size <= 0
    start_ind = 1
    end_ind = size(X,2)
end
if end_ind > size(X,2)
    end_ind = size(X,2)
end

if Rank() == 0
    @printf("Index starting from [%d] to [%d]\n", start_ind, end_ind)
end

v = ACCORDvariables(X, lambda, start_ind, end_ind)
u = ACCORDUpdate(max_outer, max_inner, tau_start, tau_min, tol)

if Rank() == 0
    @printf("Load complete. Starting iterations [%10.4lf]:\n", Dates.value(now() - start_time) * 0.001)
end
omega_nnz = accord!(u, v, start_time)
if Rank() == 0
    @printf("Saving matrix files. [%10.4lf]\n", Dates.value(now() - start_time) * 0.001)
end
omegaT = spzeros(Float64, v.p, v.p)
omegaT[:,block_distribute(start_ind, end_ind, Size())[Rank() + 1]] += v.OmegaT_old
npzwrite(join([output_dir, "-", cfmt("%04d", offset), "-", cfmt("%04d", Rank()), ".npz"]), Dict("p"=> v.p, "colptr" => omegaT.colptr, "rowval"=>omegaT.rowval, "nzval"=>omegaT.nzval))

if Rank() == 0
    @printf("Save complete. [%10.4lf]\n", Dates.value(now() - start_time) * 0.001)
end