using NPZ, LinearAlgebra, SparseArrays, DistStat, MPI, Test

if get(ENV,"JULIA_MPI_TEST_ARRAYTYPE","") == "CuArray"
    using CuArrays
    A = CuArray
else
    A = Array
end

type=[Float64,Float32]

include("pet-l1_fun.jl")
include("cmdline.jl")

variables = parse_commandline_pet_l1()

interval = variables["step"]
SA = SparseMatrixCSC
if variables["gpu"]
    using CuArrays
    using CuArrays.CUSPARSE
    SA = CuSparseMatrixCSR
end

eval_obj = variables["eval_obj"]
datafile = variables["data"]
rho = variables["reg"]
sigma = variables["sigma"]
tau   = variables["tau"]
dat = npzread(datafile)

for T in type

    n_x = dat["n_x"]
    n_t = dat["n_t"]
    e_indices = dat["e_indices"] .+ 1
    e_values = convert(Array{T}, dat["e_values"])
    m = n_t * (n_t - 1) ÷ 2
    n = n_x ^ 2

    m, n = MPI.bcast((m, n), 0, MPI.COMM_WORLD)

    D_indices = dat["D_indices"] .+ 1
    D_values  = convert(Array{T}, dat["D_values"])
    D_shape   = dat["D_shape"]

    y = dat["counts"]
    y = A(y)

    E = sparse(e_indices[1,:], e_indices[2,:], e_values, m, n)
    D = sparse(D_indices[1,:], D_indices[2,:], D_values, D_shape[1], D_shape[2])
    t = MPIVector{T,A}(undef, n)
    y, D = convert(A{T}, y), SA(D)

    E = distribute(E; A=A)
    DistStat.Bcast!(y)
    D = MPI.bcast((D), 0, MPI.COMM_WORLD)

    iter1=2000;iter2=2100

    u1 = PETUpdate_l1(;maxiter=iter1, step=interval, verbose=true)
    u2 = PETUpdate_l1(;maxiter=iter2, step=interval, verbose=true)
    v = PETVariables_l1(y, E, D, rho; eval_obj=eval_obj, sigma=sigma, tau=tau)
    reset!(v)

    result1=pet_l1!(u1, v)
    result2=pet_l1!(u2, v)

    tol=5e-03

    @test abs(result1-result2)<tol

end
