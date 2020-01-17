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

    iter=15; interval=1

    u = PETUpdate_l1(;maxiter=iter, step=interval, verbose=true)
    v = PETVariables_l1(y, E, D, rho; eval_obj=eval_obj, sigma=sigma, tau=tau)
    reset!(v)

    ans=0.8906939029570893
    result=pet_l1!(u, v)

    @test isapprox(ans,result)

end
