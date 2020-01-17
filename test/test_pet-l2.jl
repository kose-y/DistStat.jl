using NPZ, LinearAlgebra, SparseArrays, DistStat, MPI, Test

if get(ENV,"JULIA_MPI_TEST_ARRAYTYPE","") == "CuArray"
    using CuArrays
    A = CuArray
else
    A = Array
end

type=[Float64,Float32]

include("pet-l2_fun.jl")
include("cmdline.jl")

variables = parse_commandline_pet_l2()

SA = SparseMatrixCSC
if variables["gpu"]
    using CuArrays
    using CuArrays.CUSPARSE
    SA = CuSparseMatrixCSR
end

eval_obj = variables["eval_obj"]
datafile = variables["data"]
mu = variables["reg"]
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

    G_indices = dat["G_indices"] .+ 1
    G_values  = convert(Array{T}, dat["G_values"])
    G_shape   = dat["G_shape"]

    y = dat["counts"]
    y = A(y)

    E = sparse(e_indices[1,:], e_indices[2,:], e_values, m, n)
    G = sparse(G_indices[1,:], G_indices[2,:], G_values, G_shape[1], G_shape[2])
    D = sparse(D_indices[1,:], D_indices[2,:], D_values, D_shape[1], D_shape[2])
    t = MPIVector{T,A}(undef, n)

    y, G, D = convert(A{T}, y), SA(G), SA(D)

    E = distribute(E; A=A)
    DistStat.Bcast!(y)
    G, D = MPI.bcast((G, D), 0, MPI.COMM_WORLD)

    iter=15; interval=1

    u = PETUpdate_l2(;maxiter=iter, step=interval, verbose=true)
    v = PETVariables_l2(y, E, G, D, mu; eval_obj=eval_obj)
    reset!(v)

    ans=7.146081552445253
    result=pet_l2!(u, v)

    @test isapprox(ans,result)

end
