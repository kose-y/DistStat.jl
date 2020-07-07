using Pkg, MPI, Test
MPI.Init()

type=[Float64,Float32]

if get(ENV,"JULIA_MPI_TEST_ARRAYTYPE","") == "CuArray"
    using CUDA
    A = CuArray
else
    A = Array
end

for T in type
    a=A{T}(reshape(collect(1:100),10,10))
    b = MPI.Allreduce(a, MPI.MIN, MPI.COMM_WORLD)
    @test isapprox(a,b)
end
