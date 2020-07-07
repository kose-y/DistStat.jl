using Test, Pkg
using DistStat
using LinearAlgebra

if get(ENV,"JULIA_MPI_TEST_ARRAYTYPE","") == "CuArray"
    using CUDA
    ArrayType = CuArray
else
    ArrayType = Array
end

type = [Float64, Float32, Int64, Int32]

for T in type
    data = ArrayType{T}(reshape(collect(1:25), 5, 5))
    a = distribute(data)
    b = MPIArray{eltype(a), 2, ArrayType}(undef, 1, 5)
    c = ArrayType{eltype(a)}(undef, 5)

    cols = b.partitioning[DistStat.Rank()+1][2]

    @test data[:, cols] == a.localarray

    val = 500

    fill_diag!(a, val)
    data[diagind(data)] .= val

    @test data[:, cols] == a.localarray

    DistStat.diag!(b, a)
    DistStat.diag!(c, a)

    @test all(b.localarray .== val)
    @test all(c .== val)
end

