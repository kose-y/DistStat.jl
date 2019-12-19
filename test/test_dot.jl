using DistStat, Random, Test, LinearAlgebra

type=[Float32,Float64]

if get(ENV,"JULIA_MPI_TEST_ARRAYTYPE","") == "CuArray"
    using CuArrays
    ArrayType = CuArray
else
    ArrayType = Array
end

for T in type
    A=ArrayType{T}(reshape(collect(1:36),4,9))
    B=ArrayType{T}(reshape(collect(-7:28),4,9))

    A_dist=distribute(A); B_dist=distribute(B)

    @test isapprox(LinearAlgebra.dot(A_dist,B_dist),LinearAlgebra.dot(A,B))

    A_vec=vec(A); B_vec=vec(B)

    @test isapprox(LinearAlgebra.dot(A_dist,B_dist),LinearAlgebra.dot(A_vec,B_vec))

end
