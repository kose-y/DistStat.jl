using DistStat, LinearAlgebra, Test 

type=[Float32,Float64]

if get(ENV,"JULIA_MPI_TEST_ARRAYTYPE","") == "CuArray"
    using CuArrays
    ArrayType = CuArray
else
    ArrayType = Array
end

for T in type
    A=ArrayType{T}(reshape(collect(1:45),5,9))
    A_dist=distribute(A)

    @test isapprox(opnorm(A_dist,1),opnorm(A,1))
    @test isapprox(opnorm(A_dist,2),opnorm(A,2))
    @test isapprox(opnorm(A_dist,Inf),opnorm(A,Inf))

end
