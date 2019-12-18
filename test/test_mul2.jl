using DistStat, LinearAlgebra, Test, Random

type=[Float32,Float64]

if get(ENV,"JULIA_MPI_TEST_ARRAYTYPE","") == "CuArray"
    using CuArrays
    ArrayType = CuArray
else
    ArrayType = Array
end

for T in type

    A=ArrayType{T}(reshape(collect(1:63),7,9))
    B=ArrayType{T}(reshape(collect(-31:31),7,9))
    A_dist=distribute(A)
    B_dist=distribute(B)
    B_distt=distribute(ArrayType{T}(transpose(B)))

    C=MPIMatrix{T,ArrayType}(undef,9,9)
    cols1=B_dist.partitioning[DistStat.Rank()+1][2]
    cols2=B_distt.partitioning[DistStat.Rank()+1][2]

    result1=LinearAlgebra.mul!(C,transpose(A),B_dist)
    ans1=transpose(A)*B

    println(@test isapprox(result1.localarray,ans1[:,cols1]))

    result2=LinearAlgebra.mul!(transpose(C),transpose(A_dist),B)
    println(@test isapprox(result2.localarray,ArrayType{T}(transpose(ans1))[:,cols1]))

end
