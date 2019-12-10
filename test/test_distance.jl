using DistStat, Test, LinearAlgebra, Pkg
type=[Float64,Float32]

if get(ENV,"JULIA_MPI_TEST_ARRAYTYPE","") == "CuArray"
    using CuArrays
    ArrayType = CuArray
else
    ArrayType = Array
end

for T in type
    test1=[1.0 1.0 1.0; -1.0 0.0 2.0; 0.0 1.0 3.0]
    test2=[1.0 2.0 3.0; -1.0 0.0 1.0; 2.0 0.0 1.0]
    test1_conv=ArrayType{T}(test1); test2_conv=ArrayType{T}(test2)
    answer1=Array{T}(undef,3,3); answer2=Array{T}(undef,3,3)
    r1=ArrayType{T}(undef, 3, 3); r2=ArrayType{T}(undef,3,3)

    for i in 1:3
        for j in 1:3
            answer1[i,j]=sqrt(sum((test1[:,i]-test1[:,j]).^2))
            answer2[i,j]=sqrt(sum((test1[:,i]-test2[:,j]).^2))
        end
    end
    answer1=ArrayType{T}(answer1)
    answer2=ArrayType{T}(answer2)

    result1=DistStat.euclidean_distance!(r1,test1_conv,test1_conv)
    result2=DistStat.euclidean_distance!(r2,test1_conv,test2_conv)

    @test isapprox(answer1,result1)
    @test isapprox(answer2,result2)

    zero1=[0.0, 0.0, 0.0]

    @test isapprox(result1[diagind(result1)],zero1)
    @test all(result1 .>= 0)
    @test all(result2 .>= 0)

    test=reshape(collect(-24:24),7,7)
    test_conv=ArrayType{T}(test)
    test_dist=distribute(test_conv)
    cols=test_dist.partitioning[DistStat.Rank()+1][2]

    r=MPIArray{T,2,ArrayType}(undef,7,7)
    answer=Array{T}(undef,7,7)
    for i in 1:7
        for j in 1:7
            answer[i,j]=sqrt(sum((test[:,i]-test[:,j]).^2))
        end
    end
    answer=ArrayType{T}(answer)

    result=DistStat.euclidean_distance!(r,test_dist)

    @test isapprox(answer[:,cols],result.localarray)

    zero2=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    @test isapprox(answer[diagind(answer)],zero2)

    @test all(result.localarray .>= 0.0)

end
