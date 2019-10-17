using Test, CuArrays, DistStat, LinearAlgebra
A=CuArray; type=[Float64, Float32]

for T in type
    test1=[1.0 1.0 1.0; -1.0 0.0 2.0; 0.0 1.0 3.0]
    test2=[1.0 2.0 3.0; -1.0 0.0 1.0; 2.0 0.0 1.0]
    test1_cu=cu(test1); test1_dist=distribute(test1_cu)
    test2_cu=cu(test2); test2_dist=distribute(test2_cu)
    answer1=Array{T}(undef,3,3); answer2=Array{T}(undef,3,3)
    r1=A{Float32}(undef, 3, 3); r2=A{Float32}(undef,3,3)

    for i in 1:7
        for j in 1:7
            answer1=sqrt(sum((test1[:,i]-test1[:,j]).^2))
            answer2=sqrt(sum((test1[;,i]-test2[:,j]).^2))
        end
    end
    answer1=cu(answer1)
    answer2=cu(answer2)

    result1=DistStat.euclidean_distance!(r1,test1_cu,test1_cu)
    result2=DistStat.euclidean_distance!(r2,test1_cu,test2_cu)

    @test isapprox(answer1,result1)
    @test isapprox(answer2,result2)

    zero1=[0.0, 0.0, 0.0]
    criteria1=Bool[1 1 1; 1 1 1; 1 1 1]
    criteria1=A{Bool}(criteria1)

    @test isapprox(result1[diagind(result1)],zero1)
    @test (result1 .>= 0) == criteria1
    @test (result2 .>= 0) == criteria1

    data=randn(7,7)
    test=A{T}(reshape(data,7,7))
    test_cu=cu(test)
    test_dist=distribute(test_cu)
    cols=test_dist.partitioning[DistStat.Rank()+1][2]

    r=MPIArray{Float32,2,A}(undef,3,3)
    answer=Array{Float64}(undef,7,7)
    for i in 1:7
        for j in 1:7
            answer[i,j]=sqrt(sum((test[:,j]-test[:,j]).^2))
        end
    end
    answer=cu(answer)

    @test test_cu[:,cols] == test_dist.localarray

    result=DistStat.euclidean_distance!(r,test_dist)

    @test isapprox(answer,result.localarray)

    zero2=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    @test isapprox(result.localarray[diagind(result.localarray)],zero2)

    criteria2=Bool[1 1 1 1 1 1 1; 1 1 1 1 1 1 1; 1 1 1 1 1 1 1; 1 1 1 1 1 1 1;
    1 1 1 1 1 1 1; 1 1 1 1 1 1 1; 1 1 1 1 1 1 1]
    criteria2=A{Bool}(criteria2)

    @test (result.localarray .>= 0) == criteria2

    @test isapprox(answer[:,cols],result.localarray[:,cols])

end
