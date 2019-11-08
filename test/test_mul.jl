using Pkg, MPI, DistStat, Random, Test, LinearAlgebra

if haskey(Pkg.installed(), "CuArrays")
    using CuArrays
    A = CuArray
else
    A = Array
end

type=[Float64,Float32]

for T in type
    N=6; M=5
    X = Array{T}(reshape(collect(1:N*M),N,M)); X_dist=distribute(X)
    XX = Array{T}(reshape(collect(1:M*M),M,M)); XX_dist=distribute(XX)
    y = Array{T}(undef,N,N); y_dist=distribute(y)
    cols1=X_dist.partitioning[DistStat.Rank()+1][2]
    cols2=y_dist.partitioning[DistStat.Rank()+1][2]

    w1 = similar(y); w1=distribute(w1)
    result1=X*transpose(X)

    LinearAlgebra.mul!(w1,X_dist,transpose(X_dist))
    @test isapprox(result1[:,cols2],w1.localarray)

    LinearAlgebra.mul!(transpose(w1), X_dist,transpose(X_dist))
    @test isapprox(result1[:,cols2],w1.localarray)

    w2=similar(XX); w2=distribute(w2)
    result2=transpose(X)*X

    LinearAlgebra.mul!(w2,transpose(X_dist),X_dist)
    @test isapprox(result2[:,cols1],w2.localarray)

    LinearAlgebra.mul!(transpose(w2),transpose(X_dist),X_dist)
    @test isapprox(result2[:,cols1],w2.localarray)

    w3=similar(X); w3=distribute(w3)
    result3=X*XX

    LinearAlgebra.mul!(w3,X_dist,XX_dist)
    @test isapprox(result3[:,cols1],w3.localarray)

    LinearAlgebra.mul!(transpose(w3),transpose(XX_dist),transpose(X_dist))
    @test isapprox(result3[:,cols1],w3.localarray)

    LinearAlgebra.mul!(y,X_dist,transpose(X_dist))

    w4=similar(X); w4=distribute(w4)
    result4=y*X

    LinearAlgebra.mul!(w4,y,X_dist)
    @test isapprox(result4[:,cols1],w4.localarray)

    LinearAlgebra.mul!(transpose(w4),transpose(X_dist),transpose(y))
    @test isapprox(result4[:,cols1],w4.localarray)

    v1 = Array{Float64}(reshape(collect(1:6),6))
    v1_dist=distribute(v1)
    v2 = Array{Float64}(reshape(collect(1:5),5))
    v2_dist=distribute(v2)
    lv1 = Array{Float64}(undef, 6)
    lv2 = Array{Float64}(undef, 5)
    lv2=distribute(lv2)

    ans1=LinearAlgebra.mul!(lv1,X_dist,v2)
    ans2=LinearAlgebra.mul!(lv1,X_dist,v2_dist)

    @test isapprox(ans1,ans2)

    ans3=LinearAlgebra.mul!(lv2,transpose(X_dist),v1)
    ans4=LinearAlgebra.mul!(lv2,transpose(X_dist),v1_dist)

    @test isapprox(ans3.localarray,ans4.localarray)

end
