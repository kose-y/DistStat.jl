using DistStat, Pkg, Test

if get(ENV,"JULIA_MPI_TEST_ARRAYTYPE","") == "CuArray"
    using CuArrays
    A = CuArray
else
    A = Array
end

type=[Float64,Float32]

for T in type
    N=9; M=7
    X=A{T}(reshape(collect(1:63),N,M))
    X_dist=distribute(X)
    X1=A{T}(undef,1, M); X1=distribute(X1)
    X2=A{T}(undef,N,1)
    cols=X_dist.partitioning[DistStat.Rank()+1][2]

    @test sum(X)==sum(X_dist)

    @test sum!(X1,X_dist).localarray==sum(X;dims=1)[:,cols]

    @test sum!(X2,X_dist)==sum(X;dims=2)

    @test sum(X_dist;dims=1).localarray==sum(X;dims=1)[:,cols]

    @test sum(X_dist;dims=2)==sum(X;dims=2)

end
