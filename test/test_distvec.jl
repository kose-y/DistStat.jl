using DistStat, Random, Test, Pkg, LinearAlgebra

if haskey(Pkg.installed(), "CuArrays")
    using CuArrays
    A = CuArray
else
    A = Array
end

type=[Float64, Float32]

for T in type
    v= MPIVector{T,A}(undef, 10)
    v=randn!(v)
    b=distribute(Array{T}(undef,1,10))
    cols=b.partitioning[DistStat.Rank()+1][2]
    @test size(v)[1]==size(cols)[1]
    @test typeof(transpose(v))==LinearAlgebra.Transpose{T,A{T,1}}
end
