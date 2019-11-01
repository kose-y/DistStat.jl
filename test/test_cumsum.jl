using DistStat, Pkg, Random, Test

type=[Float64,Float32]

if haskey(Pkg.installed(), "CuArrays")
    using CuArrays
    A = CuArray
else
    A = Array
end

for T in type
    a = A{T}(reshape(collect(1:49),7,7))
    a_dist = distribute(a)
    cols=a_dist.partitioning[DistStat.Rank()+1][2]

    println(@test cumsum(a,dims=1)[:,cols]==cumsum(a_dist,dims=1).localarray)
    println(@test cumsum(a,dims=2)[:,cols]==cumsum(a_dist,dims=2).localarray)

end
