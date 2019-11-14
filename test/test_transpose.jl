using DistStat, Pkg, Test

type=[Float32, Float64]

if haskey(Pkg.installed(), "CuArrays")
    using CuArrays
    A = CuArray
else
    A = Array
end

for T in type
    data=A{T}(reshape(collect(1:81),9,9))
    data_dist=distribute(data)
    cols=data_dist.partitioning[DistStat.Rank()+1][2]

    b=A{T}(undef,1,9); fill!(b,1.0)
    b_dist=distribute(b)

    data_dist .= data_dist .+ b_dist
    data = data .+ b

    @test isapprox(data_dist.localarray,data[:,cols])

end
