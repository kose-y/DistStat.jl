using DistStat, Pkg, Test

type=[Float32 ,Float64]

if get(ENV,"JULIA_MPI_TEST_ARRAYTYPE","") == "CuArray"
    using CUDA
    A = CuArray
else
    A = Array
end

for T in type
    x=A(zeros(T,5,13)); x_dist=distribute(x)
    cols=x_dist.partitioning[DistStat.Rank()+1][2]

    y=A{T}(reshape(collect(-6:6),1,13))
    y_dist=distribute(y)

    x_dist .+= 8.0; x .+= 8.0

    @test isapprox(x_dist.localarray,x[:,cols])

    x_dist .*= 1.5; x .*= 1.5

    @test isapprox(x_dist.localarray,x[:,cols])

    x_dist .+= y_dist; x .+= y

    @test isapprox(x_dist.localarray, x[:, cols])

    z=A{T}([1,2,3,4,5])

    x_dist .+= z; x .+= z

    @test isapprox(x_dist.localarray,x[:,cols])

    x_dist .= log.(x_dist); x .= log.(x)

    @test isapprox(x_dist.localarray,x[:,cols])

end
