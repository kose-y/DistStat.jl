using Pkg, Test, DistStat

type=[Float64,Float32]

if get(ENV,"JULIA_MPI_TEST_ARRAYTYPE","") == "CuArray"
    using CuArrays
    ArrayType = CuArray
else
    ArrayType = Array
end

for T in type

 data =ArrayType{T}(reshape(collect(1:42),6,7))
 data_dist1 = distribute(data)
 data_dist2 = distribute(ArrayType{T}(transpose(data)))
 cols1=data_dist1.partitioning[DistStat.Rank()+1][2]
 cols2=data_dist2.partitioning[DistStat.Rank()+1][2]

 @test data_dist1.localarray==data[:,cols1]
 @test data_dist2.localarray==(ArrayType{T}(transpose(data)))[:,cols2]

end
