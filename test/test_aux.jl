using DistStat, Random, Test, Pkg

type=[Float32,Float64]

if get(ENV,"JULIA_MPI_TEST_ARRAYTYPE","") == "CuArray"
    using CuArrays
    ArrayType = CuArray
else
    ArrayType = Array
end

for T in type
  A=ArrayType{T}(undef,7,10)
  A_dist=distribute(A)
  fill!(A, 1.0)
  fill!(A_dist,1.0)
  cols1=A_dist.partitioning[DistStat.Rank()+1][2]

  println(@test isapprox(A_dist.localarray,A[:,cols1]))

  B=reshape(collect(1:70), 7, 10)
  B_dist=distribute(B)
  cols2=B_dist.partitioning[DistStat.Rank()+1][2]

  println(@test isapprox(B_dist.localarray,B[:,cols2]))

  C_dist = MPIArray{T, 2, ArrayType}(undef, 7, 9)
  cols3=C_dist.partitioning[DistStat.Rank()+1][2]
  randn!(C_dist; seed=0,common_init=true)

  C=ArrayType{T}(undef,size(C_dist))
  Random.seed!(0)
  randn!(C)
  println(@test isapprox(C_dist.localarray,C[:,cols3]))

end
