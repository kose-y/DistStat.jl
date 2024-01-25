import MPI
import MPI: COMM_WORLD
using Random, SparseArrays, BenchmarkTools, LinearAlgebra

MPI.Initialized() || MPI.Init()

@inline function Size()
    MPI.Comm_size(COMM_WORLD)
end

@inline function Rank()
    MPI.Comm_rank(COMM_WORLD)
end

include("distdirectives.jl")
include("distarray.jl")
include("distlinalg.jl")
include("reduce.jl")
include("accumulate.jl")
include("broadcast.jl")
include("arrayfunctions.jl")
include("utils.jl")

aa = randn(3,6)
bb = randn(6,4)
cc = randn(3,4)

# aa = [1 1; 1 1]
# bb = [1 2; 3 4]
# cc = [0 1; 1 0]
if Rank() == 0
    println(aa * bb)
end
A = distribute(aa)
B = distribute(bb)
C = distribute(cc)

mul_1d!(C, A, B)
show(C)

# p = 5000
# if Rank() == 0
#     aa = randn(p,p)
#     bb = randn(p,p)
#     cc = zeros(p,p)
# else
#     aa = [0.0;;]
#     bb = [0.0;;]
# end

# A = distribute(aa)
# B = distribute(bb)
# C = MPIArray{Float64, 2, Array}(undef, p,p)

# if Rank() == 0
#     println("time of base mul!:")
#     @btime begin
#         LinearAlgebra.mul!(cc,aa,bb)
#     end
# end
# sync(C)

# if Rank() == 0
#     println("time of mul! in DistStat.jl:")
# end
# @btime begin
#     LinearAlgebra.mul!(C,A,B)
#     sync(C)
# end
# if Rank() == 0
#     println("time of mul_1d! in DistStat.jl:")
# end
# @btime begin
#     mul_1d!(C,A,B)
#     sync(C)
# end

# sync(C)
# if Rank() == 1
#     println(C.localarray)
# end