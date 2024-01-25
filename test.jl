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

include("src/distdirectives.jl")
include("src/distarray.jl")
include("src/distlinalg.jl")
include("src/reduce.jl")
include("src/accumulate.jl")
include("src/broadcast.jl")
include("src/arrayfunctions.jl")
include("src/utils.jl")

aa = randn(3,6)
bb = randn(6,4)
cc = randn(3,4)

# aa = [1 1; 1 1]
# bb = [1 2; 3 4]
# cc = [0 1; 1 0]
if Rank() == 0
    println("Single process multiplication of AB:")
    println(aa * bb)
    println("Multi-process multiplication of AB:")
end
A = distribute(aa)
B = distribute(bb)
C = distribute(cc)

mul_1d!(C, A, B)
show(C)