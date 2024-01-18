import MPI
import MPI: COMM_WORLD
using Random, SparseArrays

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

aa = randn(5,4)#[1 1; 1 1]
bb = randn(4,3)#[1 2; 3 4]
cc = randn(5,3)#[0 1; 1 0]
if Rank() == 0
    println(aa * bb)
end
A = distribute(aa)
B = distribute(bb)
C = distribute(cc)

mul_1d!(C, A, B)
show(C)

# if Rank() == 0
#     println(C.localarray)
# end
# sync(C)
# if Rank() == 1
#     println(C.localarray)
# end