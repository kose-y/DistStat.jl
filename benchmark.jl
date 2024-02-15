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

if Rank() != 0
    redirect_stdout(devnull)
end

p = 5000
if Rank() == 0
    aa = randn(p,p)
    bb = randn(p,p)
    cc = zeros(p,p)
    LinearAlgebra.mul!(cc,aa,bb)
else
    aa = [0.0;;]
    bb = [0.0;;]
end

A = distribute(aa)
B = distribute(bb)
C = MPIArray{Float64, 2, Array}(undef, p, p)

LinearAlgebra.mul!(C,A,B)
mul_1d!(C,A,B)

if Rank() == 0
    println("1) time of base mul!:")
    @time begin
        LinearAlgebra.mul!(cc,aa,bb)
    end
    println("2) time of base mul!:")
    @time begin
        LinearAlgebra.mul!(cc,aa,bb)
    end
    println("3) time of base mul!:")
    @time begin
        LinearAlgebra.mul!(cc,aa,bb)
    end
end
sync(C)
println(" ")
println("1) time of mul! in DistStat.jl:")
sync(C)
@btime begin
    LinearAlgebra.mul!(C,A,B)
    sync(C)
end
println("2) time of mul! in DistStat.jl:")
sync(C)
@btime begin
    LinearAlgebra.mul!(C,A,B)
    sync(C)
end
println("3) time of mul! in DistStat.jl:")
sync(C)
@btime begin
    LinearAlgebra.mul!(C,A,B)
    sync(C)
end
println(" ")
println("1) time of mul_1d! in DistStat.jl:")
sync(C)
@btime begin
    mul_1d!(C,A,B)
    sync(C)
end
println("2) time of mul_1d! in DistStat.jl:")
sync(C)
@btime begin
    mul_1d!(C,A,B)
    sync(C)
end
println("3) time of mul_1d! in DistStat.jl:")
sync(C)
@btime begin
    mul_1d!(C,A,B)
    sync(C)
end


# if Rank() == 1
#     println(C.localarray)
# end