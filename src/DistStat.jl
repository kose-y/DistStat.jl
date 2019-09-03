module DistStat

import MPI
import MPI: COMM_WORLD

const tempcounts = Ref{Vector{Integer}}()

function __init__()
    MPI.Initialized() || MPI.Init()
    tempcounts.x = zeros(Integer, MPI.Comm_size(MPI.COMM_WORLD))
end

@inline function Size()
    MPI.Comm_size(COMM_WORLD)
end
@inline function Rank()
    MPI.Comm_rank(COMM_WORLD)
end

include("distdirectives.jl")
include("distarray.jl")
include("cuda.jl")
include("distlinalg.jl")
include("reduce.jl")
include("broadcast.jl")

include("arrayfunctions.jl")
end # module
