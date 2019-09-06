module DistStat

import MPI
import MPI: COMM_WORLD
using Requires

function __init__()
    MPI.Initialized() || MPI.Init()
    @require CuArrays="865a2d-5b23-5a0f-bc46-62713ec82fae" begin
        @require CUDAnative="" begin
            include("cuda.jl")
            set_device!()
        end
    end

end

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
include("broadcast.jl")

include("arrayfunctions.jl")
end # module
