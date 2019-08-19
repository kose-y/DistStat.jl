# TODO: Scatter (Scatterv), Gather (Gatherv), Allgather (Allgatherv)
"""
These are the simplified version of communication methods, only using MPI.COMM_WORLD.
"""

function Barrier()
    MPI.Barrier(COMM_WORLD)
end

function Bcast!(arr::AbstractArray; root::Integer=0)
    MPI.Bcast!(arr, root, COMM_WORLD)
end

function Send(arr::AbstractArray, dest::Integer; tag::Integer=0)
    MPI.Send(arr, dest, tag, COMM_WORLD)
end

function Recv!(arr::AbstractArray, src::Integer; tag::Integer=0)
    MPI.Recv!(arr, src, tag, COMM_WORLD)
end

function Isend(arr::AbstractArray, dest::Integer; tag::Integer=0)
    MPI.Isend(arr, dest, tag, COMM_WORLD)
end

function Irecv!(arr::AbstractArray, src::Integer; tag::Integer=0)
    MPI.Irecv!(arr, src, tag, COMM_WORLD)
end

function Reduce!(sendarr::AbstractArray, recvarr::AbstractArray; op=MPI.SUM, root::Integer=0)
    MPI.Reduce!(sendarr, recvarr, op, root, COMM_WORLD)
end

function Allreduce!(arr::AbstractArray; op=MPI.SUM)
    MPI.Allreduce!(arr, op, COMM_WORLD)
end

function Allgather!(sendarr::AbstractArray, recvarr::AbstractArray)
    @assert length(recvarr) >= Size() * length(sendarr)
    MPI.Allgather!(sendarr, recvarr, length(sendarr), COMM_WORLD)
end

function Allgather!(sendarr::AbstractArray, recvarr::AbstractArray, count::Integer)
    @assert length(recvarr) >= Size() * count
    MPI.Allgather!(sendarr, recvarr, count, COMM_WORLD)
end

function Allgatherv!(sendarr::AbstractArray, recvarr::AbstractArray, counts::Vector{Cint})
#### TODO: auto-detect counts (is it doable?)
    MPI.Allgatherv!(sendarr, recvarr, counts, COMM_WORLD)
end

function Scatter!(sendarr::AbstractArray, recvarr::AbstractArray; root::Integer=0)
    MPI.Scatter!(sendarr, recvarr, length(recvarr), root, COMM_WORLD)
end

function Scatterv!(sendarr::AbstractArray, recvarr::AbstractArray, counts::Vector{Cint}; root::Integer=0)
#### TODO: auto-detect counts (is it doable?)
    MPI.Scatterv!(sendarr, recvarr, counts, root, COMM_WORLD)
end

function Gather!(sendarr::AbstractArray, recvarr::AbstractArray; root::Integer=0)
    MPI.Gather!(sendarr, recvarr, length(sendbuf), root, COMM_WORLD)
end

function Gatherv!(sendarr::AbstractArray, recvarr::AbstractArray, counts::Vector{Cint}; root::Integer=0)
#### TODO: auto-detect counts (is it doable?)
    Gatherv!(sendarr, recvarr, counts, root, COMM_WORLD)
end

