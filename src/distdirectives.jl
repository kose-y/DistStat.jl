# TODO: Scatter (Scatterv), Gather (Gatherv), Allgather (Allgatherv)
"""
These are the simplified version of communication methods, only using MPI.COMM_WORLD.
"""

function Init()
    MPI.Init()
end

const Comm = MPI.COMM_WORLD
const Size = MPI.Comm_size(Comm)
const Rank = MPI.Comm_rank(Comm)

function Barrier()
    MPI.Barrier(Comm)
end

function Bcast!(arr::AbstractArray, root::Integer=0)
    @assert Base.iscontiguous(arr)
    MPI.Bcast!(arr, root, Comm)
end

function Send(arr::AbstractArray, dest::Integer, tag::Integer=0)
    @assert Base.iscontiguous(arr)
    MPI.Send(arr, dest, tag, Comm)
end

function Recv!(arr::AbstractArray, src::Integer, tag::Integer=0)
    @assert Base.iscontiguous(arr)
    MPI.Recv!(arr, src, tag, Comm)
end

function Isend(arr::AbstractArray, dest::Integer, tag::Integer=0)
    @assert Base.iscontiguous(arr)
    MPI.Isend(arr, dest, tag, Comm)
end

function Irecv!(arr::AbstractArray, src::Integer, tag::Integer=0)
    @assert Base.iscontiguous(arr)
    MPI.Irecv!(arr, src, tag, Comm)
end

function Reduce!(sendarr::AbstractArray, recvarr::AbstractArray, op=MPI.SUM, root::Integer=0)
    @assert Base.iscontiguous(sendarr)
    @assert Base.iscontiguous(recvarr)
    MPI.Reduce!(sendarr, recvarr, op, root, Comm)
end

function Allreduce!(arr::AbstractArray, op=MPI.SUM)
    @assert Base.iscontiguous(arr)
    MPI.Allreduce!(arr, op, Comm)
end

function Allgatherv!(sendarr::AbstractArray, recvarr::AbstractArray, counts::Vector{Cint})
#### TODO: partition by last dim of the array (is it doable?)
    @assert Base.iscontiguous(sendarr)
    @assert Base.iscontiguous(recvarr)
    MPI.Allgatherv!(sendarr, recvarr, counts, Comm)
end

function Scatterv!(sendarr::AbstractArray, recvarr::AbstractArray, counts::Vector{Cint}, root::Integer=0)
#### TODO: partition by last dim of the array (is it doable?)
    @assert Base.iscontiguous(sendarr)
    @assert Base.iscontiguous(recvarr)
    MPI.Scatterv!(sendarr, recvarr, counts, root, Comm)
end

function Gatherv!(sendarr::AbstractArray, recvarr::AbstractArray, counts::Vector{Cint}, root::Integer=0)
#### TODO: partition by last dim of the array (is it doable?)
    @assert Base.iscontiguous(sendarr)
    @assert Base.iscontiguous(recvarr)
    Gatherv!(sendarr, recvarr, counts, root, Comm)
end

