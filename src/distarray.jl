export MPIArray, localindices, getblock, getblock!, putblock!, allocate, forlocalpart, forlocalpart!, free, redistribute, redistribute!, sync, GlobalBlock, GhostedBlock, getglobal, globaltolocal, globalids
export @blockwise, @blockdoteq
using MPI
import LinearAlgebra
import Base: show, print_array, summary, array_summary
import Adapt: adapt
"""
Store the distribution of the array indices over the different partitions.
This class forces a continuous, ordered distribution without overlap
    ContinuousPartitioning(partition_sizes...)
Construct a distribution using the number of elements per partition in each direction, e.g:
```
p = ContinuousPartitioning([2,5,3], [2,3])
```
will construct a distribution containing 6 partitions of 2, 5 and 3 rows and 2 and 3 columns.
"""
struct ContinuousPartitioning{N} <: AbstractArray{Int,N}
    ranks::LinearIndices{N,NTuple{N,Base.OneTo{Int}}}
    index_starts::NTuple{N,Vector{Int}}
    index_ends::NTuple{N,Vector{Int}}

    function ContinuousPartitioning(partition_sizes::Vararg{Any,N}) where {N}
        index_starts = Vector{Int}.(undef,length.(partition_sizes))
        index_ends = Vector{Int}.(undef,length.(partition_sizes))
        for (idxstart,idxend,nb_elems_dist) in zip(index_starts,index_ends,partition_sizes)
            currentstart = 1
            currentend = 0
            for i in eachindex(idxstart)
                currentend += nb_elems_dist[i]
                idxstart[i] = currentstart
                idxend[i] = currentend
                currentstart += nb_elems_dist[i]
            end
        end
        ranks = LinearIndices(length.(partition_sizes))
        return new{N}(ranks, index_starts, index_ends)
    end
end

Base.IndexStyle(::Type{ContinuousPartitioning{N}}) where {N} = IndexCartesian()
Base.size(p::ContinuousPartitioning) = length.(p.index_starts)
@inline function Base.getindex(p::ContinuousPartitioning{N}, I::Vararg{Int, N}) where {N}
    return UnitRange.(getindex.(p.index_starts,I), getindex.(p.index_ends,I))
end

function partition_sizes(p::ContinuousPartitioning)
    result = (p.index_ends .- p.index_starts)
    for v in result
        v .+= 1
    end
    return result
end

"""
  (private method)
Get the rank and local 0-based index
"""
function local_index(p::ContinuousPartitioning, I::NTuple{N,Int}) where {N}
    proc_indices = searchsortedfirst.(p.index_ends, I)
    lininds = LinearIndices(Base.Slice.(p[proc_indices...]))
    return (p.ranks[proc_indices...]-1, lininds[I...] - first(lininds))
end

# Evenly distribute nb_elems over parts partitions
function distribute(nb_elems, parts)
    local_len = nb_elems ÷ parts
    remainder = nb_elems % parts
    return [p <= remainder ? local_len+1 : local_len for p in 1:parts]
end

function local_lengths(x::ContinuousPartitioning{N}) where N
    convert(Vector{Cint}, vec(map(x) do y
        prod(map(z -> length(z), y))
    end))
end

@inline allsame(x) = all(y -> y == first(x), x)

mutable struct MPIArray{T,N,A} <: AbstractArray{T,N}
    sizes::NTuple{N,Int}
    localarray::Union{A,Nothing}
    partitioning::ContinuousPartitioning{N}
    comm::MPI.Comm
    # win::MPI.Win
    myrank::Int
    local_lengths::Vector{Cint}
    
    MPIArray{T,N,A}(sizes, localarray, partitioning, comm, myrank, local_lengths) where {T,N,A} = new{T,N,A}(sizes, localarray, partitioning, comm, myrank, local_lengths)
    function MPIArray{T,N,A}(comm::MPI.Comm, partition_sizes::Vararg{AbstractVector{<:Integer},N}) where {T,N,A}
        nb_procs = MPI.Comm_size(comm)
        rank = MPI.Comm_rank(comm)
        partitioning = ContinuousPartitioning(partition_sizes...)

        localarray = A{T}(undef,length.(partitioning[rank+1]))
        # win = MPI.Win_create(localarray, comm)
        sizes = sum.(partition_sizes)
        # return new{T,N,A}(sizes, localarray, partitioning, comm, win, rank, local_lengths(partitioning))
        return new{T,N,A}(sizes, localarray, partitioning, comm, rank, local_lengths(partitioning))
    end
    MPIArray{T,N,A}(comm::MPI.Comm, partitions::NTuple{N,<:Integer}, sizes::Vararg{<:Integer,N}) where {T,N,A} = MPIArray{T,N,A}(comm, distribute.(sizes, partitions)...)
    MPIArray{T,N,A}(sizes::Vararg{<:Integer,N}) where {T,N,A} = MPIArray{T,N,A}(MPI.COMM_WORLD, (ones(Int, N-1)..., MPI.Comm_size(MPI.COMM_WORLD)), sizes...)
    MPIArray{T,N,A}(::UndefInitializer, sizes::Vararg{<:Integer,N}) where {T,N,A} = MPIArray{T,N,A}(MPI.COMM_WORLD, (ones(Int, N-1)..., MPI.Comm_size(MPI.COMM_WORLD)), sizes...)

    function MPIArray{T,N,A}(comm::MPI.Comm, localarray::AbstractArray{T,N}, nb_partitions::Vararg{<:Integer,N}) where {T,N,A}
        nb_procs = MPI.Comm_size(comm)
        rank = MPI.Comm_rank(comm)
        
        partition_size_array = reshape(MPI.Allgather(size(localarray), comm), Int.(nb_partitions)...)

        partition_sizes = ntuple(N) do dim
            idx = ntuple(i -> i == dim ? Colon() : 1,N)
            return getindex.(partition_size_array[idx...],dim)
        end

        # win = MPI.Win_create(localarray, comm)
        partitioning = ContinuousPartitioning(partition_sizes...)
        # result = new{T,N,A}(sum.(partition_sizes), A(localarray), partitioning, comm, win, rank, local_lengths(partitioning))
        result = new{T,N,A}(sum.(partition_sizes), A(localarray), partitioning, comm, rank, local_lengths(partitioning))
        return result
    end

    MPIArray{T,N,A}(localarray::AbstractArray{T,N}, nb_partitions::Vararg{<:Integer,N}) where {T,N,A} = MPIArray(MPI.COMM_WORLD, A(localarray), nb_partitions...)
end

function MPIArray(comm::MPI.Comm, init::Function, partition_sizes::Vararg{AbstractVector{<:Integer}, N}; T=nothing, A=nothing) where N
    nb_procs = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)
    partitioning = ContinuousPartitioning(partition_sizes...)
    
    localarray = construct_localpart(init, partitioning;T=T,A=A)
    
    T = eltype(localarray)
    TLA = typeof(localarray)
    A = hasproperty(TLA, :name) ? TLA.name.wrapper : TLA

    sizes = sum.(partition_sizes)

    MPIArray{T,N,A}(sizes, localarray, partitioning, comm, rank, local_lengths(partitioning))
    
end

MPIArray(init::Function, partition_sizes::Vararg{AbstractVector{<:Integer}, N};T=nothing,A=nothing) where N = MPIArray(MPI.COMM_WORLD, init, partition_sizes...;T=T,A=A)

MPIArray(init::Function, partitions::NTuple{N,<:Integer}, sizes::Vararg{<:Integer,N};T=nothing, A=nothing) where N = MPIArray(init, distribute.(sizes, partitions)...; T=T,A=A)

MPIArray(init::Function, sizes::Vararg{<:Integer,N}; T=nothing, A=nothing) where N = MPIArray(init, (ones(Int, N-1)..., MPI.Comm_size(MPI.COMM_WORLD)), sizes...; T=T, A=A)

function MPIArray(comm::MPI.Comm, localarray::AbstractArray)
    #TODO, construct from localarrays 
end

MPIArray(localarray::AbstractArray) = MPIArray(MPI.COMM_WORLD, localarray)

function construct_localpart(init, partitioning; T=nothing, A=nothing)
    localidx = partitioning[Rank() + 1]
    localpart = init(localidx)
    if A == nothing
        TLA = typeof(localpart)
        A = hasproperty(TLA, :name) ? TLA.name.wrapper : TLA
    end
    if T == nothing
        T = eltype(localpart)
    end
    adapt(A{T}, localpart)
end

MPIVector{T,A} = MPIArray{T,1,A}
MPIMatrix{T,A} = MPIArray{T,2,A}
MPIVecOrMat{T,A} = Union{MPIVector{T,A},MPIMatrix{T,A}}

Base.IndexStyle(::Type{MPIArray{T,N,A}}) where {T,N,A} = IndexCartesian()

Base.size(a::MPIArray) = a.sizes


"""
    show(io::IO, x::MPIArray)

String representation of an MPIArray
"""
function show(io::IO, m::MIME"text/plain", X::MPIArray)
    summary(io, X)
    print(io, "\nlocal part:\n")
    show(io, m, X.localarray)
end

function show(io::IO, X::MPIArray)
    print(io, "local part:\n")
    show(io, X.localarray)
end

function summary(io::IO, a::MPIArray)
    array_summary(io, a, axes(a))
    print(io, " with local block of ")
    array_summary(io, a.localarray, axes(a.localarray))
end

"""
    split_data(arr; root=0, T=T, A=A)
Splits the data (an AbstractArray in root) to the processes. 
"""
function split_data(arr::AbstractArray; root=0, T=nothing, A=nothing)
    sz = MPI.bcast(size(arr), root=root, comm=MPI.COMM_WORLD)
    #TODO
end

"""
    sync(a::MPIArray)
Collective call, making sure all operations modifying any part of the array are finished when it completes
"""
sync(a::MPIArray, ::Vararg{MPIArray, N}) where N = MPI.Barrier(a.comm)

function Base.similar(a::MPIArray{T,N,A}, ::Type{T}, dims::NTuple{N,Int}) where {T,N,A}
    old_partition_sizes = partition_sizes(a.partitioning)
    old_dims = size(a)
    new_partition_sizes = Vector{Int}[]
    remaining_nb_partitons = prod(length.(old_partition_sizes))
    for i in eachindex(dims)
        if i <= length(old_dims)
            if dims[i] == old_dims[i]
                push!(new_partition_sizes, old_partition_sizes[i])
            else
                push!(new_partition_sizes, distribute(dims[i], length(old_partition_sizes[i])))
            end
        elseif remaining_nb_partitons != 1
            push!(new_partition_sizes, distribute(dims[i], remaining_nb_partitons))
        else
            push!(new_partition_sizes, [dims[i]])
        end
        @assert remaining_nb_partitons % length(last(new_partition_sizes)) == 0
        remaining_nb_partitons ÷= length(last(new_partition_sizes))
    end
    if remaining_nb_partitons > 1
        remaining_nb_partitons *= length(last(new_partition_sizes))
        new_partition_sizes[end] = distribute(dims[end], remaining_nb_partitons)
        remaining_nb_partitons ÷= length(last(new_partition_sizes))
    end
    @assert remaining_nb_partitons == 1
    return MPIArray{T,N,A}(a.comm, new_partition_sizes...)
end

function Base.filter(f,a::MPIArray)
    error("filter is only supported on 1D MPIArrays")
end

function Base.filter(f,a::MPIArray{T,1}) where T
    return MPIArray(forlocalpart(v -> filter(f,v), a), length(a.partitioning))
end

function Base.filter!(f,a::MPIArray)
    error("filter is only supported on 1D MPIArrays")
end

function copyto!(dest::MPIArray{T,N,A}, src::MPIArray{T,N,A}) where {T,N,A}
    @assert all(dest.sizes .== src.sizes)
    @assert dest.partitioning .== src.partitioning
    @assert dest.comm == src.comm
    # dest.win = src.win
    @assert dest.myrank == src.myrank

    dest.localarray .= src.localarray
    return dest
end

Base.filter!(f,a::MPIArray{T,1,A}) where {T,A} = copy_into!(a, filter(f,a))

function redistribute(a::MPIArray{T,N,A}, partition_sizes::Vararg{Any,N}) where {T,N,A}
    rank = MPI.Comm_rank(a.comm)
    @assert prod(length.(partition_sizes)) == MPI.Comm_size(a.comm)
    partitioning = ContinuousPartitioning(partition_sizes...)
    localarray = getblock(a[partitioning[rank+1]...])
    return MPIArray(a.comm, localarray, length.(partition_sizes)...)
end

function redistribute(a::MPIArray{T,N,A}, nb_parts::Vararg{Int,N}) where {T,N,A}
    return redistribute(a, distribute.(size(a), nb_parts)...)
end

function redistribute(a::MPIArray)
    return redistribute(a, size(a.partitioning)...)
end

redistribute!(a::MPIArray{T,N,A}, partition_sizes::Vararg{Any,N})  where {T,N,A} = copy_into!(a, redistribute(a, partition_sizes...))
redistribute!(a::MPIArray) = redistribute!(a, size(a.partitioning)...)

"""
    localindices(a::MPIArray, rank::Integer)
Get the local index range (expressed in terms of global indices) of the given rank
"""
@inline localindices(a::MPIArray, rank::Integer=a.myrank) = a.partitioning[rank+1]

"""
    forlocalpart(f, a::MPIArray)
Execute the function f on the part of a owned by the current rank. It is assumed f does not modify the local part.
"""
function forlocalpart(f, a::MPIArray)
    # MPI.Win_lock(MPI.LOCK_SHARED, a.myrank, 0, a.win)
    result = f(a.localarray)
    # MPI.Win_unlock(a.myrank, a.win)
    return result
end

"""
    forlocalpart(f, a::MPIArray)
Execute the function f on the part of a owned by the current rank. The local part may be modified by f.
"""
function forlocalpart!(f, As::Vararg{AbstractArray,N}) where N
    isMPIArray = x -> (typeof(x) <: MPIArray)
    for a in As
        isMPIArray(a) # && MPI.Win_lock(MPI.LOCK_EXCLUSIVE, a.myrank, 0, a.win)
    end
    result = f([isMPIArray(a) ? a.localarray : a for a in As]...)
    for a in As
        isMPIArray(a) # && MPI.Win_unlock(a.myrank, a.win)
    end
    return result
end

function linear_ranges(indexblock)
    cr = CartesianIndices(axes(indexblock)[2:end])
    result = Vector{UnitRange{Int}}(undef,length(cr))

    for (i,carti) in enumerate(cr)
        linrange = indexblock[:,carti]
        result[i] = linrange[1]:linrange[end]
    end
    return result
end

function free(a::MPIArray{T,N}) where {T,N}
    sync(a)
    # MPI.free(a.win)
end
