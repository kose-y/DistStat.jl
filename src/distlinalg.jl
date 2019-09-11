import LinearAlgebra: Transpose
import LinearAlgebra
export fill_diag!

# temporary function. will actually add CUDA barrier function.
function sync()
    nothing 
end

"""
    Allgather_opt(B)
Returns a tuple of proper function and argument for Allgather.
"""
function Allgather_opt(B::MPIVecOrMat{T,AT}) where {T,AT}
    if allsame(B.local_lengths)
        (Allgather!, B.localarray, B.local_lengths[1])
    else
        (Allgatherv!, B.localarray, B.local_lengths)
    end
end

function Allgather_opt(B::Transpose{T,ATT} where ATT <: MPIVecOrMat{T,AT}) where {T,AT}
    B_buf = transpose(B).localarray
    tB_local_lengths = transpose(B).local_lengths
    if allsame(tB_local_lengths)
        (Allgather!, B_buf, tB_local_lengths[1])
    else
        (Allgatherv!, B_buf, tB_local_lengths)
    end
end

@inline get_local(A::MPIVecOrMat{T,AT}) where {T,AT} = A.localarray
@inline get_local(A::Transpose{T,ATT} where ATT <: MPIVecOrMat{T,AT}) where {T,AT} = transpose(transpose(A).localarray)

"""
Scenario 1: inner product, result distributed, temporary space required
A: r x [p], B: [p] x q, C: r x [q]
tmp: r x q, for local computation
"""
function LinearAlgebra.mul!(C::MPIMatrix{T,AT}, A::MPIMatrix{T,AT}, B::Transpose{T, MPIMatrix{T,AT}}; tmp::AbstractArray{T,2}=AT{T}(undef, size(A,1), size(B,2))) where {T,AT}
    @assert size(C,1) == size(A,1) && size(C,2) == size(B,2)
    @assert size(A,1) == size(tmp,1) && size(B,2) == size(tmp,2)
    localA = get_local(A)
    localB = get_local(B)
    LinearAlgebra.mul!(tmp, localA, localB)
    # Ireduce would fit well here... but is not supported on both PyTorch and MPI.jl.
    sync()
    for i = 0:Size()-1
        # NOTE: an array is set to be contiguous only if the first indices are colon.
        Reduce!(@view(tmp[:, C.partitioning[i+1][2]]), C.localarray; root=i)
    end
    C
end

function LinearAlgebra.mul!(C::Transpose{T,MPIMatrix{T,AT}}, A::MPIMatrix{T,AT},B::Transpose{T,MPIMatrix{T,AT}}; tmp::AbstractArray{T,2}=AT{T}(undef, size(B,2), size(A,1))) where {T,AT}
    LinearAlgebra.mul!(transpose(C), transpose(B), transpose(A);tmp=tmp)
end

"""
Scenario 2: short and fat matrix multiplied by a fat matrix. More communication than Scenario 1, temporary space required
A: r x [p], B: p x [q], C: r x [q]
tmp: r x p: A is Allgather!-ed.
"""
function LinearAlgebra.mul!(C::MPIMatrix{T,AT}, A::MPIMatrix{T,AT}, B::MPIMatrix{T,AT}; tmp::AbstractArray{T,2}=AT{T}(undef, size(A,1), size(A,2))) where {T,AT}
    @assert size(C,1) == size(A,1) && size(C,2) == size(B,2)
    @assert size(A,1) == size(tmp,1) && size(A, 2) == size(tmp, 2)
    localA = get_local(A)
    localB = get_local(B)
    localC = get_local(C)
    
    #Allgather
    sync()
    (Allgather_ftn!, sendbuf, count_arg) = Allgather_opt(A)
    Allgather_ftn!(sendbuf, tmp, count_arg)

    LinearAlgebra.mul!(localC, tmp, localB)
    C
end

function LinearAlgebra.mul!(C::Transpose{T,MPIMatrix{T,AT}}, A::Transpose{T,MPIMatrix{T,AT}}, B::Transpose{T,MPIMatrix{T,AT}}; tmp::AbstractArray{T,2}=AT{T}(undef, size(B,2), size(B,1))) where {T,AT}
    LinearAlgebra.mul!(transpose(C), transpose(B), transpose(A); tmp=tmp)
end


"""
Scenario 3: inner product, result broadcasted
"""
function LinearAlgebra.mul!(C::AbstractMatrix{T}, A::MPIMatrix{T,AT}, B::Transpose{T,MPIMatrix{T,AT}}) where {T,AT}
    @assert size(C, 1) == size(A, 1) && size(C, 2) == size(B, 2)
    localA = get_local(A)
    localB = get_local(B)
    LinearAlgebra.mul!(C, localA, localB)
    sync()
    Allreduce!(C)
    C
end

"""
Scenario 4: outer product, temporary space required
transpose(A) is Allgather!-ed.
"""
function LinearAlgebra.mul!(C::MPIMatrix{T,AT}, A::Transpose{T,MPIMatrix{T,AT}}, B::MPIMatrix{T,AT}; tmp::AbstractArray{T,2}=AT{T}(undef,size(A,2),size(A,1))) where {T,AT}
    @assert size(C,1) == size(A,1) && size(C,2) == size(B,2)
    @assert size(A,2) == size(tmp,1) && size(A,1) == size(tmp,2)
    localA = get_local(A)
    localB = get_local(B)
    localC = get_local(C)
    
    # Allgather
    sync()
    (Allgather_ftn!, sendbuf, count_arg) = Allgather_opt(A)
    Allgather_ftn!(sendbuf, tmp, count_arg)

    LinearAlgebra.mul!(localC, transpose(tmp), localB)
    C
end

function LinearAlgebra.mul!(C::Transpose{T,MPIMatrix{T,AT}}, A::Transpose{T,MPIMatrix{T,AT}}, B::MPIMatrix{T,AT}; tmp::AbstractArray{T,2}=AT{T}(undef,size(B,1), size(B,2))) where {T,AT}
    LinearAlgebra.mul!(transpose(C), transpose(B), transpose(A);tmp=tmp)
end

"""
Scenario 5: Small, broadcasted matrix multiplied by a distributed matrix.
A: s x r, B: r x [q], C: s x [q].
"""
function LinearAlgebra.mul!(C::MPIMatrix{T,AT}, A::Union{AbstractMatrix{T}}, B::MPIMatrix{T,AT}) where {T,AT}
    localB = get_local(B)
    localC = get_local(C)
    LinearAlgebra.mul!(localC, A, localB)
    C
end
function LinearAlgebra.mul!(C::MPIMatrix{T,AT}, A::Transpose{T,ATT} where ATT <: AbstractMatrix{T}, B::MPIMatrix{T,AT}) where {T,AT}
    localB = get_local(B)
    localC = get_local(C)
    LinearAlgebra.mul!(localC, A, localB)
    C
end

function LinearAlgebra.mul!(C::Transpose{T,MPIMatrix{T,AT}}, A::Transpose{T,MPIMatrix{T,AT}}, B::Transpose{T, ATT} where ATT <: AbstractMatrix{T}) where {T,AT}
    LinearAlgebra.mul!(transpose(C), transpose(B), transpose(A))
end

function LinearAlgebra.mul!(C::Transpose{T,MPIMatrix{T,AT}}, A::Transpose{T,MPIMatrix{T,AT}}, B::AbstractMatrix{T}) where {T,AT}
    LinearAlgebra.mul!(transpose(C), transpose(B), transpose(A))
end

"""
Scenario 6: distributed matrix x broadcasted vector multiplications
"""

"""
6.1: A: r x [q], B: q x 1, C: r x 1
"""
function LinearAlgebra.mul!(C::AbstractVector{T}, A::MPIMatrix{T,AT}, B::AbstractVector{T}) where {T,AT}
    localA = get_local(A)
    LinearAlgebra.mul!(C, localA, B[A.partitioning[Rank()+1][2]])
    sync()
    Allreduce!(C)
    C
end

"""
6.2: A: r x [q], B: [q] x 1, C: r x 1
"""
function LinearAlgebra.mul!(C::AbstractVector{T}, A::MPIMatrix{T,AT}, B::MPIVector{T,AT}) where {T,AT}
    localA = get_local(A)
    localB = get_local(B)
    LinearAlgebra.mul!(C, localA, localB)
    sync()
    Allreduce!(C)
    C
end

"""
6.3: A: [p] x q, B: [q] x 1, C: [p] x 1
"""
function LinearAlgebra.mul!(C::MPIVector{T,AT}, A::Transpose{T,MPIMatrix{T,AT}},B::MPIVector{T,AT};tmp::AbstractArray{T}=AT{T}(undef, size(B,1))) where {T,AT}
    localA = get_local(A)
    localC = get_local(C)

    # Allgather
    sync()
    (Allgather_ftn!, sendbuf, count_arg) = Allgather_opt(B)
    Allgather_ftn!(sendbuf, tmp, count_arg)

    LinearAlgebra.mul!(localC, localA, tmp)
    C
end

"""
6.4: A: [p] x r, B: r x 1, C: [p] x 1
"""
function LinearAlgebra.mul!(C::MPIVector{T,AT}, A::Transpose{T,MPIMatrix{T,AT}}, B::AbstractVector{T}) where {T,AT}
    localA = get_local(A)
    localC = get_local(C)
    LinearAlgebra.mul!(localC, localA, B)
    C
end



"""
    LinearAlgebra.diagind(M::MPIMatrix, k)

returns indices of the diagonal with respect to M.localarray. 
"""
@inline function LinearAlgebra.diagind(M::MPIMatrix, k::Integer=0)
    offset = - (M.partitioning[Rank()+1][2][1]-1) + k
    return LinearAlgebra.diagind(size(M,1), size(M.localarray,2), offset)
end

"""
    LinearAlgebra.diag(M::MPIMatrix{T,A}, k; dist::Bool=false) where {T,A}

returns the diagonal of M. If dist is false, the result is a broadcasted A{T,1}. otherwise, the result is distributed 1 x n "row vector". Distributed return is only valid for a square matrix (due to current way of distribution). 
"""
function LinearAlgebra.diag(M::MPIMatrix{T,A}, k::Integer=0; dist::Bool=false) where {T,A}
    # TODO
end

"""
    fill_diag!(M, x, k=0)

fills the diagonal of M with x.
"""
@inline function fill_diag!(M::MPIMatrix, x, k::Integer=0)
    M.localarray[LinearAlgebra.diagind(M, k)] .= x
end
