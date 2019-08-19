import LinearAlgebra: Transpose
import LinearAlgebra

# temporary function. will actually add CUDA barrier function.
function sync()
    nothing 
end

"""
    Allgather_opt(B)
Returns a tuple of proper function and argument for Allgather.
"""
function Allgather_opt(B::MPIArray{T,2,AT}) where {T,AT}
    if allsame(B.local_lengths)
        (Allgather!, B.localarray, B.local_lengths[1])
    else
        (Allgatherv!, B.localarray, B.local_lengths)
    end
end

function Allgather_opt(B::Transpose{T,MPIArray{T,2,AT}}) where {T,AT}
    B_buf = transpose(B).localarray
    tB_local_lengths = transpose(B).local_lengths
    if allsame(tB_local_lengths)
        (Allgather!, B_buf, tB_local_lengths[1])
    else
        (Allgatherv!, B_buf, tB_local_lengths)
    end
end

@inline get_local(A::MPIArray{T,2,AT}) where {T,AT} = A.localarray
@inline get_local(A::Transpose{T,MPIArray{T,2,AT}}) where {T,AT} = transpose(transpose(A).localarray)

"""
Scenario 1: inner product, result distributed, temporary space required
A: r x [p], B: [p] x q, C: r x [q]
tmp: r x q, for local computation
"""
function LinearAlgebra.mul!(C::MPIArray{T,2,AT}, A::MPIArray{T,2,AT}, B::Transpose{T, MPIArray{T,2,AT}}; tmp::AbstractArray{T,2}=AT{T}(undef, size(A,1), size(B,2))) where {T,AT}
    @assert size(C,1) == size(A,1) && size(C,2) == size(B,2)
    @assert size(A,1) == size(tmp,1) && size(B,2) == size(tmp,2)
    localA = get_local(A)
    localB = get_local(B)
    LinearAlgebra.mul!(tmp, localA, localB)
    # Ireduce would fit well here... but is not supported on both PyTorch and MPI.jl.
    sync()
    for i = 0:Size()-1
        Reduce!(tmp[C.partitioning[i+1]...], C.localarray; root=i)
    end
end

function LinearAlgebra.mul!(C::Transpose{T,MPIArray{T,2,AT}}, A::MPIArray{T,2,AT},B::Transpose{T,MPIArray{T,2,AT}}; tmp::AbstractArray{T,2}=AT{T}(undef, size(B,2), size(A,1))) where {T,AT}
    LinearAlgebra.mul!(transpose(C), transpose(B), transpose(A);tmp=tmp)
end

"""
Scenario 2: short and fat matrix multiplied by a fat matrix. More communication than Scenario 1, temporary space required
A: r x [p], B: p x [q], C: r x [q]
tmp: r x p: A is Allgather!-ed.
"""
function LinearAlgebra.mul!(C::MPIArray{T,2,AT}, A::MPIArray{T,2,AT}, B::MPIArray{T,2,AT}; tmp::AbstractArray{T,2}=AT{T}(undef, size(A,1), size(A,2))) where {A,AT}
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
end

function LinearAlgebra.mul!(C::Transpose{T,MPIArray{T,2,AT}}, A::Transpose{T,MPIArray{T,2,AT}}, B::Transpose{T,MPIArray{T,2,AT}}; tmp::AbstractArray{T,2}=AT{T}(undef, size(B,2), size(B,1))) where {A,AT}
    LinearAlgebra.mul!(transpose(C), transpose(B), transpose(A); tmp=tmp)
end


"""
Scenario 3: inner product, result broadcasted
"""
function LinearAlgebra.mul!(C::AbstractArray{T,2}, A::MPIArray{T,2,AT}, B::Transpose{T,MPIArray{T,2,AT}}) where {T,AT}
    @assert size(C, 1) == size(A, 1) && size(C, 2) == size(B, 2)
    localA = get_local(A)
    localB = get_local(B)
    LinearAlgebra.mul!(C, localA, localB)
    sync()
    Allreduce!(C)
end

"""
Scenario 4: outer product, temporary space required
transpose(A) is Allgather!-ed.
"""
function LinearAlgebra.mul!(C::MPIArray{T,2,AT}, A::Transpose{T,MPIArray{T,2,AT}}, B::MPIArray{T,2,AT}; tmp::AbstractArray{T,2}=AT{T}(undef,size(A,2),size(A,1))) where {T,AT}
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
end

function LinearAlgebra.mul!(C::Transpose{T,MPIArray{T,2,AT}}, A::Transpose{T,MPIArray{T,2,AT}}, B::MPIArray{T,2,AT}; tmp::AbstractArray{T,2}=AT{T}(undef,size(B,1), size(B,2))) where {T,AT}
    LinearAlgebra.mul!(transpose(C), transpose(B), transpose(A);tmp=tmp)
end

"""
Scenario 5: Small, broadcasted matrix multiplied by a distributed matrix.
A: s x r, B: r x [q], C: s x [q].
"""
function LinearAlgebra.mul!(C::MPIArray{T,2,AT}, A::AbstractArray{T,2}, B::MPIArray{T,2,AT}) where {T,AT}
    @assert size(C,1) == size(A,1) && size(C,2) == size(B,2)
    localB = get_local(B)
    localC = get_local(C)
    LinearAlgebra.mul!(localC, A, localB)
end

function LinearAlgebra.mul!(C::Transpose{T,MPIArray{T,2,AT}}, A::Transpose{T,MPIArray{T,2,AT}}, B::AbstractArray{T,2}) where {T,AT}
    LinearAlgebra.mul!(transpose(C), transpose(B), transpose(A))
end

"""
Scenario 6: distributed matrix x broadcasted vector multiplications
"""
