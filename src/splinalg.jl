import SparseArrays
import SparseArrays: SparseMatrixCSC, rowvals, nonzeros, nzrange
import Polyester: @batch

const MatOrView{T} = Union{Matrix{T}, SubArray{T, 2, Matrix{T}}}

function dspmm!(C::MatOrView{T}, A::MatOrView{T}, B::SparseMatrixCSC{T}) where T
    minbatch = max(size(B, 2) ÷ Threads.nthreads(), 1)
    @inbounds begin
        @batch minbatch = minbatch for j in axes(B, 2)
            C[:, j] = A * B[:,j]
        end
        return C
    end
end

function dgrad_update(S::SparseMatrixCSC{T}, G::MatOrView{T}, tau::Real) where T
    minbatch = max(size(S, 2) ÷ Threads.nthreads(), 1)
    C = Ref(zero(eltype(S))) .- tau * G
    rowinds, nzvals = rowvals(S), nonzeros(S)
    @inbounds begin
        @batch minbatch = minbatch for j in axes(S, 2)
            for i in nzrange(S, j)
                rowidx = rowinds[i]
                #C[rowidx, j] = nzvals[i] - tau * G[rowidx, j]
                C[rowidx, j] += nzvals[i]
            end
        end
    end
    return C
end

function dgrad_update!(C::Matrix{T}, S::SparseMatrixCSC{T}, G::MatOrView{T}, tau::Real) where T
    minbatch = max(size(S, 2) ÷ Threads.nthreads(), 1)
    C[:] = -tau * G
    rowinds, nzvals = rowvals(S), nonzeros(S)
    @inbounds begin
        @batch minbatch = minbatch for j in axes(S, 2)
            for i in nzrange(S, j)
                rowidx = rowinds[i]
                #C[rowidx, j] = nzvals[i] - tau * G[rowidx, j]
                C[rowidx, j] += nzvals[i]
            end
        end
    end
end