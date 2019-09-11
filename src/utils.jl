import LinearAlgebra: diagind
"""
    euclidean_distance(out, A, B; tmp_m=nothing, tmp_n=nothing)

Computes all pairwise distances between two sets of data, A(p x n) and B(p x m).
temp memory: length m, length n
Output: n x m.
"""
function euclidean_distance!(out::AT, A::AT, B::AT; tmp_n::Union{AT, Nothing}=nothing, tmp_m::Union{AT, Nothing}=nothing) where AT <: AbstractArray
    ATN =  hasproperty(AT, :name) ? AT.name.wrapper : AT

    p, n = size(A)
    p2, m = size(B)
    @assert p == p2
    @assert size(out) == (n, m)
    T = eltype(A)
    
    tmp_n = (tmp_n == nothing) ? ATN{T}(undef, n) : tmp_n
    @assert length(tmp_n) == n 
    tmp_m = (tmp_m == nothing) ? ATN{T}(undef, m) : tmp_m
    @assert length(tmp_m) == m
    
    tmp_n .= reshape(sum(A.^2; dims=1), n)
    tmp_m .= reshape(sum(B.^2; dims=1), m)

    LinearAlgebra.mul!(out, transpose(A), B)
    out .= sqrt.(max.((-2out .+ transpose(tmp_m)) .+ tmp_n, zero(T)))

    if A==B
        out[diagind(out)].= zero(T)
    end
    out
end

"""
    euclidean_distance!(out, A)

Computes all pairwise distances between data in A (p x [n]). Output: n x [n].
"""
function euclidean_distance!(out::MPIMatrix{T,A}, data::MPIMatrix{T,A}; tmp_big::Union{A, Nothing}=nothing, tmp_small::Union{A, Nothing}=nothing) where {T,A}
    p, n = size(data)
    @assert size(out) == (n, n)

    local_len = n ÷ Size()
    remainder = n % Size()

    if remainder != 0
        tmp_big = (tmp_big == nothing) ? A{T}(undef, n, local_len+1) : tmp_big
        @assert size(tmp_big) == (n, local_len + 1)
    end
    tmp_small = (tmp_small == nothing) ? A{T}(undef, n, local_len) : tmp_small
    @assert size(tmp_small) == (n, local_len)

    for r in 0:Size()-1
        this = data.localarray
        other = (r == Rank()) ? data.localarray : 
                        (r < remainder ? tmp_big : tmp_small)
        Bcast!(other; root=r)
        
        #println(out.partitioning[r+1])
        tmpout = A{T}(undef, length(out.partitioning[r+1][2]), size(out.localarray,2))

        euclidean_distance!(tmpout, other, this) 
        out.localarray[out.partitioning[r+1][2], :] .= tmpout# expected to be slow in GPU
    end
    out
end
