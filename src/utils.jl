"""
    euclidean_distance(out, A, B)

Computes all pairwise distances between two sets of data, A(p x n) and B(p x m).
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

    mul!(out, transpose(A), B)
    out .= sqrt.((-2out .+ transpose(tmp_m)) .+ tmp_n)
end

"""
    euclidean_distance!(out, A)

Computes all pairwise distances between data in A (p x [n]). Output: n x [n].
"""
function euclidean_distance!(out::MPIMatrix{T,N,A}, data::MPIMatrix{T,N,A}, verbose=False; tmp_big::Union{A, Nothing}=nothing, tmp_small::Union{A, Nothing}) where {T,N,A}
    p, n = size(A)
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
        Bcast!(other; root=i)
        
        println(out.partitioning[r+1])
        #euclidean_distance!(out.localarray[out.partitioning[r][2]..., :], other, this)
    end
    out
end
