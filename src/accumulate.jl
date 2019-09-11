import Base: cumsum, cumsum!, cumprod, cumprod!
for (fname, fname!, baseop) in [(:cumsum, :cumsum!, :+), (:cumprod, :cumprod!, :*)]
    @eval begin
        function ($fname!)(r::MPIArray{T,N,A}, a::MPIArray{T,N,A}; dims=Int, tmp::Union{Nothing,A}=nothing) where {T,N,A}
            tmp = (tmp == nothing) ? A{T}(undef, size(a)[1:N-1]...) : tmp
            @assert length(tmp) >= prod(size(a)[1:N-1]) 
            ($fname!)(r.localarray, a.localarray; dims=dims)
            if dims == N
                for i in 0:Size()-2
                    if Rank() == i
                        sync()
                        Send(@view(r.localarray[repeat([:], N-1)..., end]), i + 1)
                    elseif Rank() == i + 1
                        sync()
                        Recv!(tmp, i)
                        r.localarray .= ($baseop).(r.localarray, tmp)
                    end
                end
            end
            r
        end
        function ($fname!)(r::MPIVector{T,A}, a::MPIVector{T,A}) where {T,A}
            ($fname!)(r, a; dims=1)
        end
        function ($fname)(a::MPIArray{T,N,A}; dims=Int) where {T,N,A}
            r = MPIArray{T,N,A}(undef, size(a)...)
            ($fname!)(r, a; dims=dims)
        end
        function ($fname)(a::MPIVector{T,A}) where {T,A}
            ($fname)(a; dims=1)
        end
    end
end
