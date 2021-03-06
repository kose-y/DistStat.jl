import Base: sum, prod, maximum, minimum, sum!, prod!, maximum!, minimum!
for (fname, _fname, fname!, mpiop) in [(:sum, :_sum, :sum!, :(MPI.SUM)), (:prod, :_prod, :prod!, :(MPI.PROD)), (:maximum,:_maximum, :maximum!, :(MPI.MAX)), (:minimum, :_minimum, :minimum!, :(MPI.MIN))]
    
    @eval begin
        function ($_fname)(f::Function, a::MPIArray{T,N,A}, ::Colon) where {T,N,A}
            partial = ($fname)(f, a.localarray)
            sync()
            MPI.Allreduce(partial, $mpiop, MPI.COMM_WORLD)
        end

        function ($fname!)(r::MPIArray{T,N,A}, arr::MPIArray{T,N,A}) where {T,N,A}
            for (dr, darr) in zip(size(r), size(arr))
                @assert dr == darr || dr == 1
            end
            @assert size(r)[end] == size(arr)[end]
            ($fname!)(r.localarray, arr.localarray)
            r
        end

        function ($fname!)(r::AbstractArray{T,N}, arr::MPIArray{T,N,A}) where {T,N,A}
            @assert size(r)[end] == 1
            ($fname!)(r, arr.localarray)
            sync()
            MPI.Allreduce!(r, $mpiop, MPI.COMM_WORLD)
            r
        end

        function ($_fname)(a::MPIArray{T,N,A}, dims::NTuple{N2, Integer}) where {T,N,A,N2}
            outsize = map(x -> x in dims ? 1 : size(a, x), 1:N)
            if N in dims
                out = A{T}(undef, outsize...)
            else
                out = similar(a, outsize...)
            end
            ($fname!)(out, a)
            out
        end

        ($_fname)(f::Function, a::MPIArray{T,N,A}, dims::Integer) where {T,N,A} = ($_fname)(f, a, (dims,))
        ($fname)(f::Function, a::MPIArray{T,N,A}; dims=:) where {T,N,A} = ($_fname)(f, a, dims)
        ($_fname)(a::MPIArray{T,N,A}, ::Colon) where {T,N,A} = ($_fname)(identity, a, :)
        ($_fname)(a::MPIArray{T,N,A}, dims::Integer) where {T,N,A} = ($_fname)(a, (dims,))
        ($fname)(a::MPIArray{T,N,A}; dims=:) where {T,N,A} = ($_fname)(a, dims)
    end
end
