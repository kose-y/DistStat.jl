for (fname, _fname, fname!, mpiop) in [(:sum, :_sum, :sum!, :(MPI.SUM)), (:prod, :_prod, :prod!, :(MPI.PROD)), (:maximum,:_maximum, :maximum! :(MPI.MAX)), (:minimum, :_minimum, :minimum!, :(MPI.MIN))]
    @eval begin
        function ($_fname)(a::MPIArray{T,N,A}, ::Colon) where {T,N,A}
            partial = ($fname)(a.localarray)
            MPI.Allreduce(partial, mpiop, MPI.COMM_WORLD)
        end
        function ($_fname)(a::MPIArray{T,N,A}, dims::NTuple{N2, Integer}) where {T,N,N2,A}
            # reduce all other dimensions
            
            if N in dims
                # reduce over the last (distributed) dimension
            end
        end
        ($_fname)(a::MPIArray{T,N,A}, dims::Integer) = ($_fname)(a, (dims,)) where {T,N,A}
        function ($fname!)(r, a::MPIArray{T,N,A}) where {T,N,A}
            # TODO
        end

        
        function ($_fname)(f::Function, a::MPIArray{T,N,A}, ::Colon) where {T,N,A}
            partial = ($fname)(f, a.localarray)
            MPI.Allreduce(partial, mpiop, MPI.COMM_WORLD)
        end
        ($_fname)(f::Function, a::MPIArray{T,N,A}, dims::NTuple{N, Integer}) where {T,N,A}
            # reduce over all other dimensions
            if N in dims
                # reduce over the last (distributed) dimension
            end
        end
        ($_fname)(f::Function, a::MPIArray{T,N,A}, dims::Integer) where {T,N,A} = ($_fname)(a, (dims,))
        @inline ($fname)(a::MPIArray{T,N,A}; dims=:) where {T,N,A} = ($_fname)(a, dims)
        @inline ($fname)(f::Function, a::MPIArray{T,N,A}; dims=:) where {T,N,A} = ($_fname)(f, a, dims)
    end
end
