import Base: fill!, rand, randn
import Random: seed!, rand!, randn!
import Adapt: adapt
function fill!(a::MPIArray, x)
    forlocalpart!(y -> fill!(y, x), a) 
end
function distribute(a::AbstractArray{T}) where {A,T}
    # TODO
end
for (fname!, fname) in [(:rand!, :rand), (:randn!, :randn)]
    @eval begin
        function fname!(a::MPIArray{T,N,A};seed=nothing, common_init=false, root=0) where {T,N,A}
            # for both local initialization and initialization from master, from Array{Float64} (for the sake of reproducibility)
            if seed ≠ nothing
                # Rank is added here in order to take different samples for each rank
                common_init ? seed!(seed) : seed!(seed + Rank())
            end
            if !common_init
                forlocalpart!(y -> fname!(y), a)
            else
                # TODO: factor this part into distribute
                reqs = MPI.Request[]
                if Rank() == root
                    src = fname(size(a)...) # Intentionally creating Array{Float64}
                    # isend each block
                    for r = 0:Size()-1
                        r == Rank() && continue
                        push!(reqs, Isend(src[a.partitioning[r+1]...], r+1))
                    end
                    # convert local block into proper type
                    a.localarray .= adapt(A{T}, src[a.partitioning[Rank()+1]...])
                else
                    # irecv each block
                    recvblock = Array{Float64}(undef, size(a.localarray)...)
                    push!(reqs, Irecv(recvblock, root))

                    # convert it into proper type
                    a.localarray .= adapt(A{T}, recvblock)
                end
                
                # wait for all the requests to end
                for r in reqs
                    MPI.Wait!(r)
                end
                a
            end
        end
    end
end
