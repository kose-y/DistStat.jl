import Base: fill!, rand, randn
import Random: seed!, rand!, randn!
import Adapt: adapt
export distribute

function fill!(a::MPIArray, x)
    forlocalpart!(y -> fill!(y, x), a) 
end

"""
    distribute(a::AbstractArray; root=0, A=nothing)
distributes a dense array.
"""
function distribute(a::AbstractArray{T,N}; root=0, A=nothing) where {T, N}
    if A == nothing
        TA = typeof(a)
        A = hasproperty(TA, :name) ? TA.name.wrapper : TA
    end
    reqs = MPI.Request[]
    size_a = MPI.bcast(size(a), root, MPI.COMM_WORLD)
    rslt = MPIArray{T,N,A}(undef, size_a...)
    if Rank() == root
        for r = 0:Size() - 1
            r == Rank() && continue
            sync()
            push!(reqs, Isend(@view(a[repeat([:], N-1)..., rslt.partitioning[r+1][end]]), r))
        end
        rslt.localarray .= @view(a[repeat([:], N-1)..., rslt.partitioning[Rank()+1][end]])
    else
        sync()
        push!(reqs, Irecv!(rslt.localarray, root))
    end
    for r in reqs
        MPI.Wait!(r)
    end
    rslt
end

"""
    distribute(a::AbstractSparseMatrix; root=0, A=Array)
distributes a sparse array into a dense MPIMatrix.
"""
function distribute(a::AbstractSparseMatrix; root=0, A=Array)
    T = eltype(a)
    reqs = MPI.Request[]
    size_a = MPI.bcast(size(a), root, MPI.COMM_WORLD)
    rslt = MPIMatrix{T,A}(undef, size_a[1], size_a[2])
    if Rank() == root
        for r = 0:Size() - 1
            r == Rank() && continue
            sync()
            push!(reqs, MPI.isend(a[:, rslt.partitioning[r+1][2]], r, 0, MPI.COMM_WORLD))
        end
        local_sparse = a[:, rslt.partitioning[Rank()+1][2]]
        rslt.localarray .= A(local_sparse)
    else
        sync()
        local_sparse = MPI.recv(root, 0, MPI.COMM_WORLD)[1]
        rslt.localarray .= A(local_sparse)
    end

    for r in reqs
        MPI.Wait!(r)
    end
    rslt
end

for (fname!, fname) in [(:rand!, :rand), (:randn!, :randn)]
    @eval begin
        function $fname!(a::MPIArray{T,N,A};seed=nothing, common_init=false, root=0) where {T,N,A}
            # for both local initialization and initialization from master, from Array{Float64} (for the sake of reproducibility)
            if seed ≠ nothing
                # Rank is added here in order to take different samples for each rank
                common_init ? seed!(seed) : seed!(seed + Rank())
            end
            if !common_init
                forlocalpart!(y -> $fname!(y), a)
            else
                tmp = MPIArray{Float64, N, Array}(undef, size(a)...)
                if Rank() == root
                    src = $fname(size(a)...) # Intentionally creating Array{Float64}
                else
                    src = zeros(zeros(Int, N)...)
                end
                tmp = distribute(src; root=root)
                forlocalpart!(tmp, a) do x, y
                    y .= adapt(A{T}, x)
                end

                a
            end
        end
    end
end
