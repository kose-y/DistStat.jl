import NPZ
import MPI
import MPI: COMM_WORLD
export npyread

function npyread(filename::AbstractString; root=0, A=Array)
    success = 0
    shape = 0
    T = nothing
    hdrend = 0
    toh = nothing

    if Rank() == root
        f = open(filename, "r")
        b = read!(f, Vector{UInt8}(undef, NPZ.MaxMagicLen))
        if NPZ.samestart(b, NPZ.NPYMagic)
            seekstart(f)
            hdr = NPZ.readheader(f)
            if hdr.fortran_order
                shape = hdr.shape
                T = eltype(hdr)
                toh = hdr.descr
                hdrend = mark(f)

                success = 1
                close(f)
            else
                success = -1
                close(f)
            end
        else
            close(f)
        end
    end
    success = MPI.bcast(success, root, MPI.COMM_WORLD)
    if success == 0
        error("not a NPY file supported: $filename")
    end
    if success == -1
        error("NPY file must be in fortran order: $filename")
    end
    shape, T, hdrend, toh = MPI.bcast((shape, T, hdrend, toh), root, MPI.COMM_WORLD)

    rslt = MPIArray{T, length(shape), A}(undef, shape...)
    arr_skip = sum(rslt.local_lengths[1:Rank()])
    
    f = open(filename, "r")
    seek(f, hdrend + arr_skip * sizeof(T))
    rslt.localarray = map(toh, read!(f, Array{T}(undef, size(rslt.localarray))))
    close(f)
    rslt
end