using .CUDAnative
using .CuArrays
using .CuArrays.CUDAdrv

function set_device!()
    lcomm = MPI.Comm_split_type(COMM_WORLD, MPI.MPI_COMM_TYPE_SHARED, MPI.Comm_rank(COMM_WORLD))
    CUDAnative.device!(MPI.Comm_rank(lcomm) % length(devices()))
end

function sync()
    synchronize()
end
