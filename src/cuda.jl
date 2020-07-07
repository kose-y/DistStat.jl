using .CUDA

function set_device!()
    lcomm = MPI.Comm_split_type(COMM_WORLD, MPI.MPI_COMM_TYPE_SHARED, MPI.Comm_rank(COMM_WORLD))
    CUDA.device!(MPI.Comm_rank(lcomm) % length(devices()))
end

function sync()
    synchronize()
end
