mpirun -np 1 julia --project=./cuda mds.jl --rows=10000 --cols=10000 --r=20 --iter=10000 --gpu --Float32 --init_from_master > rslts/mds-20-1gpu.txt
mpirun -np 2 julia --project=./cuda mds.jl --rows=10000 --cols=10000 --r=20 --iter=10000 --gpu --Float32 --init_from_master > rslts/mds-20-2gpu.txt
mpirun -np 3 julia --project=./cuda mds.jl --rows=10000 --cols=10000 --r=20 --iter=10000 --gpu --Float32 --init_from_master > rslts/mds-20-3gpu.txt
mpirun -np 4 julia --project=./cuda mds.jl --rows=10000 --cols=10000 --r=20 --iter=10000 --gpu --Float32 --init_from_master > rslts/mds-20-4gpu.txt
mpirun -np 5 julia --project=./cuda mds.jl --rows=10000 --cols=10000 --r=20 --iter=10000 --gpu --Float32 --init_from_master > rslts/mds-20-5gpu.txt
mpirun -np 6 julia --project=./cuda mds.jl --rows=10000 --cols=10000 --r=20 --iter=10000 --gpu --Float32 --init_from_master > rslts/mds-20-6gpu.txt
mpirun -np 7 julia --project=./cuda mds.jl --rows=10000 --cols=10000 --r=20 --iter=10000 --gpu --Float32 --init_from_master > rslts/mds-20-7gpu.txt
mpirun -np 8 julia --project=./cuda mds.jl --rows=10000 --cols=10000 --r=20 --iter=10000 --gpu --Float32 --init_from_master > rslts/mds-20-8gpu.txt

mpirun -np 1 julia --project=./cuda mds.jl --rows=10000 --cols=10000 --r=40 --iter=10000 --gpu --Float32 --init_from_master > rslts/mds-40-1gpu.txt
mpirun -np 2 julia --project=./cuda mds.jl --rows=10000 --cols=10000 --r=40 --iter=10000 --gpu --Float32 --init_from_master > rslts/mds-40-2gpu.txt
mpirun -np 3 julia --project=./cuda mds.jl --rows=10000 --cols=10000 --r=40 --iter=10000 --gpu --Float32 --init_from_master > rslts/mds-40-3gpu.txt
mpirun -np 4 julia --project=./cuda mds.jl --rows=10000 --cols=10000 --r=40 --iter=10000 --gpu --Float32 --init_from_master > rslts/mds-40-4gpu.txt
mpirun -np 5 julia --project=./cuda mds.jl --rows=10000 --cols=10000 --r=40 --iter=10000 --gpu --Float32 --init_from_master > rslts/mds-40-5gpu.txt
mpirun -np 6 julia --project=./cuda mds.jl --rows=10000 --cols=10000 --r=40 --iter=10000 --gpu --Float32 --init_from_master > rslts/mds-40-6gpu.txt
mpirun -np 7 julia --project=./cuda mds.jl --rows=10000 --cols=10000 --r=40 --iter=10000 --gpu --Float32 --init_from_master > rslts/mds-40-7gpu.txt
mpirun -np 8 julia --project=./cuda mds.jl --rows=10000 --cols=10000 --r=40 --iter=10000 --gpu --Float32 --init_from_master > rslts/mds-40-8gpu.txt

mpirun -np 1 julia --project=./cuda mds.jl --rows=10000 --cols=10000 --r=60 --iter=10000 --gpu --Float32 --init_from_master > rslts/mds-60-1gpu.txt
mpirun -np 2 julia --project=./cuda mds.jl --rows=10000 --cols=10000 --r=60 --iter=10000 --gpu --Float32 --init_from_master > rslts/mds-60-2gpu.txt
mpirun -np 3 julia --project=./cuda mds.jl --rows=10000 --cols=10000 --r=60 --iter=10000 --gpu --Float32 --init_from_master > rslts/mds-60-3gpu.txt
mpirun -np 4 julia --project=./cuda mds.jl --rows=10000 --cols=10000 --r=60 --iter=10000 --gpu --Float32 --init_from_master > rslts/mds-60-4gpu.txt
mpirun -np 5 julia --project=./cuda mds.jl --rows=10000 --cols=10000 --r=60 --iter=10000 --gpu --Float32 --init_from_master > rslts/mds-60-5gpu.txt
mpirun -np 6 julia --project=./cuda mds.jl --rows=10000 --cols=10000 --r=60 --iter=10000 --gpu --Float32 --init_from_master > rslts/mds-60-6gpu.txt
mpirun -np 7 julia --project=./cuda mds.jl --rows=10000 --cols=10000 --r=60 --iter=10000 --gpu --Float32 --init_from_master > rslts/mds-60-7gpu.txt
mpirun -np 8 julia --project=./cuda mds.jl --rows=10000 --cols=10000 --r=60 --iter=10000 --gpu --Float32 --init_from_master > rslts/mds-60-8gpu.txt
