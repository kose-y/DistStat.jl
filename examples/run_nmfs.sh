mpirun -np 1 julia --project=./cuda nmf-apg.jl --rows=10000 --cols=10000 --r=20 --iter=10000 --gpu --Float32 --init_from_master > rslts/nmf-apg-20-1gpu.txt
mpirun -np 2 julia --project=./cuda nmf-apg.jl --rows=10000 --cols=10000 --r=20 --iter=10000 --gpu --Float32 --init_from_master > rslts/nmf-apg-20-2gpu.txt
mpirun -np 3 julia --project=./cuda nmf-apg.jl --rows=10000 --cols=10000 --r=20 --iter=10000 --gpu --Float32 --init_from_master > rslts/nmf-apg-20-3gpu.txt
mpirun -np 4 julia --project=./cuda nmf-apg.jl --rows=10000 --cols=10000 --r=20 --iter=10000 --gpu --Float32 --init_from_master > rslts/nmf-apg-20-4gpu.txt
mpirun -np 5 julia --project=./cuda nmf-apg.jl --rows=10000 --cols=10000 --r=20 --iter=10000 --gpu --Float32 --init_from_master > rslts/nmf-apg-20-5gpu.txt
mpirun -np 6 julia --project=./cuda nmf-apg.jl --rows=10000 --cols=10000 --r=20 --iter=10000 --gpu --Float32 --init_from_master > rslts/nmf-apg-20-6gpu.txt
mpirun -np 7 julia --project=./cuda nmf-apg.jl --rows=10000 --cols=10000 --r=20 --iter=10000 --gpu --Float32 --init_from_master > rslts/nmf-apg-20-7gpu.txt
mpirun -np 8 julia --project=./cuda nmf-apg.jl --rows=10000 --cols=10000 --r=20 --iter=10000 --gpu --Float32 --init_from_master > rslts/nmf-apg-20-8gpu.txt

mpirun -np 1 julia --project=./cuda nmf-apg.jl --rows=10000 --cols=10000 --r=40 --iter=10000 --gpu --Float32 --init_from_master > rslts/nmf-apg-40-1gpu.txt
mpirun -np 2 julia --project=./cuda nmf-apg.jl --rows=10000 --cols=10000 --r=40 --iter=10000 --gpu --Float32 --init_from_master > rslts/nmf-apg-40-2gpu.txt
mpirun -np 3 julia --project=./cuda nmf-apg.jl --rows=10000 --cols=10000 --r=40 --iter=10000 --gpu --Float32 --init_from_master > rslts/nmf-apg-40-3gpu.txt
mpirun -np 4 julia --project=./cuda nmf-apg.jl --rows=10000 --cols=10000 --r=40 --iter=10000 --gpu --Float32 --init_from_master > rslts/nmf-apg-40-4gpu.txt
mpirun -np 5 julia --project=./cuda nmf-apg.jl --rows=10000 --cols=10000 --r=40 --iter=10000 --gpu --Float32 --init_from_master > rslts/nmf-apg-40-5gpu.txt
mpirun -np 6 julia --project=./cuda nmf-apg.jl --rows=10000 --cols=10000 --r=40 --iter=10000 --gpu --Float32 --init_from_master > rslts/nmf-apg-40-6gpu.txt
mpirun -np 7 julia --project=./cuda nmf-apg.jl --rows=10000 --cols=10000 --r=40 --iter=10000 --gpu --Float32 --init_from_master > rslts/nmf-apg-40-7gpu.txt
mpirun -np 8 julia --project=./cuda nmf-apg.jl --rows=10000 --cols=10000 --r=40 --iter=10000 --gpu --Float32 --init_from_master > rslts/nmf-apg-40-8gpu.txt

mpirun -np 1 julia --project=./cuda nmf-apg.jl --rows=10000 --cols=10000 --r=60 --iter=10000 --gpu --Float32 --init_from_master > rslts/nmf-apg-60-1gpu.txt
mpirun -np 2 julia --project=./cuda nmf-apg.jl --rows=10000 --cols=10000 --r=60 --iter=10000 --gpu --Float32 --init_from_master > rslts/nmf-apg-60-2gpu.txt
mpirun -np 3 julia --project=./cuda nmf-apg.jl --rows=10000 --cols=10000 --r=60 --iter=10000 --gpu --Float32 --init_from_master > rslts/nmf-apg-60-3gpu.txt
mpirun -np 4 julia --project=./cuda nmf-apg.jl --rows=10000 --cols=10000 --r=60 --iter=10000 --gpu --Float32 --init_from_master > rslts/nmf-apg-60-4gpu.txt
mpirun -np 5 julia --project=./cuda nmf-apg.jl --rows=10000 --cols=10000 --r=60 --iter=10000 --gpu --Float32 --init_from_master > rslts/nmf-apg-60-5gpu.txt
mpirun -np 6 julia --project=./cuda nmf-apg.jl --rows=10000 --cols=10000 --r=60 --iter=10000 --gpu --Float32 --init_from_master > rslts/nmf-apg-60-6gpu.txt
mpirun -np 7 julia --project=./cuda nmf-apg.jl --rows=10000 --cols=10000 --r=60 --iter=10000 --gpu --Float32 --init_from_master > rslts/nmf-apg-60-7gpu.txt
mpirun -np 8 julia --project=./cuda nmf-apg.jl --rows=10000 --cols=10000 --r=60 --iter=10000 --gpu --Float32 --init_from_master > rslts/nmf-apg-60-8gpu.txt

mpirun -np 1 julia --project=./cuda nmf-mult.jl --rows=10000 --cols=10000 --r=20 --iter=10000 --gpu --Float32 --init_from_master  > rslts/nmf-mult-20-1gpu.txt
mpirun -np 2 julia --project=./cuda nmf-mult.jl --rows=10000 --cols=10000 --r=20 --iter=10000 --gpu --Float32 --init_from_master  > rslts/nmf-mult-20-2gpu.txt
mpirun -np 3 julia --project=./cuda nmf-mult.jl --rows=10000 --cols=10000 --r=20 --iter=10000 --gpu --Float32 --init_from_master  > rslts/nmf-mult-20-3gpu.txt
mpirun -np 4 julia --project=./cuda nmf-mult.jl --rows=10000 --cols=10000 --r=20 --iter=10000 --gpu --Float32 --init_from_master  > rslts/nmf-mult-20-4gpu.txt
mpirun -np 5 julia --project=./cuda nmf-mult.jl --rows=10000 --cols=10000 --r=20 --iter=10000 --gpu --Float32 --init_from_master > rslts/nmf-mult-20-5gpu.txt
mpirun -np 6 julia --project=./cuda nmf-mult.jl --rows=10000 --cols=10000 --r=20 --iter=10000 --gpu --Float32 --init_from_master > rslts/nmf-mult-20-6gpu.txt
mpirun -np 7 julia --project=./cuda nmf-mult.jl --rows=10000 --cols=10000 --r=20 --iter=10000 --gpu --Float32 --init_from_master > rslts/nmf-mult-20-7gpu.txt
mpirun -np 8 julia --project=./cuda nmf-mult.jl --rows=10000 --cols=10000 --r=20 --iter=10000 --gpu --Float32 --init_from_master > rslts/nmf-mult-20-8gpu.txt

mpirun -np 1 julia --project=./cuda nmf-mult.jl --rows=10000 --cols=10000 --r=40 --iter=10000 --gpu --Float32 --init_from_master > rslts/nmf-mult-40-1gpu.txt
mpirun -np 2 julia --project=./cuda nmf-mult.jl --rows=10000 --cols=10000 --r=40 --iter=10000 --gpu --Float32 --init_from_master > rslts/nmf-mult-40-2gpu.txt
mpirun -np 3 julia --project=./cuda nmf-mult.jl --rows=10000 --cols=10000 --r=40 --iter=10000 --gpu --Float32 --init_from_master > rslts/nmf-mult-40-3gpu.txt
mpirun -np 4 julia --project=./cuda nmf-mult.jl --rows=10000 --cols=10000 --r=40 --iter=10000 --gpu --Float32 --init_from_master > rslts/nmf-mult-40-4gpu.txt
mpirun -np 5 julia --project=./cuda nmf-mult.jl --rows=10000 --cols=10000 --r=40 --iter=10000 --gpu --Float32 --init_from_master > rslts/nmf-mult-40-5gpu.txt
mpirun -np 6 julia --project=./cuda nmf-mult.jl --rows=10000 --cols=10000 --r=40 --iter=10000 --gpu --Float32 --init_from_master > rslts/nmf-mult-40-6gpu.txt
mpirun -np 7 julia --project=./cuda nmf-mult.jl --rows=10000 --cols=10000 --r=40 --iter=10000 --gpu --Float32 --init_from_master > rslts/nmf-mult-40-7gpu.txt
mpirun -np 8 julia --project=./cuda nmf-mult.jl --rows=10000 --cols=10000 --r=40 --iter=10000 --gpu --Float32 --init_from_master > rslts/nmf-mult-40-8gpu.txt

mpirun -np 1 julia --project=./cuda nmf-mult.jl --rows=10000 --cols=10000 --r=60 --iter=10000 --gpu --Float32 --init_from_master > rslts/nmf-mult-60-1gpu.txt
mpirun -np 2 julia --project=./cuda nmf-mult.jl --rows=10000 --cols=10000 --r=60 --iter=10000 --gpu --Float32 --init_from_master > rslts/nmf-mult-60-2gpu.txt
mpirun -np 3 julia --project=./cuda nmf-mult.jl --rows=10000 --cols=10000 --r=60 --iter=10000 --gpu --Float32 --init_from_master > rslts/nmf-mult-60-3gpu.txt
mpirun -np 4 julia --project=./cuda nmf-mult.jl --rows=10000 --cols=10000 --r=60 --iter=10000 --gpu --Float32 --init_from_master > rslts/nmf-mult-60-4gpu.txt
mpirun -np 5 julia --project=./cuda nmf-mult.jl --rows=10000 --cols=10000 --r=60 --iter=10000 --gpu --Float32 --init_from_master > rslts/nmf-mult-60-5gpu.txt
mpirun -np 6 julia --project=./cuda nmf-mult.jl --rows=10000 --cols=10000 --r=60 --iter=10000 --gpu --Float32 --init_from_master > rslts/nmf-mult-60-6gpu.txt
mpirun -np 7 julia --project=./cuda nmf-mult.jl --rows=10000 --cols=10000 --r=60 --iter=10000 --gpu --Float32 --init_from_master > rslts/nmf-mult-60-7gpu.txt
mpirun -np 8 julia --project=./cuda nmf-mult.jl --rows=10000 --cols=10000 --r=60 --iter=10000 --gpu --Float32 --init_from_master > rslts/nmf-mult-60-8gpu.txt
