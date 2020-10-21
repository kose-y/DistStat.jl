mpirun -np 1 julia --project=./cuda cox.jl --rows=10000 --cols=10000 --lambda=1e-6 --iter=10000 --gpu --Float32 --init_from_master --eval_obj > rslts/cox-20-1gpu.txt
mpirun -np 2 julia --project=./cuda cox.jl --rows=10000 --cols=10000 --lambda=1e-6 --iter=10000 --gpu --Float32 --init_from_master --eval_obj > rslts/cox-20-2gpu.txt
mpirun -np 3 julia --project=./cuda cox.jl --rows=10000 --cols=10000 --lambda=1e-6 --iter=10000 --gpu --Float32 --init_from_master --eval_obj > rslts/cox-20-3gpu.txt
mpirun -np 4 julia --project=./cuda cox.jl --rows=10000 --cols=10000 --lambda=1e-6 --iter=10000 --gpu --Float32 --init_from_master --eval_obj > rslts/cox-20-4gpu.txt
mpirun -np 5 julia --project=./cuda cox.jl --rows=10000 --cols=10000 --lambda=1e-6 --iter=10000 --gpu --Float32 --init_from_master --eval_obj > rslts/cox-20-5gpu.txt
mpirun -np 6 julia --project=./cuda cox.jl --rows=10000 --cols=10000 --lambda=1e-6 --iter=10000 --gpu --Float32 --init_from_master --eval_obj > rslts/cox-20-6gpu.txt
mpirun -np 7 julia --project=./cuda cox.jl --rows=10000 --cols=10000 --lambda=1e-6 --iter=10000 --gpu --Float32 --init_from_master --eval_obj > rslts/cox-20-7gpu.txt
mpirun -np 8 julia --project=./cuda cox.jl --rows=10000 --cols=10000 --lambda=1e-6 --iter=10000 --gpu --Float32 --init_from_master --eval_obj > rslts/cox-20-8gpu.txt

