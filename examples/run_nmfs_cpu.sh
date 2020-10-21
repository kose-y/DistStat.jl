export OMP_NUM_THREADS=1
mpirun -np 20 julia --project=. nmf-apg.jl --rows=10000 --cols=10000 --r=20 --iter=10000  --init_from_master > rslts/nmf-apg-20-cpu.txt

mpirun -np 20 julia --project=. nmf-apg.jl --rows=10000 --cols=10000 --r=40 --iter=10000  --init_from_master > rslts/nmf-apg-40-cpu.txt

mpirun -np 20 julia --project=. nmf-apg.jl --rows=10000 --cols=10000 --r=60 --iter=10000  --init_from_master > rslts/nmf-apg-60-cpu.txt

mpirun -np 20 julia --project=. nmf-mult.jl --rows=10000 --cols=10000 --r=20 --iter=10000  --init_from_master  > rslts/nmf-mult-20-cpu.txt

mpirun -np 20 julia --project=. nmf-mult.jl --rows=10000 --cols=10000 --r=40 --iter=10000  --init_from_master > rslts/nmf-mult-40-cpu.txt

mpirun -np 20 julia --project=. nmf-mult.jl --rows=10000 --cols=10000 --r=60 --iter=10000  --init_from_master > rslts/nmf-mult-60-cpu.txt
