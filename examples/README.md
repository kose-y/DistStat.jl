# Benchmarks for `DistStat.jl`

These files contain scripts to measure performance of the examples for `DistStat.jl` package.

## Running Benchmarks

For our experiments in Section 5, we used Julia version 1.2.0.
You can either use the older version used for our experiments by checking out the tag `experiment` of this repository, or use the recent set of package by using the `master` branch.

### Multi-GPU

The files `run_nmfs.sh`, `run_mds.sh`, and `run_cox.sh` contain scripts to run experiments for the Section 5. 
The scripts were executed in a system with 8 Nvidia GTX 1080 GPUs with CUDA 9.0. OpenMPI 3.0 compiled with [CUDA support](https://www.open-mpi.org/faq/?category=buildcuda) was used for the experiments.

The necessary packages can be installed using the commands:

```julia
using Pkg
pkg"add ArgParse CSV CUDA"
```

### Virtual Clusters using CfnCluster (or ParallelCluster) with Sun Grid Engine

The directory `cluster` contains the scripts to run the experiments in cluster setting (`*.job`). They launch the jobs with 36 threads per node on a virtual cluster launched with cfncluster, with the job scheduler Sun Grid Engine. We used the older cfncluster for the experiments in Section 5, but cfncluster is now updated to AWS ParallelCluster. The lines beginning with `module load` in the `*.job` files need to be removed when using ParallelCluster, the updated version of cfncluster.

#### CfnCluster/ParallelCluster
First, CfnCluster or ParallelCluster needs to be installed on a local machine. 
The virtual clusters are set up by placing the file `cluster/config` as `~/.cfncluster/config`.  This can be set up by placing the file `cluster/config_parallelcluster` as `~/.parallelcluster/config`. The files are configured to use 1-20 `c5.18xlarge` instances as the workers, with 36 MPI slots per instance. 
Visit [here](https://cfncluster.readthedocs.io/en/latest/configuration.html) to see more information regarding setup ParallelCluster, and [here](https://docs.aws.amazon.com/parallelcluster/index.html) for the newer ParallelCluster. 

Then, the command
```bash
cfncluster create example
```
is issued to launch the virtual cluster named `example`. For ParallelCluster, change `cfncluster` to `pcluster`. 
When the virtual cluster is created, they can be accessed via ssh using the command 
```bash
cfncluster ssh example -i <private key file>
```

The virtual cluster is terminated using the command:
```bash
cfncluster delete example
```

#### Software setup under the shared file system

For our experiments, we first installed the Intel MKL in `/shared/intel`, where the shared file system is mounted under `/shared`.
Then, Julia 1.2.0 was compiled from source incorporating the MKL. 
The Julia packages are installed in `/shared/julia_pkgs` using the environment variable `JULIA_DEPOT_PATH=/shared/julia_pkgs`.
The necessary packages can be installed using the following command:
```julia
using Pkg
pkg"add ArgParse CSV"
```

For each job, the script `cluster/julia_setup.sh` (placed in `/shared/julia_setup.sh` on the cloud) set up the necessary environment variables.

#### Running the jobs

The jobs are executed by submitting the `.job` file to the Sun Grid Engine. This is done by the command, for example:
```
qsub nmf-4.job 100000 200000 20
```

The file name of each `*.job` file with the number of nodes intended to be utilized.
Each `*.job` file has command-line arguments to set up data size and other parameters (e.g. the size of inner dimension for nonnegative matrix factorization). See `jobscripts/run-nmf.sh` to see how they are utilized. 

#### Changing number of slots per instance
As stated above, the files provided are for 36 processes with a single thread per instance. Some changes are needed to run experiments with 2 processes with 18 threads each.

The number of MPI slots per node is determined in the line `extra_json = {"cfncluster" : { "cfn_scheduler_slots" : "36"}` (`extra_json = {"cluster" : { "cfn_scheduler_slots" : "36"}` for ParallelCluster) of the configuration file. This value was changed to "2". 
Furthermore, the line `export OMP_NUM_THREADS=1` in each of the `*.job` file was changed to `export OMP_NUM_THREADS=18`. 
