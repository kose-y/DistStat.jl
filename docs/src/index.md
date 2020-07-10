
# DistStat.jl: Towards Unified Programming for High-performance Statistical Computing Environments in Julia

## Introduction
`DistStat.jl` implements a distributed array data structure on both distributed CPU and GPU environments and also provides an easy-to-use interface to the structure in the programming language Julia. A user can switch between the underlying array implementations for CPU cores or GPUs only with minor configuration changes. Furthermore, `DistStat.jl` can generalize to any environment on which the message passing interface (`MPI`) is supported. This package leverages on the built-in support for multiple dispatch and vectorization semantics of Julia, resulting in easy-to-use syntax for elementwise operations and distributed linear algebra.

## Software Interface 

`DistStat.jl` implements a distributed `MPI`-based array data structure `MPIArray` as the core data structure for implementations of `AbstractArray`s. It uses [`MPI.jl`](https://github.com/JuliaParallel/MPI.jl) as a backend. It has been tested for basic `Array`s and `CuArray`s from [`CUDA.jl`](https://github.com/JuliaGPU/CUDA.jl). The standard vectorized "dot" operations can be used for convenient element-by-element operations as well as broadcasting operations on `MPIArray`s. Furthermore, simple distributed matrix operations for `MPIMatrix`, or two-dimensional `MPIArray`s, are also implemented. Reduction and accumulation operations are supported for `MPIArrays` of any dimensions.   The package can be loaded by:
```julia
using DistStat
```
If GPUs are available, one that is to be used is automatically selected in a round-robin fashion upon loading the package. The rank, or the "ID" of a process, and the size, or the total number of the processes, can be accessed by:
```julia
DistStat.Rank()
DistStat.Size()
```
Ranks are indexed 0-based, following the `MPI` standard.

### Data Structure for Distributed MPI Array

In `DistStat.jl`, a distributed array data type `MPIArray{T,N,AT}` is defined. Here, parameter `T` is the type of each element of an array, e.g., `Float64` or `Float32`. Parameter `N` is the dimension of the array, `1` for vector and `2` for matrix. Parameter `AT` is the implementation of `AbstractArray` used for base operations: `Array` for CPU array, and `CuArray` for the arrays on Nvidia GPUs (requires `CUDA.jl`). If there are multiple CUDA devices, a device is assigned to a process automatically by teh rank of the process modulo the size. This assignment scheme extends to the setting in which there are multiple GPU devices in multiple CPU nodes. The type `MPIArray{T,N,AT}` is a subtype of `AbstractArray{T,N}`. In `MPIArray{T,N,AT`, each rank holds a contiguous block of the full data in `AT{T,N}` split by the `N`-th dimension, or the last dimension of an `MPIArray`.

In the special case of a two-dimenaional array, aliased by `MPIMatrix{T,AT}`, the data is column-major ordered and column-split. The transpose of this matrix has type of  `Transpose{T,MPIMatrix{T,AT}}` which is row-major ordered, row-split. There also is an alias for one-dimensional array `MPIArray{T,1,A}`, which is `MPIVector{T,A}`.

#### Creation
The syntax `MPIArray{T,N,A}(undef, m, ...)` creates an ininitialized `MPIArray`. For example,


```julia
a = MPIArray{Float64, 2, Array}(undef, 3, 4)
```

creates an uninitialized 3 $\times$ 4 distributed array based on local `Array`s of double precision floating-point numbers. The size of this array, the type of each element, and number of dimensions can be accessed using the usual functions in Julia:


```julia
size(a)
eltype(a)
ndims(a)
```

Local data held by each process can be accessed by appending `.localarray` to the name of the array, e.g., 


```julia
a.localarray
```

Matrices are split as evenly as possible. For example, if the number of processes is 4 and the `size(a) == (3, 7)`, processes of rank 0 through 2 hold the local data of size (3, 2) and the rank-3 process holds the local data of size (3, 1).

An `MPIArray` can also be created by distributing an array in a single process. For example, in the following code: 


```julia
if DistStat.Rank() == 0
    dat = [1, 2, 3, 4]
else
    dat = Array{Int64}(undef, 0)
end
d = distribute(dat)
```

the data is defined in rank 0 process, and the other processes have an empty instance of `Array{Int64}`. Using the function `distribute`, the `MPIArray{Int64, 1, Array}` of the data `[1, 2, 3, 4]`, equally distributed over four processes, is created.

#### Filling an array
An `MPIArray` `a` can be filled with a  number `x` using the usual syntax of the function `fill!(a, x)`. For example, `a` can be filled with zero:


```julia
fill!(a, 0)
```

#### Random number generation
An array can also be filled with random values, extending `Random.rand!()` for the standard uniform distribution and `Random.randn()!` for the standard normal distribution. The following code fills `a` with uniform(0, 1) random numbers:


```julia
using Random
rand!(a)
```

In cases such as unit testing, generating identical data for any configuration is important. For this purpose, the following interface is defined:


```julia
function rand!(a::MPIArray{T,N,A}; seed=nothing, common_init=false, root=0) where {T,N,A}
```

If the keyword argument `common_init=true` is set, the data are generated from the process with rank `root`. The `seed` can also be configured. If `commonn_init==false` and `seed==k`, the seed for each process is defined by `k` plus the rank.

### "Dot" syntax and vectorization


The "dot" broadcasting feature of `DistStat.jl` follows the standard Julia syntax. This syntax provides a convenient way to operate on both multi-node clusters and multi-GPU workstations with the same code. For example, the soft-thresholding operator, which commonly appears in sparse regression can be defined in the element level:


```julia
function soft_threshold(x::T, λ::T) ::T where T <: AbstractFloat
    x > λ && return (x - λ)
    x < -λ && return (x + λ)
    return zero(T)
end
```

This function can be applied to each element of an `MPIArray` using the dot broadcasting, as follows. When the dot operation is used for an `MPIArray{T,N,AT}`, it is naturally passed to inner array implementation `AT`. Consider the following arrays filled with random numbers from the standard normal distribution:


```julia
a = MPIArray{Float64, 2, Array}(undef, 2, 4) |> randn!
b = MPIArray{Float64, 2, Array}(undef, 2, 4) |> randn!
```

The function `soft_threshold()` is applied elementwisely as the following:


```julia
a .= soft_threshold.(a .+ 2 .* b, 0.5)
```

The three dot operations, `.=`, `.+`, and `.*`, are fused into a single loop (in CPU) or a single kernel (in GPU) internally. 

A singleton non-last dimension is treated as if the array is repeated along that dimension, just like `Array` operations.


```julia
c = MPIArray{Float64, 2, Array}(undef, 1, 4) |> rand!
a .= soft_threshold.(a .+ 2 .* c, 0.5)
```

works as if `c` were a $2 \times 4$ array, with its content repeated twice.

It is a little bit subtle with the last dimension, as the `MPIArray{T,N,AT}`s are split along that dimension. It works if the broadcast array has the type `AT` and holds the same data across the processes. For example,


```julia
d = Array{Float64}(undef, 2, 1); fill!(d, -0.1)
a .= soft_threshold.(a .+ 2 .* d, 0.5)
```

As with any dot operations in Julia, the operatons for `DistStat.jl` are convenient but usually not the fastest option, Its implementations can be further optimized by specializeing in specific array types.

### Reduction opeartions and accumulation operations 
Reduction operations, such as `sum()`, `prod()`, `maximum()`, `minimum()`, and accumulations such as `cumsum()`, `cumsum!()`, `cumprod()`, `cumprod!()` are implemented just like their base counterparts, computing cumulative sums and products. Example usages of `sum()` and `sum!()` are:


```julia
sum(a)
sum(abs2, a) # sum of squared absolute value
sum(a, dims=1) # column sums
sum(a, dims=2)
sum(a, dims=(1,2)) # returns 1x1 MPIArray
sum!(c, a) # columnwise sum 
sum!(d, a) # rowwise sum
```

The first line computes the elementwise sum of `a`. The second line computes the sum of squared absolute values (`abs2()` is the method that computes the squared absolute values). The third and fourth lines compute the column sums and row sums, respectively. Similar to the dot operations, the third line reduces along the distributed dimensions, and returns a broadcast local `Array`. The fifth line returns the sum of all elements, but the data type is a $1 \times 1$ `MPIArray`. The syntax `sum!(p, q)` selects which dimension to reduce based on the shape of `p`, the first argument. The sixth line computes the columnwise sum and saves it to `c`, because `c` is a $1 \times 4$ `MPIArray`. The seventh line computes rowwise sum, because `d` is a $2 \times 1$ local `Array`.  

Given below are examples for `cumsum()` and `cumsum!()`:


```julia
# Accumulations
cumsum(a; dims=1) # columnwise accumulation
cumsum(a; dims=2) # rowwise accumulation
cumsum!(b, a; dims=1) # columnwise, result saved in `b`
cumsum!(b, a; dims=2)
```

The first line computes the columnwise cumulative sum, and the second line computes the rowwise cumulative sum. So do the third and fourth lines, but save the results in `b`, which has the same size as `a`. 

### Distributed Linear Algebra

#### Dot product

The method `LinearAlgebra.dot()` for `MPIArray`s is defined just like the base `LinearAlgebra.dot()`, which sums all the elements after an elementwise multiplication of the two argument arrays: 


```julia
using LinearAlgebra
dot(a, b)
```

#### Operations on the diagonals
The "getter" method for the diagonal, `diag!(d, a)`, and the "setter" method for the diagonal, `fill_diag!()`, are also available.
The former obtains the main diagonal of the `MPIMatrix` `a` and is stored in `d`. If `d` is an `MPIMatrix` with a single row, the result is obtained in a distributed form. On the other hand, if `d` is a local `AbstractArray`, all elements of the main diagonal is copied to all processes as a broadcast `AbstractArray`:


```julia
M = MPIMatrix{Float64, Array}(undef, 4, 4) |> rand!
v_dist = MPIMatrix{Float64, Array}(undef, 1, 4)
v = Array{Float64}(undef, 4)
diag!(v_dist, M)
diag!(v, M)
```

#### Matrix multiplication

The method `LinearAlgebra.mul!(C, A, B)` is implemented for `MPIMatrix`, in which the multiplication of `A` and `B` is stored in `C`. Matrix multiplications for 17 different combinations of types for `A`, `B`, and `C`, including matrix-vector multiplications are included in the package. It is worth noting that transpose of an `MPIMatrix` is a row-major ordered, row-split matrix. While the base syntax of `mul!(C, A, B)` is always available, any temporary memory to save intermediate results can also be provided as a keyword argument in order to avoid repetitive allocations in iterative algorithms, as in `mul!(C, A, B; tmp=Array(undef, 3, 4)`. The user should determine which shape of `C` minimizes communication and suits better for their application. `MPIColVector{T, AT}` is defined as `Union{MPIVector{T,AT}, Transpose{T, MPIMatrix{T,AT}}}` to include transposed `MPIMatrix` with a single row. The 17 possible combinations of arguments available are listed below:


```julia
LinearAlgebra.mul!(C::MPIMatrix{T,AT}, A::MPIMatrix{T,AT}, B::Transpose{T, MPIMatrix{T,AT}};
                            tmp::AbstractArray{T,2}=AT{T}(undef, size(A,1), size(B,2))) where {T,AT}
LinearAlgebra.mul!(C::Transpose{T,MPIMatrix{T,AT}}, A::MPIMatrix{T,AT},B::Transpose{T,MPIMatrix{T,AT}};
                            tmp::AbstractArray{T,2}=AT{T}(undef, size(B,2), size(A,1))) where {T,AT}
LinearAlgebra.mul!(C::MPIMatrix{T,AT}, A::MPIMatrix{T,AT}, B::MPIMatrix{T,AT};
                            tmp::AbstractArray{T,2}=AT{T}(undef, size(A,1), size(A,2))) where {T,AT}
LinearAlgebra.mul!(C::Transpose{T,MPIMatrix{T,AT}}, A::Transpose{T,MPIMatrix{T,AT}}, B::Transpose{T,MPIMatrix{T,AT}};
                            tmp::AbstractArray{T,2}=AT{T}(undef, size(B,2), size(B,1))) where {T,AT}
LinearAlgebra.mul!(C::AbstractMatrix{T}, A::MPIMatrix{T,AT}, B::Transpose{T,MPIMatrix{T,AT}}) where {T,AT}
LinearAlgebra.mul!(C::MPIMatrix{T,AT}, A::Transpose{T,MPIMatrix{T,AT}}, B::MPIMatrix{T,AT};
                            tmp::AbstractArray{T,2}=AT{T}(undef,size(A,2),size(A,1))) where {T,AT}
LinearAlgebra.mul!(C::Transpose{T,MPIMatrix{T,AT}}, A::Transpose{T,MPIMatrix{T,AT}}, B::MPIMatrix{T,AT};
                            tmp::AbstractArray{T,2}=AT{T}(undef,size(B,1), size(B,2))) where {T,AT}
LinearAlgebra.mul!(C::MPIMatrix{T,AT}, A::Union{AbstractMatrix{T}}, B::MPIMatrix{T,AT}) where {T,AT}
LinearAlgebra.mul!(C::MPIMatrix{T,AT}, A::Transpose{T,ATT} where ATT <: AbstractMatrix{T}, B::MPIMatrix{T,AT}) where {T,AT}
LinearAlgebra.mul!(C::Transpose{T,MPIMatrix{T,AT}}, A::Transpose{T,MPIMatrix{T,AT}},
                            B::Transpose{T, ATT} where ATT <: AbstractMatrix{T}) where {T,AT}
LinearAlgebra.mul!(C::Transpose{T,MPIMatrix{T,AT}}, A::Transpose{T,MPIMatrix{T,AT}}, B::AbstractMatrix{T}) where {T,AT}
LinearAlgebra.mul!(C::AbstractVector{T}, A::MPIMatrix{T,AT}, B::AbstractVector{T}) where {T,AT}
LinearAlgebra.mul!(C::AbstractVector{T}, A::Transpose{T, MPIMatrix{T,AT}}, B::AbstractVector{T}) where {T,AT}
const MPIColVector{T,AT} = Union{MPIVector{T,AT},Transpose{T,MPIMatrix{T,AT}}}
LinearAlgebra.mul!(C::AbstractVector{T}, A::MPIMatrix{T,AT}, B::MPIColVector{T,AT}) where {T,AT}
LinearAlgebra.mul!(C::MPIColVector{T,AT}, A::Transpose{T,MPIMatrix{T,AT}}, B::AbstractVector{T}) where {T,AT} 
LinearAlgebra.mul!(C::MPIColVector{T,AT}, A::MPIMatrix{T,AT}, B::MPIColVector{T,AT};
                            tmp::AbstractArray{T}=AT{T}(undef, size(C, 1))) where {T,AT}
LinearAlgebra.mul!(C::MPIColVector{T,AT}, A::Transpose{T,MPIMatrix{T,AT}},B::MPIColVector{T,AT};
                            tmp::AbstractArray{T}=AT{T}(undef, size(B,1))) where {T,AT}
```

#### Operator norms

The method `opnorm()` either evaluates ($\ell_1$ and $\ell_\infty$) or approximates ($\ell_2$)  matrix operator norms, defined for a matrix $A \in \mathbb{R}^{m \times n}$ as $\|A\| = \sup\{\|Ax\|: x \in \mathbb{R}^n \text{ with } \|x\| = 1\}$ for each respective vector norm. 


```julia
opnorm(a, 1)
opnorm(a, 2)
opnorm(a, Inf)
```

The $\ell_2$-norm estimation implements the power iteration, and can be further configured for convergence and number of iterations. There also is an implementation based on the inequality $\|A\|_2 \le \|A\|_1 \|A\|_\infty$ (`method="quick"`), which overestimates the $\ell_2$-norm.


```julia
opnorm(a, 2; method="power", tol=1e-6, maxiter=1000, seed=95376)
```


```julia
opnorm(a, 2; method="quick")
```

### Simplified MPI Interfaces

`DistStat.jl` also provides a simplified version of `MPI` primitives. These primitives allow omission of the constant `MPI.COMM_WORLD`, tag for communication, and the root if it is zero. They are modeled after the convention of the `distributed` subpackage of `PyTorch`, but with richer interfaces including `Allgatherv!()`, `Gatherv!()`, and `Scatterv!()`. These primitives are not only exposed to the user, but also extensively used in the linear algebra and array operation routines explained above. `Array`s can be used as arguments fo rthe following functions, and `CuArrays` can be used if `CUDA`-aware `MPI` implementation such as `OpenMPI` is available on the system. The following list shows the signatures of simplified `MPI` methods. See [documentations](https://juliaparallel.github.io/MPI.jl/latest/) of `MPI.jl` for more information.


```julia
Barrier()
Bcast!(arr::AbstractArray; root::Integer=0)
Send(arr::AbstractArray, dest::Integer; tag::Integer=0)
Recv!(arr::AbstractArray, src::Integer; tag::Integer=0)
Isend(arr::AbstractArray, dest::Integer; tag::Integer=0)
Irecv!(arr::AbstractArray, src::Integer; tag::Integer=0)
Reduce!(sendarr::AbstractArray, recvarr::AbstractArray; 
    op=MPI.SUM, root::Integer=0)
Allreduce!(arr::AbstractArray; op=MPI.SUM)
Allgather!(sendarr::AbstractArray, recvarr::AbstractArray)
Allgather!(sendarr::AbstractArray, recvarr::AbstractArray, 
    count::Integer)
Allgatherv!(sendarr::AbstractArray, recvarr::AbstractArray, 
    counts::Vector{<:Integer})
Scatter!(sendarr::AbstractArray, recvarr::AbstractArray; 
    root::Integer=0)
Scatterv!(sendarr::AbstractArray, recvarr::AbstractArray, 
    counts::Vector{<:Integer}; root::Integer=0)
Gather!(sendarr::AbstractArray, recvarr::AbstractArray; 
    root::Integer=0)
Gatherv!(sendarr::AbstractArray, recvarr::AbstractArray, 
    counts::Vector{<:Integer}; root::Integer=0)
```
