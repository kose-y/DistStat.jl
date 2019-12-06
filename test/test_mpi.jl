using Pkg, MPI, Test
MPI.Init()

type=[Float64,Float32]

if haskey(Pkg.installed(), "CuArrays")
    using CuArrays
    A = CuArray
else
    A = Array
end

for T in type
    a=A{T}(reshape(collect(1:100),10,10))
    b = MPI.Allreduce(a, MPI.MIN, MPI.COMM_WORLD)
    println((@test isapprox(a,b)))
end
