using DistStat, LinearAlgebra,Random,Test, Pkg

type=[Float64,Float32]
if haskey(Pkg.installed(), "CuArrays")
    using CuArrays
    A= CuArray
else
    A = Array
end


for T in type
    a=A{T}(reshape(collect(1:81),9,9))
    a_dist=distribute(a)

    println(@test isapprox(opnorm(a),opnorm(a_dist)))

end
