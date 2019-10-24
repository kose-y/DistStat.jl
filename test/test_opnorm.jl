using DistStat, LinearAlgebra,Random,Test

type=[Float64,Float32]
A=Array

for T in type
    a=rand(7,7)
    a_dist=distribute(a)
    cols=a_dist.partitioning[DistStat.Rank()+1][2]
    n=div(7,DistStat.Rank()+1)
    m=rem(7,DistStat.Rank()+1)
    ans=Vector{Int64}(undef,DistStat.Rank()+1)
    for i in 1:(DistStat.Rank()+1)
        if (i <= DistStat.Rank())
            ans[i]=n+m
        else
            ans[i]=n
        end
    end

    result=Vector{Float64}(undef,DistStat.Rank()+1)
    for i in 1:(DistStat.Rank()+1)
        if (i==1)
            result=opnorm(a[:,1:ans[i]])
        else
            result=opnorm(a[:,(ans[i-1]+1):ans[i]])
        end
    end

    println(@test all(isapprox(opnorm(a_dist),result)))

end
