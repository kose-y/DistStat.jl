using Base.Broadcast
import Base.Broadcast: BroadcastStyle, Broadcasted

struct MPIArrayStyle{Style <: BroadcastStyle} <: Broadcast.AbstractArrayStyle{Any} end
MPIArrayStyle(::S) where {S} = MPIArrayStyle{S}()
MPIArrayStyle(::S, ::Val{N}) where {S,N} = MPIArrayStyle(S(Val(N)))
MPIArrayStyle(::Val{N}) where {N} = MPIArrayStyle{Broadcast.DefaultArrayStyle{N}}()

BroadcastStyle(::Type{<:MPIArray{<:Any, N, A}}) where {N,A} = MPIArrayStyle(BroadcastStyle(A), Val(N))

function BroadcastStyle(::MPIArrayStyle{AStyle}, ::MPIArrayStyle{BStyle}) where {AStyle, BStyle}
    MPIArrayStyle(BroadcastStyle(AStyle, BStyle))()
end

BroadcastStyle(::Type{<:SubArray{<:Any, <:Any, <:T}}) where T <: DArray = BroadcastStyle(T)

function Broadcast.broadcasted(::MPIArrayStyle{Style}, f, args...) where Style
    inner = Broadcast.broadcasted(Style(), f, args...)
    if inner isa Broadcasted
        return Broadcasted{MPIArrayStyle{Style}}(inner.f, inner.args, inner.axes)
    else
        return inner
    end
end


# TODO: For Transpose, Adjoint, SubArray, etc.
const MPIDestArray = MPIArray


function Base.similar(bc::Broadcasted{<:MPIArrayStyle{Style}}, ::Type{ElType}) where {Style, ElType}
    a = MPIArray(map(length, axes(bc)))
    
    #TODO

end


@inline function Base.copyto!(dest::MPIDestArray, bc::Broadcasted{Nothing})
    axes(dest) == axes(bc) || Broadcast.throwdm(axes(dest), axes(bc))
    #TODO 
end


