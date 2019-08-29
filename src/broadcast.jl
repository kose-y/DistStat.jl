#TODO: add @inline, test with np >= 2, test with CuArrays
using Base.Broadcast
import Base.Broadcast: BroadcastStyle, Broadcasted
import Base: tail

struct MPIArrayStyle{Style <: BroadcastStyle} <: Broadcast.AbstractArrayStyle{Any} end
MPIArrayStyle(::S) where {S} = MPIArrayStyle{S}()
MPIArrayStyle(::S, ::Val{N}) where {S,N} = MPIArrayStyle(S(Val(N)))
MPIArrayStyle(::Val{N}) where {N} = MPIArrayStyle{Broadcast.DefaultArrayStyle{N}}()

BroadcastStyle(::Type{<:MPIArray{T, N, A}}) where {T,N,A} = MPIArrayStyle(BroadcastStyle(A{T,N}), Val(N))

function BroadcastStyle(::MPIArrayStyle{AStyle}, ::MPIArrayStyle{BStyle}) where {AStyle, BStyle}
    MPIArrayStyle(BroadcastStyle(AStyle, BStyle))()
end

BroadcastStyle(::Type{<:SubArray{<:Any, <:Any, <:T}}) where T <: MPIArray = BroadcastStyle(T)

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
    MPIArray(map(length, axes(bc))...) do I
        bcc = Broadcasted{Style}(identity, (), map(length, I))
        similar(bcc, ElType)
    end
end


function Base.copyto!(dest::MPIDestArray, bc::Broadcasted)
    axes(dest) == axes(bc) || Broadcast.throwdm(axes(dest), axes(bc))
    dbc = bcdistribute(bc)

    lidcs = dest.partitioning[Rank()+1]
    lbc = bclocal(dbc, lidcs)
    Base.copyto!(dest.localarray, lbc)
    return dest
end

function Base.copy(bc::Broadcasted{<:MPIArrayStyle})
    dbc = bcdistribute(bc)
    MPIArray(map(length, axes(bc))...) do I
        lbc = Broadcast.instantiate(bclocal(dbc, I))
        copy(lbc)
    end
end

bcdistribute(bc::Broadcasted{Style}) where Style = Style <: BroadcastStyle ? Broadcasted{MPIArrayStyle{Style}}(bc.f, bcdistribute_args(bc.args), bc.axes) : bc
bcdistribute(bc::Broadcasted{Style}) where Style <: MPIArrayStyle  = Broadcasted{Style}(bc.f, bcdistribute_args(bc.args), bc.axes)

bcdistribute(x::T) where T = _bcdistribute(BroadcastStyle(T), x)
_bcdistribute(::MPIArrayStyle, x) = x
_bcdistribute(::Broadcast.AbstractArrayStyle{0}, x) = x
_bcdistribute(::Broadcast.AbstractArrayStyle, x) = x # error("not implemented") # distribute(x) TODO: define distribute
_bcdistribute(::Broadcast.AbstractArrayStyle, x::AbstractRange) = x
_bcdistribute(::Any, x) = x

bcdistribute_args(args::Tuple) = (bcdistribute(args[1]), bcdistribute_args(tail(args))...)
bcdistribute_args(args::Tuple{Any}) = (bcdistribute(args[1]),)
bcdistribute_args(args::Tuple{}) = ()


_bcview(::Tuple{}, ::Tuple{}) = ()
_bcview(::Tuple{}, view::Tuple) = ()
_bcview(shape::Tuple, ::Tuple{}) = (shape[1], _bcview(tail(shape),())...)
function _bcview(shape::Tuple, view::Tuple)
    return (_bcview1(shape[1], view[1]), _bcview(tail(shape), tail(view))...)
end

# _bcview1 handles the logic for a single dimension
function _bcview1(a, b)
    if a==1 || a == 1:1
        return 1:1
    elseif first(a) <= first(b) <= last(a) &&
            first(a) <= last(b) <= last(b)
        return b
    else
        throw(DimensionMismatch("broadcast view could not be constructed"))
    end
end

@inline bclocal(bc::Broadcasted{MPIArrayStyle{Style}}, idxs) where Style = Broadcasted{Style}(bc.f, bclocal_args(_bcview(axes(bc), idxs), bc.args))

# bclocal(bc::Broadcasted{Nothing}, idxs) = bc
bclocal(x::T, idxs) where T = _bclocal(BroadcastStyle(T), x, idxs)
function _bclocal(::MPIArrayStyle, x, idxs)
    bcidxs = _bcview(axes(x), idxs)
    @assert all(map( (y, z) -> (y  == z) || length(z) == 1 , bcidxs, x.partitioning[Rank() + 1]))
    x.localarray
    # makelocal(x, bcidxs...) # TODO: makelocal
end

_bclocal(::Broadcast.AbstractArrayStyle{0}, x, idxs) = x
function _bclocal(::Broadcast.AbstractArrayStyle, x::AbstractRange, idxs)
    @assert length(idxs) == 1
    x[idxs[1]]
end
function _bclocal(::Broadcast.Style{Tuple}, x, idxs)
    @assert length(idxs) == 1
    tuple((e for (i,e) in enumerate(x) if i in idxs[1])...)
end
_bclocal(::Broadcast.AbstractArrayStyle, x, idxs) = x
_bclocal(::Any, x, idxs) = error("don't know how to localize $x with $idxs")

@inline bclocal_args(idxs, args::Tuple) = (bclocal(args[1], idxs), bclocal_args(idxs, tail(args))...)
bclocal_args(idxs, args::Tuple{Any}) = (bclocal(args[1], idxs),)
bclocal_args(idxs, args::Tuple{}) = ()
