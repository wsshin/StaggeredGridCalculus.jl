# enumerated types
export AbstractBoundType, LowerBound, UpperBound, BoundTypes

export GridType
export PD, nPR, nDL, nPD

# functions
export gt_w, numel, alter

# Signs
abstract type AbstractBoundType end
show(io::IO, ::MIME"text/plain", bt::AbstractBoundType) = show(io, bt)

struct LowerBound <: AbstractBoundType end
show(io::IO, ::LowerBound) = print(io, "lower bound")
getindex(A::Tuple, ::LowerBound) = first(A)
getindex(A::AbstractVector, ::LowerBound) = first(A)
setindex!(A, a, ::LowerBound) = setindex!(A, a, firstindex(A))  # (A[begin] = a; A)
conj(::LowerBound) = UpperBound()

struct UpperBound <: AbstractBoundType end
show(io::IO, ::UpperBound) = print(io, "upper bound")
getindex(A::Tuple, ::UpperBound) = last(A)
getindex(A::AbstractVector, ::UpperBound) = last(A)
setindex!(A, a, ::UpperBound) = setindex!(A, a, lastindex(A))  # (A[end] = a; A)
conj(::UpperBound) = LowerBound()

instances(AbstractBoundType) = (LowerBound(), UpperBound())


# Grid types
const nPR, nDL = 1, 2  # primal, dual grids
const nPD = SVector(nPR,nDL)  # tuple allocation is more efficient than array allocation
@enum GridType PRIM=nPR DUAL
const PD = SVector(PRIM, DUAL)
for ins in instances(GridType); @eval export $(Symbol(ins)); end  # export all instances
Base.string(ins::GridType) = ins==PRIM ? "primal" : "dual"
alter(ins::GridType) = ins==PRIM ? DUAL : PRIM

# Return grid types of the w-component of the field F, given the grid types gt₀ of the
# corners of the voxels whose edges are the field lines.
# If w is not one of the valid axes, return the voxel corner grid types gt₀.
gt_w(
    nw::Int,  # component index of field
    gt₀::SVector{K,GridType}  # grid type of corners of voxel whose edges are field lines
) where {K} =
    broadcast((a,w,g)->(a==w ? alter(g) : g), SVector{K}(1:K), nw, gt₀)  # no change from gt₀ for nw ∉ {1,...,K}


# Functions for enumerated types
numel(::Type{T}) where {T} = length(instances(T))
alter(n::Int) = 3-n
