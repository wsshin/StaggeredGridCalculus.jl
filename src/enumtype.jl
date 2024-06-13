# enumerated types
export AbstractBoundType, LowerBound, UpperBound
export AbstractGridType, PrimalGrid, DualGrid

# functions
export gt_w

# Bound types
abstract type AbstractBoundType end
show(io::IO, ::MIME"text/plain", bt::AbstractBoundType) = show(io, bt)

struct LowerBound <: AbstractBoundType end
show(io::IO, ::LowerBound) = print(io, "lower bound")

getindex(A::Tuple, ::LowerBound) = A[begin]
getindex(A::AbstractVector, ::LowerBound) = A[begin]
setindex!(A, a, ::LowerBound) = setindex!(A, a, firstindex(A))  # (A[begin] = a; A)

conj(::LowerBound) = UpperBound()

struct UpperBound <: AbstractBoundType end
show(io::IO, ::UpperBound) = print(io, "upper bound")

getindex(A::Tuple, ::UpperBound) = A[begin+1]
getindex(A::AbstractVector, ::UpperBound) = A[begin+1]
setindex!(A, a, ::UpperBound) = setindex!(A, a, firstindex(A)+1)  # (A[end] = a; A)

conj(::UpperBound) = LowerBound()

instances(::Type{AbstractBoundType}) = (LowerBound(), UpperBound())


# Grid types
abstract type AbstractGridType end
show(io::IO, ::MIME"text/plain", gt::AbstractGridType) = show(io, gt)

struct PrimalGrid <: AbstractGridType end
show(io::IO, ::PrimalGrid) = print(io, "primal grid")
getindex(A::Tuple, ::PrimalGrid) = A[begin]
getindex(A::AbstractVector, ::PrimalGrid) = A[begin]
setindex!(A, a, ::PrimalGrid) = setindex!(A, a, firstindex(A))  # (A[begin] = a; A)
conj(::PrimalGrid) = DualGrid()

struct DualGrid <: AbstractGridType end
show(io::IO, ::DualGrid) = print(io, "dual grid")
getindex(A::Tuple, ::DualGrid) = A[begin+1]
getindex(A::AbstractVector, ::DualGrid) = A[begin+1]
setindex!(A, a, ::DualGrid) = setindex!(A, a, firstindex(A)+1)  # (A[end] = a; A)
conj(::DualGrid) = PrimalGrid()

instances(::Type{AbstractGridType}) = (PrimalGrid(), DualGrid())


# Return grid types of the w-component of the field F, given the grid types gt₀ of the
# corners of the voxels whose edges are the field F lines.
# If w is not one of the valid axes, return the voxel corner grid types gt₀.
gt_w(
    nw::Int,  # component index of field
    gt₀::SVector{K,AbstractGridType}  # grid type of corners of voxel whose edges are field lines
) where {K} =
    # Below, map() doesn't work because length(nw) is 1, not K.
    broadcast((ax,w,g)->(ax==w ? alter(g) : g), 1:K, nw, gt₀)  # no change from gt₀ for nw ∉ {1,...,K}
