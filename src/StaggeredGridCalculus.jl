module StaggeredGridCalculus

# @reexport makes all exported symbols of the exported packages available in module using
# StaggeredGridCalculus.
using Reexport
@reexport using LinearAlgebra, SparseArrays
using StaticArrays
using Base.Threads: @spawn  # to be used in matrixfree/differential/*.jl

## Type aliases
# Below, use Int instead of Int64 for compatibility with 32-bit systems (e.g., x86 in appveyor.yml).
const Float = typeof(0.0)  # use Float = Float128 for quadruple precision in the future
const CFloat = Complex{Float}

const Tuple2 = NTuple{2}
const Tuple3 = NTuple{3}
const Tuple4 = NTuple{4}
Tuple22{T} = Tuple2{Tuple2{T}}
Tuple23{T} = Tuple2{Tuple3{T}}
Tuple24{T} = Tuple2{Tuple4{T}}
Tuple32{T} = Tuple3{Tuple2{T}}
Tuple33{T} = Tuple3{Tuple3{T}}

const AbsVec = AbstractVector
const AbsMat = AbstractMatrix
const AbsArr = AbstractArray

const VecBool = Vector{Bool}
const VecInt = Vector{Int}
const VecFloat = Vector{Float}
const VecComplex = Vector{CFloat}

const AbsVecBool = AbsVec{Bool}
const AbsVecInt = AbsVec{Int}
const AbsVecFloat = AbsVec{Float}
const AbsVecComplex = AbsVec{CFloat}

const AbsVecInteger = AbsVec{<:Integer}
const AbsVecReal = AbsVec{<:Real}
const AbsVecNumber = AbsVec{<:Number}

const MatFloat = Matrix{Float}
const MatComplex = Matrix{CFloat}

const AbsMatFloat = AbsMat{Float}
const AbsMatComplex = AbsMat{CFloat}

const AbsMatReal = AbsMat{<:Real}
const AbsMatNumber = AbsMat{<:Number}

const AbsArrFloat = AbsArr{Float}
const AbsArrComplex = AbsArr{CFloat}

# Below, without {N}, `where T<:AbsArrNumber{3}}` is not equivalent to
# `where {T<:AbstractArray{<:Number,3}}`.
const AbsArrNumber{N} = AbsArr{<:Number,N}

const SBool{K} = SVector{K,Bool}
const SFloat{K} = SVector{K,Float}
const SInt{K} = SVector{K,Int}
const SNumber{K} = SVector{K,<:Number}

# The order of inclusion matters: if types or functions in file A are used in file B, file A
# must be included first.
include("enumtype.jl")
include("util.jl")
include("grid.jl")
# include("gridgen.jl")
include("matrix/matrix.jl")
include("matrixfree/matrixfree.jl")
include("pml.jl")

end # module StaggeredGridCalculus
