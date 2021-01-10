module StaggeredGridCalculus

# @reexport makes all exported symbols of the exported packages available in module using
# StaggeredGridCalculus.
using Reexport
@reexport using LinearAlgebra, SparseArrays
using AbbreviatedTypes
using Base.Threads: @threads  # used in matrixfree/differential/*.jl

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
