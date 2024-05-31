using StaggeredGridCalculus
using Test
# using Statistics: mean
using AbbreviatedTypes; @define_types_with(Float64)

Base.isapprox(a::Tuple, b::Tuple; kws...) = all(p -> isapprox(p...; kws...), zip(a,b))

# @testset "StaggeredGridCalculus" begin

include("enumtype.jl")
include("util.jl")
include("grid.jl")
# include("interval.jl")
# include("gridgen.jl")
include("partial.jl")
include("curl.jl")
include("divergence.jl")
include("gradient.jl")
include("line.jl")
include("permutation.jl")
include("pml.jl")

# end  # @testset "StaggeredGridCalculus"
