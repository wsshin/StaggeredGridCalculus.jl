# About the construction of a grid
#
# - The domain boundaries are always primal grid planes.
#
# - The dual grid planes are always mid-sections between adjacent primal grid planes.
#
# - In each Cartesian axis, the lower- and upper-boundary conditions are always chosen the
# same.  For example, if the lower boundary is Bloch, the upper boundary must be also Bloch.
# Similarly, if the lower boundary is PEC, the upper boundary is also PEC.  (More on this
# below.)
#
# - The primal and dual grid planes are not necessarily associated with the E- and H-field
# planes.  Which of the primal and dual grid planes to use for the E- and H-fields is
# decided by the choice of the boundary condition.
#
#       - The PEC and PMC symmetry boundary conditions are realized by putting the relavant
#       fields on the domain boundary as the tangential components.  Therefore, we use the
#       primal grid planes as the E-field planes for the PEC boundary condition, and as the
#       H-field planes for the PMC boundary condition.
#
#       - On the other hand, for the Bloch boundary condition we can use the primal grid
#       planes as either of the E- and H-field planes.  Note that Maxwell's equations are
#       ∇ₘ × μ⁻¹ ∇ₑ × E - ω² ε E = -i ω J.  Once we decide that in the w-axis direciton
#       (w = x,y,z) we are going to use the primal grid planes as the E-field planes, we can
#       simply use the forward difference (which is for primal fields) as ∂/∂w in
#       constructing ∇ₑ.
#
# - The notion of U, V, and ξ, η is now inappropriate.  I used to use U to indicate the
# primal field, but there is no such thing like the primal field, because the E-field, for
# example, does not have to be completely on the primal grid planes.  The notion makes sense
# only in each Cartesian direction, but not in all Cartesian directions.


# About the symmetry boundary condition
#
# - The symmetry boundary is used to implement PEC and PMC on the boundaries of the domain.
# The symmetry boundary condition zeros the tangential fields to the boundary.  Because the
# domain boundaries are always primal grid planes, this means that PEC is implemented by
# using the primal grid planes as the E-tangential planes and the dual grid plane as the
# H-tangential planes.
#
# - An important property of the symmetry boundary plane is that it simultaneously zeros the
# tangential field and normal field defined on the boundary.  Here, the two fields are of
# different kinds.  For example, if the primal grid planes are E-tangential planes, on the
# boundary planes tangential E-fields and normal H-fields are defined.  The symmetry
# boundary condition zeros both kinds of the fields.
#
# The fact that symmetry boundary condition zeros *any* fields defined *on* the boundary
# greatly simplify the implementation of the effect of the boundary condition in
# differential operators.  For example, for the PEC boundary condition, the forward
# derivative operator subtracts fields between primal grid planes, but these fields can be
# either tangential E-fields or normal H-fields.  We have to create the derivative operators
# to handle the boundary conditions properly (such that they zero the fields that must be
# zero), but this property allows us to use the same derivative operator for both E- and H-
# fields.
#
# - You never need to mix the Bloch boundary condition and symmetry boundary condition like
# PEC and PMC in one axis (i.e., don't need to put the Bloch boundary condition at one end
# and the symmetry boundary condition at the other end), because by using the symmetry
# boundary condition on both ends of the domain your domain is automatically repeated to
# infinity.  (In other words, symmetry boundary condition is a way to simulate a periodic
# structure whose unit cell is mirror-symmetric.  Note that inside a mirror-symmetric
# structure, the field distribution can be mirror-symmetric or antisymmetric.)
#
# In some cases you may want to put PEC on one end and PMC on the other end of the domain to
# simulate a half of a mirror-symmetric unit cell of a periodic structure where the field
# distribution is symmetric at the center plane of the unhalved unit cell but antisymmetric
# at the boundaries of the unit cell.  At first glance, supporting such a mixed symmetry
# boundary condition would be difficult to support, because PEC and PMC are realized by
# zeroing the tangential E- and H-fields.  Implementing PEC on the lower boundary and PMC on
# the upper boundray means therefore, e.g., to start the domain with a primal grid plane
# (containing tangential E-fields) and end the domain with a dual grid plane (containing
# tangential H-fields), which requires putting only a half grid on the upper boundary and
# thus not ideal.  In fact, we can still keep a whole integral number of grid cells
# to implement the PMC on the lower boundary and PEC on the upper boundary by using the cool
# trick I used in FD3D.  (We cannot implement PEC on the lower boundary and PMC on the upper
# boundary even with that trick.)  However, supporting such a mixed boundary condition makes
# the code unnecesasrily convoluted, so I will not support it for now.  (Consider supporting
# it by passing something like an "ismixed" flag.)

# About matrix bandwidth vs. boundary condition
#
# Sparsity pattern of matrix with Bloch boundary condition:
# ⠻⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙
# ⠀⠈⠻⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠈⠻⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠈⠻⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠈⠻⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣦⡀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣦⡀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣦⡀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣦⡀⠀
# ⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣦
#
# Note the nonzero entries at the top-right and bottom-left corners.
#
# Sparsity pattern of matrix with symmetry boundary condition:
# ⠻⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠈⠻⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠈⠻⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠈⠻⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠈⠻⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣦⡀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣦⡀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣦⡀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣦⡀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣦
#
# No nonzero entries at those corners.  Hence, the symmetry boundary condition produces much
# narrower bandwidth, which leads less fill-ins in matrix factors.

export Grid  # types
export isproper_blochphase, create_e⁻ⁱᵏᴸ  # functions

# Below, the transformation τ refers to the transformation of a ghost point back to the
# non-ghost point that the material parameter at the ghost point comes from.  For the Bloch
# boundary condition, τ transforms the ghost point to the other side of the domain; for the
# symmetry boundary condition, τ transforms the ghost point to its mirror symmetry position
# with respect to the boundary.
#
# All the fields of Ghosted include ghost points.  Therefore, their lengths are N.+1, rather
# than N.  (Even thought σ seems to deal with reflection of ghost points, it doesn't; it has
# N entries, so it is defined in Grid rather than in Ghosted.)
struct Ghosted{K}
    lg::NTuple{2,NTuple{K,Vector{Float64}}}  # lg[PrimalGrid()][k] = primal vertex locations in k-direction with ghost points
    τlg::NTuple{2,NTuple{K,Vector{Float64}}}  # τlg[PrimalGrid()][k] = primal vertex locations in k-direction with transformed ghost points into domain; all τlg are in domain
    τind::NTuple{2,NTuple{K,Vector{Int}}}  # τind[PrimalGrid()][k] = indices of ghosted.lg corresponding to transformed points in to domain by boundary conditions
    ∆τ::NTuple{2,NTuple{K,Vector{Float64}}}  # ∆τ[PrimalGrid()][k] = amount of "shift" to get points into domain by Bloch boundary conditions (ignoring transformation by reflection by symmetry boundary, which is handled by Grid.σ)
end

# Consider storing lvxlbounds, which is
# ([(ldual₀,ldual₁), (ldual₁,ldual₂), ..., (ldualₙ₋₁,ldualₙ)],
#  [(lprim₁,lprim₂), ..., (lprimₙ₋₁,lprimₙ), (lprimₙ,lprimₙ₊₁)]
# The purpose of this field is easy retrieval of voxel bounds surrounding a voxel center.
# For example, for a voxel center l[gt][i], the surrounding voxel bounds are lvxlbounds[gt][i],
# regardless of the value of gt.  You don't need to adjust i by ±1.
#
# Another option.  To get the voxel bounds for voxel centers at l[gt], take l[conj(gt)],
# and prepend or append the ghost point appropriately.  Let the resulting array lvxlbounds.
# Then, for a voxel center l[gt][i], the voxel bounds are lvxlbounds[[i,i+1]], regardless of
# the value of gt.
struct Grid{K}
    N::SVector{K,Int}  # N[k] = number of grid cells in k-direction
    L::SVector{K,Float64}  # L[k] = length of grid (domain) in k-direction
    l::NTuple{2,NTuple{K,Vector{Float64}}}  # l[PrimalGrid()][k] = primal vertex locations in k-direction
    ∆l::NTuple{2,NTuple{K,Vector{Float64}}}  # ∆l[PrimalGrid()][k] = (∆l at primal vertices in k) == diff(l[DualGrid()][k] including ghost point)
    isbloch::SVector{K,Bool}  # isbloch[k]: true if boundary condition in k-direction is Bloch
    σ::NTuple{2,NTuple{K,Vector{Bool}}}  # false only for non-ghost points exactly on symmetry boundary (= first point in primal grid)
    bounds::NTuple{2,SVector{K,Float64}}  # bounds[LowerBound()][k] = boundary of domain at (-) end in k-direction
    ghosted::Ghosted{K}  # data related to ghost points
end

# Constructor taking non-static vectors.
Grid(lprim::NTuple{K,AbstractVector{<:Real}}, isbloch::AbstractVector{Bool}) where {K} = Grid(lprim, SVector{K}(isbloch))

Base.size(grid::Grid) = grid.N.data
Base.length(grid::Grid) = prod(size(grid))

# Constructor calling the inner constructor.
function Grid(
    lprim::NTuple{K,AbstractVector{<:Real}},  # primal grid plane locations, including both domain boundaries
    isbloch::SVector{K,Bool}
) where {K}
    @argcheck all(issorted.(lprim))  # all entry vectors of lprim must be sorted

    # For array inputs, create separate copies.
    lprim = collect.(float.(lprim))  # collect instead of deepcopy to turn range (e.g., 1.0:10.0) into vector

    ldual = movingavg.(lprim)  # NTuple{K,Vector{Float64}}

    bounds = SVector.((first.(lprim), last.(lprim)))  # NTuple{2,SVector{K,<:Real}}
    L = bounds[UpperBound()] - bounds[LowerBound()]  # SVector{K,<:Real}

    # Prepare lprim and ldual to calculate ∆lprim and ∆ldual.
    # Note that ∆lprim is not ∆'s or lprim, but ∆'s defined at lprim locations.
    for k = 1:K
        # To calculate ∆'s at lprim, we need one more ldual outside the negative boundary.
        ldual₀ = isbloch[k] ? ldual[k][end]-L[k] : 2bounds[LowerBound()][k]-ldual[k][1]
        prepend!(ldual[k], ldual₀)  # length(ldual[k]) = N[k]+1
    end

    # Below, (∆lprim, ∆ldual) is not (diff(lprim), diff(ldual)), but swapped.
    ∆lprim = diff.(ldual)  # NTuple{K,Vector{Float64}}; length(∆lprim[k]) = N[k]
    ∆ldual = diff.(lprim)  # NTuple{K,Vector{Float64}}; length(∆ldual[k]) = N[k]
    ∆l = (∆lprim, ∆ldual)

    # Set N (number of grid cells along the axis).
    @assert length.(∆l[PrimalGrid()]) == length.(∆l[DualGrid()])  # lprim, ldual, ∆lprim, ∆ldual have the same length
    N = SVector(length.(∆l[PrimalGrid()]))  # SVector{3,Int}

    # Find the locations of the ghost points transformed into the domain by the boundary
    # conditions.  Note that regardless of the boundary condition, the primal ghost point is
    # on the positive side (so it is lprim[end]) and the dual ghost point is on the negative
    # side (so it is ldual[1]).
    # Make sure these locations are where the objects assigned to the ghost points are found.

    # Construct an instance of Ghosted.  Note that lprim and ldual include ghost points now.
    τind = (map(n->collect(1:n+1), N.data), map(n->collect(1:n+1), N.data))  # NTuple{2,NTuple{3,Vector{Int}}}
    τindg_prim = .!isbloch.*N .+ 1  # SVector{K,Int}: N+1 for symmetry; 1 for Bloch
    τindg_dual = isbloch.*N .+ 1 .+ .!isbloch  # SVector{K,Int}: 2 for symmetry; N+1 for Bloch
    τindg = (τindg_prim, τindg_dual)  # NTuple{2,SVector{K,Int}}

    τlg = (deepcopy(lprim), deepcopy(ldual))

    ∆τ = (zeros.((N.+1).data), zeros.((N.+1).data))  # NTuple{2,NTuple{K,Vector{Float64}}}

    for k = 1:K
        τind[PrimalGrid()][k][end] = τindg[PrimalGrid()][k]  # N+1 for symmetry; 1 for Bloch
        τind[DualGrid()][k][1] = τindg[DualGrid()][k]  # 2 for symmetry; N+1 for Bloch

        # Below, getindex() does not return an index, but the value at the given index.
        τlg[PrimalGrid()][k][end] = getindex(lprim[k], τind[PrimalGrid()][k][end])  # lprim[k][τind[PrimalGrid()][k][end]]
        τlg[DualGrid()][k][1] = getindex(ldual[k], τind[DualGrid()][k][1])  # ldual[k][τind[DualGrid()][k][1]]

        ∆τ[PrimalGrid()][k][end] = -L[k] * isbloch[k]
        ∆τ[DualGrid()][k][1] = L[k] * isbloch[k]
    end

    ghosted = Ghosted{K}((deepcopy(lprim), deepcopy(ldual)), τlg, τind, ∆τ)

    # Construct σ, which is false on non-Bloch (= symmetry) boundaries.
    # Note that σ's are vectors of length-N and therefore do not include ghost points.
    σprim = ones.(Bool, N.data)  # NTuple{K,Vector{Bool}}; all filled with 'True'
    σdual = ones.(Bool, N.data)  # NTuple{K,Vector{Bool}}; all filled with 'True'
    for k = 1:K
        # Primal grid points are on the symmetry boundary if they are the first (non-ghost)
        # points and the symmetry boundary is used.
        σprim[k][1] = isbloch[k]

        # Dual grid points are never on any domain boundary, so no change on σdual.
    end
    σ = (σprim, σdual)

    # Eliminate ghost points.  Regardless of the boundary type, the primal ghost point
    # is on the positive side (so lprim[end] is dropped) and the dual ghost point is on
    # the negative side (so ldual[1] is dropped).
    pop!.(lprim)
    popfirst!.(ldual)
    @assert length.(lprim) == length.(ldual) == N.data  # lprim, ldual, ∆lprim, ∆ldual have the same length
    l = (lprim, ldual)

    return Grid{K}(N, L, l, ∆l, isbloch, σ, bounds, ghosted)
end

# To-do
# When needed, implement `section` (or `cross_section`) that returns Grid{K₁} from Grid{K₀}
# for K₁ < K₀.  Provide the directions (Axis) across which the cross section in taken.  Use
# the inner constructor of Grid to construct the cross section.

# Additional functions
Base.ndims(::Grid{K}) where {K} = K
Base.in(l::SVector{K}, g::Grid{K}) where {K} = all(g.bounds[LowerBound()] .≤ l .≤ g.bounds[UpperBound()])

# For the Bloch boundary, check if e⁻ⁱᵏᴸ is indeed a phase factor (i.e., amplitude = 1).
# For the non-Bloch (= symmetry) boundary, check if e⁻ⁱᵏᴸ = 1.
#
# e⁻ⁱᵏᴸ is the accumulated phase from the negative-end boundary to the positive-end one.  L
# is the distance between the boundaries (= domain size).  It is exp(-ikL), not exp(+ikL),
# because we assume the exp(+iωt) time dependence and thus the position dependence is
# exp(-ik⋅r).
isproper_blochphase(e⁻ⁱᵏᴸ::Number, isbloch::Bool) = (isbloch && abs(e⁻ⁱᵏᴸ)==1) || (!isbloch && e⁻ⁱᵏᴸ==1)
isproper_blochphase(e⁻ⁱᵏᴸ::SVector{K,<:Number}, isbloch::SVector{K,Bool}) where {K} =
    all(isproper_blochphase.(e⁻ⁱᵏᴸ, isbloch))

create_e⁻ⁱᵏᴸ(kbloch::Real, L::Real) = iszero(kbloch) ? 1 : exp(-im * kbloch * L)
create_e⁻ⁱᵏᴸ(kbloch::SVector{K,<:Real}, L::SVector{K,<:Real}) where {K} = create_e⁻ⁱᵏᴸ.(kbloch, L)
