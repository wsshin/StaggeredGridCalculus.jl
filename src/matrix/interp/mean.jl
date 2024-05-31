# Average fields along the field directions (i.e., Fw along the w-direction).  See the
# description of create_m̂() below.

# create_mean() assumes the space dimension and field dimension are the same.  In other
# words, when the space coordinate indices are (i,j,k), then the field has three vector
# components.  Therefore, for the input field array F[i,j,k,w], we assume w = 1:3.  This
# assumption is reasonable, because line-averaging the w-component of the field along the
# w-direction requires the interpolation points to be a half-grid point away from the
# original points along the w-direction, which means that the domain has a nonzero thickness
# along the w-direction.  Therefore, if the field has the w-component, the domain should
# have the w-dimension as well.
#
# There are cases where the averaging needs to be done in the direction orthogonal to the
# field direciton.  For example, in the TM equation, we have Ez at pixel corners, and we may
# want to know Ez at the pixel edges (at Hx- and Hy-points).  In this case the field
# dimension is 1 and different from the shape dimension = 2.  However, it doesn't make sense
# to have Ez at the Hx- and Hy-points simultaneously, because then the density of Ez points
# is doubled.  We usually need to know Ez at either the Hx-points or the Hy-points, but not
# both.  Then the needed operator can be easily generated using create_m̂().
export create_mean

# Assumption: we don't calculate averages for interpolated fields.  In other words, we
# calculate averages for the original fields defined on Yee's grid.  (There are exceptions;
# see below.)  This assumption simplifies implementation of the symmetry boundary condition
# in the averaging operators greatly.
#
# For example, suppose we use the primal grid planes as the E-field planes.  Then, for the
# symmetry boundary condition (which is the PEC boundary condition in this case), the fields
# defined on the domain boundaries are the tangential  E-fields and normal H-fields.  Both
# fields are zero on the symmetry boundaries.  Therefore, if we assume that we calculate
# averages only for those fields defined on Yee's grid, we do not have to construct
# separate averaging operators for the tangential E-field and normal H-field.
#
# See the comments in differential.jl as well.

# Note for SALT users
# The averaging scheme in SALT will be mostly used with the Bloch boundary condition, so the
# implementation of the impact of boundary conditions in the averaging scheme is actually
# not that important, because field averaging at the boundaries is trivial for the Bloch
# boundary condition.

# The forward and (-1) × backward averaging operators are not the transpose of each other
# when the symmetry boundary condition is used: one has element with the factor of 2
# multiplied while the other with the factor of zero.  Still, the symmetry boundary enforces
# a special structure on the subpixel smoothed material parameter tensor on the symmetry
# boundary.  This tensor of a special structure has zero entries where the non-transpose
# relationship between the forward and backward averaging matters, and eventually makes the
# effect of this non-transpose relationship does not propagate up to the symmetry of the
# final operator (i.e., the final operator is still symmetric even though the forward and
# (-1) × backward averaging operators are not the transpose of each other). See my notes
# entitled [Beginning of the part added on Aug/14/2018] in RN - Subpixel Smoothing.nb.

# Wrapper to create the arithmetic averaging operator by default
# This corresponds to the wrappers to create discrete versions of differential operators,
# but the nonzero enries in the matrix are 0.5 rather than 1.0.
#
# Note that when the positional arguments are SVector's, this function may call the concrete
# implementation directly instead of the next wrapper function taking ∆l and ∆l′⁻¹ even if
# the keyword arguments are not SVector's, because methods are dispatched based only on the
# types of positional arguments.  Therefore, we convert the keyword arguments into SVector's.
create_mean(
    isfwd::AbstractVector{Bool},  # isfwd[w] = true|false: m̂_w is forward|backward averaging
    N::AbstractVector{<:Integer},  # size of grid
    isbloch::AbstractVector{Bool}=fill(true,length(isfwd)),  # isbloch[w] = true|false : w-boundary condition is Bloch|symmetric
    e⁻ⁱᵏᴸ::AbstractVector{<:Number}=ones(length(isfwd));  # e⁻ⁱᵏᴸ[w]: Bloch phase factor in w-direction
    cmpfirst::Bool=true  # true to use Cartesian-component-major ordering for more tightly banded matrix
) =
    create_mean(isfwd, fill.(1.0,(N...,)), fill.(1.0,(N...,)), isbloch, e⁻ⁱᵏᴸ; cmpfirst)

# Wrapper to convert AbstractVector's to SVector's.
create_mean(
    isfwd::AbstractVector{Bool},  # isfwd[w] = true|false: m̂_w is forward|backward averaging
    ∆l::NTuple{K,AbstractVector{<:Number}},  # line segments to multiply with; vectors of length N
    ∆l′⁻¹::NTuple{K,AbstractVector{<:Number}},  # inverse of line segments to divide by; vectors of length N
    isbloch::AbstractVector{Bool}=fill(true,K),  # isbloch[w] = true|false : w-boundary condition is Bloch|symmetric
    e⁻ⁱᵏᴸ::AbstractVector{<:Number}=ones(K);  # e⁻ⁱᵏᴸ[w]: Bloch phase factor in w-direction
    cmpfirst::Bool=true  # true to use Cartesian-component-major ordering for more tightly banded matrix
) where {K} =
    create_mean(SVector{K,Bool}(isfwd), ∆l, ∆l′⁻¹, SVector{K,Bool}(isbloch), SVector{K}(e⁻ⁱᵏᴸ); cmpfirst)

# Creates the field-averaging operator for all three Cartegian components.
#
# The function used to support setting nonzero matrix blocks at off-diagonal blocks, but the
# need for doing it is absorbed in MaxwellFDM.jl/src/param.jl/create_param_matrix() by
# putting the material parameters at off-diagonal blocks.  Therefore, the field-averaging
# operators always the diagonal blocks as nonzero blocks, as they perform averaging on the
# already generated input and output fields.  See RN - Subpixel Smoothing > [Update (May/13/2018)].
function create_mean(
    isfwd::SVector{K,Bool},  # isfwd[w] = true|false: m̂_w is forward|backward averaging
    ∆l::NTuple{K,AbstractVector{<:Number}},  # line segments to multiply with; vectors of length N
    ∆l′⁻¹::NTuple{K,AbstractVector{<:Number}},  # inverse of line segments to divide by; vectors of length N
    isbloch::SVector{K,Bool},  # isbloch[w] = true|false : w-boundary condition is Bloch|symmetric
    e⁻ⁱᵏᴸ::SVector{K,<:Number};  # e⁻ⁱᵏᴸ[w]: Bloch phase factor in w-direction
    cmpfirst::Bool=true  # true to use Cartesian-component-major ordering for more tightly banded matrix
) where {K}
    T = promote_type(eltype.(∆l)..., eltype.(∆l′⁻¹)..., eltype(e⁻ⁱᵏᴸ))  # eltype(eltype(∆l)) can be Any if ∆l is inhomogeneous
    N = SVector(length.(∆l))
    M = prod(N)
    KM = K * M

    Itot = Vector{Int}(undef, 2KM)
    Jtot = Vector{Int}(undef, 2KM)
    Vtot = Vector{T}(undef, 2KM)

    indblk = 0  # index of matrix block
    for nw = 1:K  # direction of averaging
        I, J, V = create_minfo(nw, isfwd[nw], N, ∆l[nw], ∆l′⁻¹[nw], isbloch[nw], e⁻ⁱᵏᴸ[nw])  # averaging along nv-direction

        nv = nw  # component of output field generated by w-directional averaging operator m̂_w (row index of matrix block)
        istr, ioff = cmpfirst ? (K, nv-K) : (1, M*(nv-1))  # (row stride, row offset)

        nu = nw  # component of input field to feed to w-directional averaging operator m̂_w (column index of matrix block)
        jstr, joff = cmpfirst ? (K, nu-K) : (1, M*(nu-1))  # (column stride, column offset)

        @. I = istr * I + ioff
        @. J = jstr * J + joff

        # For some reason, using .= below is slower because it uses 1 allocatiotn.  On the
        # other hand, using = does not use allocation and therefore faster.
        indₛ, indₑ = ind_blk*2M + 1, (ind_blk+1)*2M  # each I, J, V is length-2M
        Itot[indₛ:indₑ] = I
        Jtot[indₛ:indₑ] = J
        Vtot[indₛ:indₑ] = V
        ind_blk += 1
    end

    return dropzeros!(sparse(Itot, Jtot, Vtot, KM, KM))  # KM×KM matrix
end
