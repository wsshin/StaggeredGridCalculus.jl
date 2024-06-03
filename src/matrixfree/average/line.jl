# Average fields along the field directions (i.e., Fw along the w-direction).  See the
# description of matrix/mean.jl/create_m̂().  Technically, apply_m̂!() can be used to average
# fields normal to the direction of averaging, but it is not used that way in apply_line_avg!().

# Assumes the space dimension and field dimension are the same.  In other words, when the
# space coordinate indices are (i,j,k), then the field has three vector components.
# Therefore, for the input field array F[i,j,k,w], we assume w = 1:3.

# The functions calculate the average and add to the output array, instead of replacing the
# values stored in the output array.  Therefore, if the derivative values themselves are
# desired, pass the output array initialized with zeros.

# Inside @threads, avoid defining β, which requires another let block to avoid unnecessary
# allocations.

export apply_line_avg!

# Wrapper for converting AbstractVector's to SVector's for arithmetic averaging
# This corresponds to the wrappers to create discrete versions of differential operators,
# but the nonzero enries in the matrix are 0.5 rather than 1.0.
apply_line_avg!(
    G::AbstractArray{<:Number},  # output field; G[i,j,k,w] is w-component of G at (i,j,k)
    F::AbstractArray{<:Number},  # input field; G[i,j,k,w] is w-component of G at (i,j,k)
    isfwd::AbstractVector{Bool},  # isfwd[w] = true|false: m̂_w is forward|backward averaging
    isbloch::AbstractVector{Bool}=ones(Bool,length(isfwd)),  # isbloch[w] = true|false : w-boundary condition is Bloch|symmetric
    e⁻ⁱᵏᴸ::AbstractVector{<:Number}=ones(Int,length(isfwd));  # e⁻ⁱᵏᴸ[w]: Bloch phase factor in w-direction
    kwargs...
) =
    (K = length(isfwd); apply_line_avg!(G, F, SVector{K}(isfwd), SVector{K}(isbloch), SVector{K}(e⁻ⁱᵏᴸ); kwargs...))

# Wrapper for converting AbstractVector's to SVector's for weighted averaging
apply_line_avg!(
    G::AbstractArray{<:Number},  # output field; G[i,j,k,w] is w-component of G at (i,j,k)
    F::AbstractArray{<:Number},  # input field; G[i,j,k,w] is w-component of G at (i,j,k)
    isfwd::AbstractVector{Bool},  # isfwd[w] = true|false: m̂_w is forward|backward averaging
    ∆l::NTuple{K,AbstractVector{<:Number}},  # line segments to multiply with; vectors of length N
    ∆l′⁻¹::NTuple{K,AbstractVector{<:Number}},  # inverse of line segments to divide by; vectors of length N
    isbloch::AbstractVector{Bool}=ones(Bool,K),  # isbloch[w] = true|false : w-boundary condition is Bloch|symmetric
    e⁻ⁱᵏᴸ::AbstractVector{<:Number}=ones(Int,K);  # e⁻ⁱᵏᴸ[w]: Bloch phase factor in w-direction
    kwargs...
) where {K} =
    apply_line_avg!(G, F, SVector{K}(isfwd), ∆l, ∆l′⁻¹, SVector{K}(isbloch), SVector{K}(e⁻ⁱᵏᴸ); kwargs...)

# Concrete implementation for arithmetic averaging
# Initially this was implemented by passing a tuple of ones() os ∆l and ∆l′⁻¹, but because
# arithmetic averaging is used frequently, the optimized version is implemented separately.
function apply_line_avg!(
    G::AbstractArray{<:Number,K₊₁},  # output field; G[i,j,k,w] is w-component of G at (i,j,k)
    F::AbstractArray{<:Number,K₊₁},  # input field; G[i,j,k,w] is w-component of G at (i,j,k)
    isfwd::SVector{K,Bool},  # isfwd[w] = true|false: m̂_w is forward|backward averaging
    isbloch::SVector{K,Bool},  # isbloch[w] = true|false : w-boundary condition is Bloch|symmetric
    e⁻ⁱᵏᴸ::SVector{K,<:Number};  # e⁻ⁱᵏᴸ[w]: Bloch phase factor in w-direction
    cmp_in=1:K,
    kwargs...
) where {K,K₊₁}  #  K is space dimension; K₊₁ = K + 1
    @assert K₊₁==K+1

    @argcheck all(1 .≤ cmp_in .≤ K)
    @argcheck allunique(cmp_in)

    for nw = cmp_in  # direction of averaging
        nu = nw  # component of input field to feed to w-directional averaging operator m̂_w (column index of matrix block)

        Fᵤ = selectdim(F, K₊₁, nu)  # nu-th component of input field
        Gᵤ = selectdim(G, K₊₁, nu)  # nu-th component of output field

        apply_m̂!(Gᵤ, Fᵤ, nw, isfwd[nw], isbloch[nw], e⁻ⁱᵏᴸ[nw]; kwargs...)
    end
end

# Concrete implementation for weighted averaging
# For the implementation details, see the comments in matrix/mean.jl.
function apply_line_avg!(
    G::AbstractArray{<:Number,K₊₁},  # output field; G[i,j,k,w] is w-component of G at (i,j,k)
    F::AbstractArray{<:Number,K₊₁},  # input field; G[i,j,k,w] is w-component of G at (i,j,k)
    isfwd::SVector{K,Bool},  # isfwd[w] = true|false: m̂_w is forward|backward averaging
    ∆l::NTuple{K,AbstractVector{<:Number}},  # line segments to multiply with; vectors of length N
    ∆l′⁻¹::NTuple{K,AbstractVector{<:Number}},  # inverse of line segments to divide by; vectors of length N
    isbloch::SVector{K,Bool},  # isbloch[w] = true|false : w-boundary condition is Bloch|symmetric
    e⁻ⁱᵏᴸ::SVector{K,<:Number};  # e⁻ⁱᵏᴸ[w]: Bloch phase factor in w-direction
    cmp_in=1:K,
    kwargs...
) where {K,K₊₁}  #  K is space dimension; K₊₁ = K + 1
    @assert K₊₁==K+1

    @argcheck all(1 .≤ cmp_in .≤ K)
    @argcheck allunique(cmp_in)

    for nw = cmp_in  # direction of averaging
        nu = nw  # component of input field to feed to w-directional averaging operator m̂_w (column index of matrix block)

        Fᵤ = selectdim(F, K₊₁, nu)  # nu-th component of input field
        Gᵤ = selectdim(G, K₊₁, nu)  # nu-th component of output field

        apply_m̂!(Gᵤ, Fᵤ, nw, isfwd[nw], ∆l[nw], ∆l′⁻¹[nw], isbloch[nw], e⁻ⁱᵏᴸ[nw]; kwargs...)
    end
end
