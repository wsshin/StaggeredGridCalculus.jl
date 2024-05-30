# Assumes the space dimension and field dimension are the same.  In other words, when the
# space coordinate indices are (i,j,k), then the field has three vector components.
# Therefore, for the input field array F[i,j,k,w], we assume w = 1:3.

# The functions add the calculated values to the existing values of the output array.
# Therefore, if the derivative values themselves are desired, pass the output array
# initialized with zeros.

export apply_divg!

# Wrapper to apply the discrete divergence by default
#
# Note that when the positional arguments are SVector's, this function may call the concrete
# implementation directly instead of the next wrapper function taking ∆l⁻¹ even if the
# keyword arguments are not SVector's, because methods are dispatched based only on the
# types of positional arguments.  Therefore, we convert the keyword arguments into SVector's.
apply_divg!(
    gv̲G::AbstractArray{<:Number},  # output array of scalar; in 3D, g[i,j,k] is g at (i,j,k)
    F::AbstractArray{<:Number},  # input field; in 3D, F[i,j,k,w] is w-component of F at (i,j,k)
    isfwd::AbstractVector{Bool},  # isfwd[w] = true|false: ∂w is forward|backward difference
    ∆l⁻¹::NTuple{K,Number}=ntuple(x->1,length(isfwd)),  # ∆l⁻¹[w]: inverse of uniform distance (not distances) between grid planes in w-direction
    isbloch::AbstractVector{Bool}=ones(Bool,K),  # boundary conditions in x, y, z
    e⁻ⁱᵏᴸ::AbstractVector{<:Number}=ones(Int,K);  # Bloch phase factor in x, y, z
    kwargs...
) where {K} =
    (N = size(F)[1:K]; apply_divg!(gv̲G, F, isfwd, fill.(∆l⁻¹,N), isbloch, e⁻ⁱᵏᴸ; kwargs...))

# Wrapper for converting AbstractVector's to SVector's
apply_divg!(
    gv̲G::AbstractArray{<:Number},  # output array of scalar; in 3D, g[i,j,k] is g at (i,j,k)
    F::AbstractArray{<:Number},  # input field; in 3D, F[i,j,k,w] is w-component of F at (i,j,k)
    isfwd::AbstractVector{Bool},  # isfwd[w] = true|false: ∂w is forward|backward difference
    ∆l⁻¹::NTuple{K,AbstractVector{<:Number}},  # ∆l⁻¹[w]: inverse of distances between grid planes in w-direction
    isbloch::AbstractVector{Bool}=ones(Bool,K),  # boundary conditions in x, y, z
    e⁻ⁱᵏᴸ::AbstractVector{<:Number}=ones(Int,K);  # Bloch phase factor in x, y, z
    kwargs...
) where {K} =
    apply_divg!(gv̲G, F, SVector{K}(isfwd), ∆l⁻¹, SVector{K}(isbloch), SVector{K}(e⁻ⁱᵏᴸ); kwargs...)

# Concrete implementation for a scalar field output.
function apply_divg!(
    g::AbstractArray{<:Number,K},  # output array of scalar; in 3D, g[i,j,k] is g at (i,j,k)
    F::AbstractArray{<:Number,K₊₁},  # input field; in 3D, F[i,j,k,w] is w-component of F at (i,j,k)
    isfwd::SVector{K,Bool},  # isfwd[w] = true|false: ∂w is forward|backward difference
    ∆l⁻¹::NTuple{K,AbstractVector{<:Number}},  # ∆l⁻¹[w]: inverse of distances between grid planes in w-direction
    isbloch::SVector{K,Bool},  # boundary conditions in K dimensions
    e⁻ⁱᵏᴸ::SVector{K,<:Number};  # Bloch phase factors in K dimensions
    cmp_in=1:K,  # integer or vector of integers in 1:K; input components for which divergence is taken; can be used to, e.g.,  perform 2D divergence for 3D vector field; does not change number of columns (input remains K-dimensional)
    α::Number=1,  # scale factor to multiply to result before adding it to g: g += α ∇⋅F
    overwrite::Bool=true  # false to add result to output storage g
) where {K,K₊₁}
    @assert K₊₁==K+1
    @argcheck allunique(cmp_in)

    # Accumurate partial derivatives.
    for nw = cmp_in  # direction of differentiation
        nu = nw  # component of input field to feed to ∂w
        Fᵤ = selectdim(F, K₊₁, nu)  # nu-th component of input field
        apply_∂!(g, Fᵤ, nw, isfwd[nw], ∆l⁻¹[nw], isbloch[nw], e⁻ⁱᵏᴸ[nw]; α, overwrite=(nw==cmp_in[1]))  # g += α ∂Fᵤ/∂w
    end

    return nothing
end

# Concrete implementation for a vector field output.
# This is not necessary because we can simply pass g = @view(G[:,:,:,cmp_out]) to apply_divg!(),
# but it is implemented to be aligned with create_divg().
#
# Note that overwrite = true only overwrites cmp_out-portion of G, not the entire G.
apply_divg!(
    G::AbstractArray{<:Number,K₊₁},  # output array of scalar; in 3D, g[i,j,k] is g at (i,j,k)
    F::AbstractArray{<:Number,K₊₁},  # input field; in 3D, F[i,j,k,w] is w-component of F at (i,j,k)
    isfwd::SVector{K,Bool},  # isfwd[w] = true|false: ∂w is forward|backward difference
    ∆l⁻¹::NTuple{K,AbstractVector{<:Number}},  # ∆l⁻¹[w]: inverse of distances between grid planes in w-direction
    isbloch::SVector{K,Bool},  # boundary conditions in K dimensions
    e⁻ⁱᵏᴸ::SVector{K,<:Number};  # Bloch phase factors in K dimensions
    cmp_out::Integer,  # creates square matrix putting scalar funciton on cmp_out-th Cartesian direction of output vector field
    kwargs...
) where {K,K₊₁} = (
    g = @view(G[:,:,:,cmp_out]);
    apply_divg!(g, F, isfwd, ∆l⁻¹, isbloch, e⁻ⁱᵏᴸ; kwargs...)
)
