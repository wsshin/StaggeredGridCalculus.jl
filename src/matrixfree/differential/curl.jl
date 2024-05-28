# The functions add the calculated values to the existing values of the output array.
# Therefore, if the derivative values themselves are desired, pass the output array
# initialized with zeros.

# To-dos
# - Test if using a separate index vector (which is an identity map) to iterate over a 3D
# array makes iteration significantly slower.  This is to simulate the case of FEM.

export apply_curl!

# Wrapper to apply the discrete curl by default
apply_curl!(
    G::AbstractArray{<:Number,K₊₁},  # output field; G[i,j,k,w] is w-component of G at (i,j,k)
    F::AbstractArray{<:Number,K₊₁},  # input field; F[i,j,k,w] is w-component of F at (i,j,k)
    isfwd::AbstractVector{Bool},  # isfwd[w] = true|false: ∂w is forward|backward difference
    ∆l⁻¹::NTuple{K,Number}=ntuple(k->1,length(isfwd)),  # ∆l⁻¹[w]: inverse of distance (not distances) between grid planes in w-direction; assume ordered by right-hand rule
    isbloch::AbstractVector{Bool}=ones(Bool,K),  # isblock[w]: boundary conditions in w-direction
    e⁻ⁱᵏᴸ::AbstractVector{<:Number}=ones(Int,K);  # e⁻ⁱᵏᴸ[w]: Bloch phase factor in w-direction
    kwargs...
) where {K,K₊₁} =
    (N = size(G)[1:K]; apply_curl!(G, F, isfwd, fill.(∆l⁻¹,N), isbloch, e⁻ⁱᵏᴸ; kwargs...))

# Wrapper for converting AbstractVector's to SVector's
apply_curl!(
    G::AbstractArray{<:Number,K₊₁},  # output field; G[i,j,k,w] is w-component of G at (i,j,k)
    F::AbstractArray{<:Number,K₊₁},  # input field; F[i,j,k,w] is w-component of F at (i,j,k)
    isfwd::AbstractVector{Bool},  # isfwd[w] = true|false: ∂w is forward|backward difference
    ∆l⁻¹::NTuple{K,AbstractVector{<:Number}},  # ∆l⁻¹[w]: inverse of distances between grid planes in w-direction; assume ordered by right-hand rule
    isbloch::AbstractVector{Bool}=ones(Bool,K),  # isblock[w]: boundary conditions in w-direction
    e⁻ⁱᵏᴸ::AbstractVector{<:Number}=ones(Int,K);  # e⁻ⁱᵏᴸ[w]: Bloch phase factor in w-direction
    cmp_out::AbstractVector{<:Integer}=1:K,  # Cartesian directions of output field components; not necessarily 1:K, because, e.g., Ez can be defined on xy-shape
    cmp_in::AbstractVector{<:Integer}=1:K,  # Cartesian directions of input field; not necessarily 1:K, because, e.g., Ez can be defined on xy-shape
    α::Number=1,  # scale factor to multiply to result before adding it to G: G += α ∇×F
    overwrite::Bool=true  # false to add result to output storage G
) where {K,K₊₁} = (
    # I should not cast e⁻ⁱᵏᴸ into a complex vector, because then the entire curl matrix
    # becomes a complex matrix.  Sometimes I want to keep it real (e.g., when no PML and
    # Bloch phase factors are used).  In fact, this is the reason why I accept e⁻ⁱᵏᴸ instead
    # of constructing it from k and L as exp.(-im .* k .* L), which is always complex even
    # if k = 0.
    #
    # I should not cast ∆l⁻¹ to a vector of any specific type (e.g., Float64, ComplexF64), either,
    # because sometimes I would want to even create an integral curl operator.
    Kout = length(cmp_out);
    Kin = length(cmp_in);
    apply_curl!(
        G, F, SVector{K}(isfwd), ∆l⁻¹, SVector{K}(isbloch), SVector{K}(e⁻ⁱᵏᴸ);
        cmp_out=SVector{Kout}(cmp_out), cmp_in=SVector{Kin}(cmp_in), α, overwrite)
)

# Concrete implementation
function apply_curl!(
    G::AbstractArray{<:Number,K₊₁},  # output field; G[i,j,k,w] is w-component of G at (i,j,k)
    F::AbstractArray{<:Number,K₊₁},  # input field; F[i,j,k,w] is w-component of F at (i,j,k)
    isfwd::SVector{K,Bool},  # isfwd[w] = true|false: ∂w is forward|backward difference
    ∆l⁻¹::NTuple{K,AbstractVector{<:Number}},  # ∆l⁻¹[w]: inverse of distances between grid planes in w-direction; assume ordered by right-hand rule
    isbloch::SVector{K,Bool},  # isblock[w]: boundary conditions in w-direction
    e⁻ⁱᵏᴸ::SVector{K,<:Number};  # e⁻ⁱᵏᴸ[w]: Bloch phase factor in w-direction
    cmp_out::SVector{Kout,Int}=SVector{K}(1:K),  # Cartesian directions of output field components; not necessarily 1:K, because, e.g., Ez can be defined on xy-shape
    cmp_in::SVector{Kin,Int}=SVector{K}(1:K),  # Cartesian directions of input field; not necessarily 1:K, because, e.g., Ez can be defined on xy-shape
    α::Number=1,  # scale factor to multiply to result before adding it to G: G += α ∇×F
    overwrite::Bool=true  # false to add result to output storage G
) where {K,K₊₁,Kout,Kin}
    @assert K₊₁ == K+1

    for (ind_nv, nv) = enumerate(cmp_out)  # nv is direction of output field
        Gᵥ = selectdim(G, K₊₁, ind_nv)  # v-component of output field

        overwrite && (Gᵥ .= 0)
        for (ind_nu, nu) = enumerate(cmp_in)  # nu is direction of input field
            # ∇×F = 1̂ (∂₂F₃ - ∂₃F₂) + 2̂ (∂₃F₁ - ∂₁F₃) + 3̂ (∂₁F₂ - ∂₂F₁)
            #
            # In matrix form,
            # ⎡ 0  -∂₃  ∂₂⎤
            # ⎢ ∂₃  0  -∂₁⎥
            # ⎣-∂₂  ∂₁  0 ⎦
            nu==nv && continue  # if zero block, skip to next nu

            # nw is the index of vector N.  N is needed by create_∂info().  If K = 2,
            # N = [Ny,Nz] and nw = 1 for y and nw = 2 for z.  In other words, the code is
            # xyz-agnostic: the 1st component of ∆l⁻¹ is along the 1st direction, the 2nd
            # component of ∆l⁻¹ is along the 2nd direction, etc, and they do not correspond
            # to x and y.
            nw = 6 - nv - nu  # direction of differentiation; 6 = 1 + 2 + 3
            @argcheck 1 ≤ nw ≤ K

            parity = mod(nv-nu+1, 3) - 1  # sign of (nv,nu)-entry of above matrix

            Fᵤ = selectdim(F, K₊₁, ind_nu)  # u-component of input field
            apply_∂!(Gᵥ, Fᵤ, nw, isfwd[nw], ∆l⁻¹[nw], isbloch[nw], e⁻ⁱᵏᴸ[nw], α=parity*α, overwrite=false)  # Gᵥ += α (±∂Fᵤ/∂w)
        end
    end

    return nothing
end
