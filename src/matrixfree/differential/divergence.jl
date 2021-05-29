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
apply_divg!(g::AbsArrNumber,  # output array of scalar; in 3D, g[i,j,k] is g at (i,j,k)
            F::AbsArrNumber,  # input field; in 3D, F[i,j,k,w] is w-component of F at (i,j,k)
            ::Val{OP},  # Val(:(=)) or Val(:(+=)): set (=) or add (+=) operator to use
            isfwd::AbsVecBool,  # isfwd[w] = true|false: create ∂w by forward|backward difference
            ∆l⁻¹::Tuple{Vararg{Number}}=ntuple(x->1.0,length(isfwd)),  # ∆l⁻¹[w]: inverse of uniform distance between grid planes in w-direction
            isbloch::AbsVecBool=fill(true,length(isfwd)),  # boundary conditions in K dimensions
            e⁻ⁱᵏᴸ::AbsVecNumber=ones(length(isfwd));  # Bloch phase factors in K dimensions
            α::Number=1.0  # scale factor to multiply to result before adding it to g: g += α ∇⋅F
            ) where {OP} =
    (N = size(g); K = length(isfwd); apply_divg!(g, F, Val(OP), isfwd, fill.(∆l⁻¹,N), isbloch, e⁻ⁱᵏᴸ; α))

# Wrapper for converting AbstractVector's to SVec's
apply_divg!(g::AbsArrNumber,  # output array of scalar; in 3D, g[i,j,k] is g at (i,j,k)
            F::AbsArrNumber,  # input field; in 3D, F[i,j,k,w] is w-component of F at (i,j,k)
            ::Val{OP},  # Val(:(=)) or Val(:(+=)): set (=) or add (+=) operator to use
            isfwd::AbsVecBool,  # isfwd[w] = true|false: create ∂w by forward|backward difference
            ∆l⁻¹::NTuple{K,AbsVecNumber},  # ∆l⁻¹[w]: inverse of distances between grid planes in x-direction
            isbloch::AbsVecBool=fill(true,K),  # boundary conditions in K dimensions
            e⁻ⁱᵏᴸ::AbsVecNumber=ones(K);  # Bloch phase factors in K dimensions
            α::Number=1.0  # scale factor to multiply to result before adding it to g: g += α ∇⋅F
            ) where {K,OP} =
    apply_divg!(g, F, Val(OP), SBool{K}(isfwd), ∆l⁻¹, SBool{K}(isbloch), SVec{K}(e⁻ⁱᵏᴸ); α)

# Concrete implementation
function apply_divg!(g::AbsArrNumber{K},  # output array of scalar; in 3D, g[i,j,k] is g at (i,j,k)
                     F::AbsArrNumber{K₊₁},  # input field; in 3D, F[i,j,k,w] is w-component of F at (i,j,k)
                     ::Val{OP},  # Val(:(=)) or Val(:(+=)): set (=) or add (+=) operator to use
                     isfwd::SBool{K},  # isfwd[w] = true|false: create ∂w by forward|backward difference
                     ∆l⁻¹::NTuple{K,AbsVecNumber},  # ∆l[w]: inverse of distances between grid planes in x-direction
                     isbloch::SBool{K},  # boundary conditions in K dimensions
                     e⁻ⁱᵏᴸ::SNumber{K};  # Bloch phase factors in K dimensions
                     α::Number=1.0  # scale factor to multiply to result before adding it to g: g += α ∇⋅F
                     ) where {K,K₊₁,OP}
    @assert K₊₁==K+1

    # First partial derivative
    nw = 1
    nu = nw  # component of input field to feed to ∂w
    Fu = selectdim(F, K₊₁, nu)  # nu-th component of input field
    apply_∂!(g, Fu, Val(OP), nw, isfwd[nw], ∆l⁻¹[nw], isbloch[nw], e⁻ⁱᵏᴸ[nw]; α)  # g += α ∂Fu/∂w

    # Remaining partial derivatives
    for nw = 2:K  # direction of differentiation
        nu = nw  # component of input field to feed to ∂w
        Fu = selectdim(F, K₊₁, nu)  # nu-th component of input field
        apply_∂!(g, Fu, Val(:(+=)), nw, isfwd[nw], ∆l⁻¹[nw], isbloch[nw], e⁻ⁱᵏᴸ[nw]; α)  # g += α ∂Fu/∂w
    end

    return nothing
end
