# Assumes the space dimension and field dimension are the same.  In other words, when the
# space coordinate indices are (i,j,k), then the field has three vector components.
# Therefore, for the output field array G[i,j,k,w], we assume w = 1:3.

# The functions add the calculated values to the existing values of the output array.
# Therefore, if the derivative values themselves are desired, pass the output array
# initialized with zeros.

export apply_grad!

# Wrapper to apply the discrete gradient by default
#
# Note that when the positional arguments are SVector's, this function may call the concrete
# implementation directly instead of the next wrapper function taking ∆l⁻¹ even if the
# keyword arguments are not SVector's, because methods are dispatched based only on the
# types of positional arguments.  Therefore, we convert the keyword arguments into SVector's.
apply_grad!(G::AbsArrNumber,  # output field; in 3D, G[i,j,k,w] is w-component of G at (i,j,k)
            f::AbsArrNumber,  # input array of scalar; in 3D, f[i,j,k] is f at (i,j,k)
            ::Val{OP},  # Val(:(=)) or Val(:(+=)): set (=) or add (+=) operator to use
            isfwd::AbsVecBool,  # isfwd[w] = true|false: create ∂w by forward|backward difference
            ∆l⁻¹::Tuple{Vararg{Number}}=ntuple(x->1.0,length(isfwd)),  # ∆l⁻¹[w]: inverse of uniform distance between grid planes in w-direction
            isbloch::AbsVecBool=fill(true,length(isfwd)),  # boundary conditions in K dimensions
            e⁻ⁱᵏᴸ::AbsVecNumber=ones(length(isfwd));  # Bloch phase factors in K dimensions
            α::Number=1.0  # scale factor to multiply to result before adding it to g: g += α ∇⋅F
            ) where {OP} =
    (N = size(f); K = length(isfwd); apply_grad!(G, f, Val(OP), isfwd, fill.(∆l⁻¹,N), isbloch, e⁻ⁱᵏᴸ; α))

# Wrapper for converting AbstractVector's to SVec's
apply_grad!(G::AbsArrNumber,  # output field; in 3D, G[i,j,k,w] is w-component of G at (i,j,k)
            f::AbsArrNumber,  # input array of scalar; in 3D, f[i,j,k] is f at (i,j,k)
            ::Val{OP},  # Val(:(=)) or Val(:(+=)): set (=) or add (+=) operator to use
            isfwd::AbsVecBool,  # isfwd[w] = true|false: create ∂w by forward|backward difference
            ∆l⁻¹::NTuple{K,AbsVecNumber},  # ∆l⁻¹[w]: inverse of distances between grid planes in x-direction
            isbloch::AbsVecBool=fill(true,K),  # boundary conditions in K dimensions
            e⁻ⁱᵏᴸ::AbsVecNumber=ones(K);  # Bloch phase factors in K dimensions
            α::Number=1.0  # scale factor to multiply to result before adding it to g: g += α ∇⋅F
            ) where {K,OP} =
    apply_grad!(G, f, Val(OP), SBool{K}(isfwd), ∆l⁻¹, SBool{K}(isbloch), SVec{K}(e⁻ⁱᵏᴸ); α)

# Concrete implementation
function apply_grad!(G::AbsArrNumber{K₊₁},  # output field; in 3D, G[i,j,k,w] is w-component of G at (i,j,k)
                     f::AbsArrNumber{K},  # input array of scalar; in 3D, f[i,j,k] is f at (i,j,k)
                     ::Val{OP},  # Val(:(=)) or Val(:(+=)): set (=) or add (+=) operator to use
                     isfwd::SBool{K},  # isfwd[w] = true|false: create ∂w by forward|backward difference
                     ∆l⁻¹::NTuple{K,AbsVecNumber},  # ∆l[w]: inverse of distances between grid planes in x-direction
                     isbloch::SBool{K},  # boundary conditions in K dimensions
                     e⁻ⁱᵏᴸ::SNumber{K};  # Bloch phase factors in K dimensions
                     α::Number=1.0  # scale factor to multiply to result before adding it to g: g += α ∇⋅F
                     ) where {K,K₊₁,OP}
    @assert K₊₁==K+1

    for nw = 1:K  # direction of differentiation
        nv = nw  # component of output field generated by ∂w (index of matrix block)
        Gv = selectdim(G, K₊₁, nv)  # nv-th component of output field
        apply_∂!(Gv, f, Val(OP), nw, isfwd[nw], ∆l⁻¹[nw], isbloch[nw], e⁻ⁱᵏᴸ[nw]; α) # Gv += α ∂f/∂w
    end

    return nothing
end
