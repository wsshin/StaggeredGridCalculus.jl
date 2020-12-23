# Assumes the space dimension and field dimension are the same.  In other words, when the
# space coordinate indices are (i,j,k), then the field has three vector components.
# Therefore, for the input field array F[i,j,k,w], we assume w = 1:3.

# The functions add the calculated values to the existing values of the output array.
# Therefore, if the derivative values themselves are desired, pass the output array
# initialized with zeros.

export apply_divg!

apply_divg!(g::AbsArrNumber,  # output array of scalar; in 3D, g[i,j,k] is g at (i,j,k)
            F::AbsArrNumber,  # input field; in 3D, F[i,j,k,w] is w-component of F at (i,j,k)
            isfwd::AbsVecBool,  # isfwd[w] = true|false: create ∂w by forward|backward difference
            ∆l⁻¹::Tuple{Vararg{AbsVecNumber}}=ones.(size(g)),  # ∆l⁻¹[w]: inverse of distances between grid planes in x-direction
            isbloch::AbsVecBool=fill(true,length(isfwd)),  # boundary conditions in K dimensions
            e⁻ⁱᵏᴸ::AbsVecNumber=ones(length(isfwd));  # Bloch phase factors in K dimensions
            permute∂::AbsVecInt=1:length(isfwd),  # permute∂[w]: location of ∂w block
            scale∂::AbsVecNumber=ones(length(isfwd)),  # scale∂[w]: scale factor to multiply to ∂w
            α::Number=1.0) =  # scale factor to multiply to result before adding it to g: g += α ∇⋅F
    (K = length(isfwd); apply_divg!(g, F, SVector{K}(isfwd), ∆l⁻¹, SVector{K}(isbloch), SVector{K}(e⁻ⁱᵏᴸ), permute∂=SVector{K}(permute∂), scale∂=SVector{K}(scale∂), α=α))

function apply_divg!(g::AbsArrNumber{K},  # output array of scalar; in 3D, g[i,j,k] is g at (i,j,k)
                     F::AbsArrNumber{K₊₁},  # input field; in 3D, F[i,j,k,w] is w-component of F at (i,j,k)
                     isfwd::SBool{K},  # isfwd[w] = true|false: create ∂w by forward|backward difference
                     ∆l⁻¹::NTuple{K,AbsVecNumber},  # ∆l[w]: inverse of distances between grid planes in x-direction
                     isbloch::SBool{K},  # boundary conditions in K dimensions
                     e⁻ⁱᵏᴸ::SNumber{K};  # Bloch phase factors in K dimensions
                     permute∂::SInt{K}=SVector(ntuple(identity, Val(K))),  # permute∂[w]: location of ∂w block
                     scale∂::SNumber{K}=SVector(ntuple(k->1.0, Val(K))),  # scale∂[w]: scale factor to multiply to ∂w
                     α::Number=1.0  # scale factor to multiply to result before adding it to g: g += α ∇⋅F
                     ) where {K,K₊₁}
    @assert(K₊₁==K+1)

    for nw = 1:K  # direction of differentiation
        nu = permute∂[nw]  # component of input field to feed to ∂w
        Fu = selectdim(F, K₊₁, nu)  # nu-th component of input field
        @spawn apply_∂!(g, Fu, nw, isfwd[nw], ∆l⁻¹[nw], isbloch[nw], e⁻ⁱᵏᴸ[nw], α=α*scale∂[nw])  # g += (α scale∂[w]) ∂Fu/∂w
    end

    return nothing
end
