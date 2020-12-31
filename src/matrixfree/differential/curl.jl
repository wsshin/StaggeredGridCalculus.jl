# The functions add the calculated values to the existing values of the output array.
# Therefore, if the derivative values themselves are desired, pass the output array
# initialized with zeros.

# To-dos
# - Test if using a separate index vector (which is an identity map) to iterate over a 3D
# array makes iteration significantly slower.  This is to simulate the case of FEM.

export apply_curl!

# Wrapper for converting AbstractVector's to SVector's
apply_curl!(G::AbsArrNumber{4},  # output field; G[i,j,k,w] is w-component of G at (i,j,k)
            F::AbsArrNumber{4},  # input field; F[i,j,k,w] is w-component of F at (i,j,k)
            ::Val{OP},  # Val(:(=)) or Val(:(+=)): set (=) or add (+=) operator to use
            isfwd::AbsVecBool,  # isfwd[w] = true|false: create ∂w by forward|backward difference
            ∆l⁻¹::Tuple3{AbsVecNumber}=ones.(size(F)[1:3]),  # ∆l⁻¹[w]: inverse of distances between grid planes in x-direction
            isbloch::AbsVecBool=fill(true,length(isfwd)),  # boundary conditions in x, y, z
            e⁻ⁱᵏᴸ::AbsVecNumber=ones(length(isfwd));  # Bloch phase factor in x, y, z
            α::Number=1.0  # scale factor to multiply to result before adding it to G: G += α ∇×F
            ) where {OP} =
    # I should not cast e⁻ⁱᵏᴸ into a complex vector, because then the entire curl matrix
    # becomes a complex matrix.  Sometimes I want to keep it real (e.g., when no PML and
    # Bloch phase factors are used).  In fact, this is the reason why I accept e⁻ⁱᵏᴸ instead
    # of constructing it from k and L as exp.(-im .* k .* L), which is always complex even
    # if k = 0.
    #
    # I should not cast ∆l⁻¹ to a vector of any specific type (e.g., Float, CFloat), either,
    # because sometimes I would want to even create an integral curl operator.
    (K = length(isfwd); apply_curl!(G, F, Val(OP), SVector{K}(isfwd), ∆l⁻¹, SVector{K}(isbloch), SVector{K}(e⁻ⁱᵏᴸ), α=α))

# Concrete implementation
function apply_curl!(G::AbsArrNumber{4},  # output field; G[i,j,k,w] is w-component of G at (i,j,k)
                     F::AbsArrNumber{4},  # input field; F[i,j,k,w] is w-component of F at (i,j,k)
                     ::Val{OP},  # Val(:(=)) or Val(:(+=)): set (=) or add (+=) operator to use
                     isfwd::SBool{3},  # isfwd[w] = true|false: create ∂w by forward|backward difference
                     ∆l⁻¹::Tuple3{AbsVecNumber},  # ∆l⁻¹[w]: inverse of distances between grid planes in x-direction
                     isbloch::SBool{3},  # boundary conditions in x, y, z
                     e⁻ⁱᵏᴸ::SNumber{3};  # Bloch phase factor in x, y, z
                     α::Number=1.0  # scale factor to multiply to result before adding it to G: G += α ∇×F
                     ) where {OP}
    for nv = 1:3  # Cartesian compotent of output field
        Gv = @view G[:,:,:,nv]  # v-component of output field

        parity = 1
        nw = next1(nv)  # direction of differentiation
        nu = 6 - nv - nw  # Cantesian component of input field; 6 = nX + nY + nZ
        Fu = @view F[:,:,:,nu]  # u-component of input field

        apply_∂!(Gv, Fu, Val(OP), nw, isfwd[nw], ∆l⁻¹[nw], isbloch[nw], e⁻ⁱᵏᴸ[nw], α=parity*α)  # Gv += α (±∂Fu/∂w)
    end

    for nv = 1:3  # Cartesian compotent of output field
        Gv = @view G[:,:,:,nv]  # v-component of output field

        parity = -1
        nw = prev1(nv)  # direction of differentiation
        nu = 6 - nv - nw  # Cantesian component of input field; 6 = nX + nY + nZ
        Fu = @view F[:,:,:,nu]  # u-component of input field

        apply_∂!(Gv, Fu, Val(:(+=)), nw, isfwd[nw], ∆l⁻¹[nw], isbloch[nw], e⁻ⁱᵏᴸ[nw], α=parity*α)  # Gv += α (±∂Fu/∂w)
    end

    # @sync begin
    #     @spawn apply_∂!(@view(G[:,:,:,1]), @view(F[:,:,:,3]), Val(OP), 2, isfwd[2], ∆l⁻¹[2], isbloch[2], e⁻ⁱᵏᴸ[2], α=α)  # Gv += α (±∂Fu/∂w)
    #     @spawn apply_∂!(@view(G[:,:,:,2]), @view(F[:,:,:,1]), Val(OP), 3, isfwd[3], ∆l⁻¹[3], isbloch[3], e⁻ⁱᵏᴸ[3], α=α)  # Gv += α (±∂Fu/∂w)
    #     @spawn apply_∂!(@view(G[:,:,:,3]), @view(F[:,:,:,2]), Val(OP), 1, isfwd[1], ∆l⁻¹[1], isbloch[1], e⁻ⁱᵏᴸ[1], α=α)  # Gv += α (±∂Fu/∂w)
    # end
    #
    # @sync begin
    #     @spawn apply_∂!(@view(G[:,:,:,1]), @view(F[:,:,:,2]), Val(:(+=)), 3, isfwd[3], ∆l⁻¹[3], isbloch[3], e⁻ⁱᵏᴸ[3], α=-α)  # Gv += α (±∂Fu/∂w)
    #     @spawn apply_∂!(@view(G[:,:,:,2]), @view(F[:,:,:,3]), Val(:(+=)), 1, isfwd[1], ∆l⁻¹[1], isbloch[1], e⁻ⁱᵏᴸ[1], α=-α)  # Gv += α (±∂Fu/∂w)
    #     @spawn apply_∂!(@view(G[:,:,:,3]), @view(F[:,:,:,1]), Val(:(+=)), 2, isfwd[2], ∆l⁻¹[2], isbloch[2], e⁻ⁱᵏᴸ[2], α=-α)  # Gv += α (±∂Fu/∂w)
    # end

    return nothing
end
