export apply_curl!

# To-dos
# - Test if using a separate index vector (which is an identity map) to iterate over a 3D
# array makes iteration significantly slower.  This is to simulate the case of FEM.
apply_curl!(G::T,  # output field; G[i,j,k,w] is w-component of G at (i,j,k)
            F::T,  # input field; F[i,j,k,w] is w-component of F at (i,j,k)
            isfwd::AbsVecBool,  # isfwd[w] = true|false: create ∂w by forward|backward difference
            isbloch::AbsVecBool=fill(true,length(isfwd)),  # boundary conditions in x, y, z
            e⁻ⁱᵏᴸ::AbsVecNumber=ones(length(isfwd));  # Bloch phase factor in x, y, z
            α::Number=1  # scale factor to multiply to result before adding it to G: G += α ∇×F
            ) where {T<:AbsArrNumber{4}} =
    (∆l = ones.(size(F)[1:3]); apply_curl!(G, F, isfwd, ∆l, isbloch, e⁻ⁱᵏᴸ, α=α))

apply_curl!(G::T,  # output field; G[i,j,k,w] is w-component of G at (i,j,k)
            F::T,  # input field; F[i,j,k,w] is w-component of F at (i,j,k)
            isfwd::AbsVecBool,  # isfwd[w] = true|false: create ∂w by forward|backward difference
            ∆l::Tuple3{AbsVecNumber},  # ∆l[w]: distances between grid planes in x-direction
            isbloch::AbsVecBool=fill(true,length(isfwd)),  # boundary conditions in x, y, z
            e⁻ⁱᵏᴸ::AbsVecNumber=ones(length(isfwd));  # Bloch phase factor in x, y, z
            α::Number=1  # scale factor to multiply to result before adding it to G: G += α ∇×F
            ) where {T<:AbsArrNumber{4}} =
    # I should not cast e⁻ⁱᵏᴸ into a complex vector, because then the entire curl matrix
    # becomes a complex matrix.  Sometimes I want to keep it real (e.g., when no PML and
    # Bloch phase factors are used).  In fact, this is the reason why I accept e⁻ⁱᵏᴸ instead
    # of constructing it from k and L as exp.(-im .* k .* L), which is always complex even
    # if k = 0.
    #
    # I should not cast ∆l to a vector of any specific type (e.g., Float, CFloat), either,
    # because sometimes I would want to even create an integral curl operator.
    (K = length(isfwd); apply_curl!(G, F, SVector{K}(isfwd), ∆l, SVector{K}(isbloch), SVector{K}(e⁻ⁱᵏᴸ), α=α))

function apply_curl!(G::T,  # output field; G[i,j,k,w] is w-component of G at (i,j,k)
                     F::T,  # input field; F[i,j,k,w] is w-component of F at (i,j,k)
                     isfwd::SBool{3},  # isfwd[w] = true|false: create ∂w by forward|backward difference
                     ∆l::Tuple3{AbsVecNumber},  # ∆l[w]: distances between grid planes in x-direction
                     isbloch::SBool{3}=SVector(true,true,true),  # boundary conditions in x, y, z
                     e⁻ⁱᵏᴸ::SNumber{3}=SVector(1.0,1.0,1.0);  # Bloch phase factor in x, y, z
                     α::Number=1  # scale factor to multiply to result before adding it to G: G += α ∇×F
                     ) where {T<:AbsArrNumber{4}}
    for nv = 1:3  # Cartesian compotent of output field
        Gv = @view G[:,:,:,nv]  # v-component of output field
        parity = 1
        for nw = next2(nv)  # direction of differentiation
            nu = 6 - nv - nw  # Cantesian component of input field; 6 = nX + nY + nZ
            Fu = @view F[:,:,:,nu]  # u-component of input field

            # Need to avoid allocation in parity*∆l[nw]
            apply_∂!(Gv, Fu, nw, isfwd[nw], ∆l[nw], isbloch[nw], e⁻ⁱᵏᴸ[nw], α=parity*α)  # Gv += α (±∂Fu/∂w)
            parity = -1
        end
    end

    return nothing
end
