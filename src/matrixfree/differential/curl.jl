# The functions add the calculated values to the existing values of the output array.
# Therefore, if the derivative values themselves are desired, pass the output array
# initialized with zeros.

# To-dos
# - Test if using a separate index vector (which is an identity map) to iterate over a 3D
# array makes iteration significantly slower.  This is to simulate the case of FEM.

export apply_curl!

# Wrapper to apply the discrete curl by default
apply_curl!(G::AbsArrNumber{K₊₁},  # output field; G[i,j,k,w] is w-component of G at (i,j,k)
            F::AbsArrNumber{K₊₁},  # input field; F[i,j,k,w] is w-component of F at (i,j,k)
            ::Val{OP},  # Val(:(=)) or Val(:(+=)): set (=) or add (+=) operator to use
            isfwd::AbsVecBool,  # isfwd[w] = true|false: create ∂w by forward|backward difference
            ∆l⁻¹::NTuple{K,Number}=ntuple(k->1.0,length(isfwd)),  # ∆l⁻¹[w]: inverse of uniform distance between grid planes in x-direction
            isbloch::AbsVecBool=fill(true,K),  # boundary conditions in x, y, z
            e⁻ⁱᵏᴸ::AbsVecNumber=ones(K);  # Bloch phase factor in x, y, z
            cmp_shp::AbsVecInteger=1:K,
            cmp_out::AbsVecInteger=1:size(G,K₊₁),
            cmp_in::AbsVecInteger=1:size(F,K₊₁),
            α::Number=1.0  # scale factor to multiply to result before adding it to G: G += α ∇×F
            ) where {K,K₊₁,OP} =
    (N = size(G)[1:K]; apply_curl!(G, F, Val(OP), isfwd, fill.(∆l⁻¹,N), isbloch, e⁻ⁱᵏᴸ; cmp_shp, cmp_out, cmp_in, α))

# Wrapper for converting AbstractVector's to SVec's
apply_curl!(G::AbsArrNumber{K₊₁},  # output field; G[i,j,k,w] is w-component of G at (i,j,k)
            F::AbsArrNumber{K₊₁},  # input field; F[i,j,k,w] is w-component of F at (i,j,k)
            ::Val{OP},  # Val(:(=)) or Val(:(+=)): set (=) or add (+=) operator to use
            isfwd::AbsVecBool,  # isfwd[w] = true|false: create ∂w by forward|backward difference
            ∆l⁻¹::NTuple{K,AbsVecNumber},  # ∆l⁻¹[w]: inverse of distances between grid planes in x-direction
            isbloch::AbsVecBool=fill(true,K),  # boundary conditions in x, y, z
            e⁻ⁱᵏᴸ::AbsVecNumber=ones(K);  # Bloch phase factor in x, y, z
            cmp_shp::AbsVecInteger=1:K,
            cmp_out::AbsVecInteger=1:size(G,K₊₁),
            cmp_in::AbsVecInteger=1:size(F,K₊₁),
            α::Number=1.0  # scale factor to multiply to result before adding it to G: G += α ∇×F
            ) where {K,K₊₁,OP} =
    # I should not cast e⁻ⁱᵏᴸ into a complex vector, because then the entire curl matrix
    # becomes a complex matrix.  Sometimes I want to keep it real (e.g., when no PML and
    # Bloch phase factors are used).  In fact, this is the reason why I accept e⁻ⁱᵏᴸ instead
    # of constructing it from k and L as exp.(-im .* k .* L), which is always complex even
    # if k = 0.
    #
    # I should not cast ∆l⁻¹ to a vector of any specific type (e.g., Float, CFloat), either,
    # because sometimes I would want to even create an integral curl operator.
    (Kout = length(cmp_out); Kin = length(cmp_in);
     apply_curl!(G, F, Val(OP), SVec{K}(isfwd), ∆l⁻¹, SVec{K}(isbloch), SVec{K}(e⁻ⁱᵏᴸ);
                 cmp_shp=SInt{K}(cmp_shp), cmp_out=SInt{Kout}(cmp_out), cmp_in=SInt{Kin}(cmp_in), α))

# Concrete implementation
function apply_curl!(G::AbsArrNumber{K₊₁},  # output field; G[i,j,k,w] is w-component of G at (i,j,k)
                     F::AbsArrNumber{K₊₁},  # input field; F[i,j,k,w] is w-component of F at (i,j,k)
                     ::Val{OP},  # Val(:(=)) or Val(:(+=)): set (=) or add (+=) operator to use
                     isfwd::SBool{K},  # isfwd[w] = true|false: create ∂w by forward|backward difference
                     ∆l⁻¹::NTuple{K,AbsVecNumber},  # ∆l⁻¹[w]: inverse of distances between grid planes in x-direction
                     isbloch::SBool{K},  # boundary conditions in x, y, z
                     e⁻ⁱᵏᴸ::SNumber{K};  # Bloch phase factor in x, y, z
                     cmp_shp::SInt{K}=SVec(ntuple(identity, Val(K))),
                     cmp_out::SInt{Kout}=SVec(ntuple(identity, size(G,K₊₁))),
                     cmp_in::SInt{Kin}=SVec(ntuple(identity, size(G,K₊₁))),
                     α::Number=1.0  # scale factor to multiply to result before adding it to G: G += α ∇×F
                     ) where {K,K₊₁,Kout,Kin,OP}
    @assert K₊₁ == K+1

    for ind_nv = 1:Kout
        nv = cmp_out[ind_nv]  # Cartesian compotent of output field
        Gv = selectdim(G, K₊₁, ind_nv)  # v-component of output field

        op = Val(OP)
        for ind_nu = 1:Kin
            nu = cmp_in[ind_nu]  # Cartesian compotent of input field
            parity = CURL_BLK[nv,nu]  # CURL_BLK defined in matrix/curl.jl

            if iszero(parity)
                if op == Val(:(=))
                    Gv .= 0
                    op = Val(:(+=))
                end
                continue
            end

            nw = 6 - nv - nu  # direction of differentiation; 6 = 1 + 2 + 3
            is_nw = cmp_shp.==nw
            sum(is_nw)==1 || @error "cmp_shp = $cmp_shp does not have one and only one nw = $nw"
            ind_nw = findfirst(is_nw)

            Fu = selectdim(F, K₊₁, ind_nu)  # u-component of input field
            apply_∂!(Gv, Fu, op, ind_nw, isfwd[ind_nw], ∆l⁻¹[ind_nw], isbloch[ind_nw], e⁻ⁱᵏᴸ[ind_nw], α=parity*α)  # Gv += α (±∂Fu/∂w)

            op = Val(:(+=))

            # # Equivalent to the above code:
            # if op == Val(:(=))
            #     if iszero(parity)
            #         Gv .= 0
            #     else
            #         nw = 6 - nv - nu  # direction of differentiation; 6 = 1 + 2 + 3
            #         is_nw = cmp_shp.==nw
            #         @assert sum(is_nw)==1  # in cmp_shp, one and only one entry is nw
            #         ind_nw = findfirst(is_nw)
            #
            #         Fu = @view F[:,:,:,ind_nu]  # u-component of input field
            #         apply_∂!(Gv, Fu, op, ind_nw, isfwd[ind_nw], ∆l⁻¹[ind_nw], isbloch[ind_nw], e⁻ⁱᵏᴸ[ind_nw], α=parity*α)  # Gv += α (±∂Fu/∂w)
            #     end
            #
            #     op = Val(:(+=))
            # else  # op == Val(:(+=))
            #     if iszero(parity)
            #         # Do nothing.
            #     else
            #         nw = 6 - nv - nu  # direction of differentiation; 6 = 1 + 2 + 3
            #         is_nw = cmp_shp.==nw
            #         @assert sum(is_nw)==1  # in cmp_shp, one and only one entry is nw
            #         ind_nw = findfirst(is_nw)
            #
            #         Fu = @view F[:,:,:,ind_nu]  # u-component of input field
            #         apply_∂!(Gv, Fu, op, ind_nw, isfwd[ind_nw], ∆l⁻¹[ind_nw], isbloch[ind_nw], e⁻ⁱᵏᴸ[ind_nw], α=parity*α)  # Gv += α (±∂Fu/∂w)
            #     end
            # end
        end
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
