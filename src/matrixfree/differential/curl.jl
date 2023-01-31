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
            isfwd::AbsVecBool,  # isfwd[w] = true|false: create ∂w by forward|backward difference
            ∆l⁻¹::NTuple{K,Number}=ntuple(k->1.0,length(isfwd)),  # ∆l⁻¹[w]: inverse of uniform distance between grid planes in x-direction
            isbloch::AbsVecBool=fill(true,K),  # boundary conditions in x, y, z
            e⁻ⁱᵏᴸ::AbsVecNumber=ones(Int,K);  # Bloch phase factor in x, y, z
            cmp_shp::AbsVecInteger=1:K,
            cmp_out::AbsVecInteger=1:size(G,K₊₁),
            cmp_in::AbsVecInteger=1:size(F,K₊₁),
            α::Number=1,  # scale factor to multiply to result before adding it to G: G += α ∇×F
            overwrite::Bool=true  # false to add result to output storage G
            ) where {K,K₊₁} =
    (N = size(G)[1:K]; apply_curl!(G, F, isfwd, fill.(∆l⁻¹,N), isbloch, e⁻ⁱᵏᴸ; cmp_shp, cmp_out, cmp_in, α, overwrite))

# Wrapper for converting AbstractVector's to SVec's
apply_curl!(G::AbsArrNumber{K₊₁},  # output field; G[i,j,k,w] is w-component of G at (i,j,k)
            F::AbsArrNumber{K₊₁},  # input field; F[i,j,k,w] is w-component of F at (i,j,k)
            isfwd::AbsVecBool,  # isfwd[w] = true|false: create ∂w by forward|backward difference
            ∆l⁻¹::NTuple{K,AbsVecNumber},  # ∆l⁻¹[w]: inverse of distances between grid planes in x-direction
            isbloch::AbsVecBool=fill(true,K),  # boundary conditions in x, y, z
            e⁻ⁱᵏᴸ::AbsVecNumber=ones(Int,K);  # Bloch phase factor in x, y, z
            cmp_shp::AbsVecInteger=1:K,
            cmp_out::AbsVecInteger=1:size(G,K₊₁),
            cmp_in::AbsVecInteger=1:size(F,K₊₁),
            α::Number=1,  # scale factor to multiply to result before adding it to G: G += α ∇×F
            overwrite::Bool=true  # false to add result to output storage G
            ) where {K,K₊₁} =
    # I should not cast e⁻ⁱᵏᴸ into a complex vector, because then the entire curl matrix
    # becomes a complex matrix.  Sometimes I want to keep it real (e.g., when no PML and
    # Bloch phase factors are used).  In fact, this is the reason why I accept e⁻ⁱᵏᴸ instead
    # of constructing it from k and L as exp.(-im .* k .* L), which is always complex even
    # if k = 0.
    #
    # I should not cast ∆l⁻¹ to a vector of any specific type (e.g., Float, ComplexF), either,
    # because sometimes I would want to even create an integral curl operator.
    (Kout = length(cmp_out); Kin = length(cmp_in);
     apply_curl!(G, F, SVec{K}(isfwd), ∆l⁻¹, SVec{K}(isbloch), SVec{K}(e⁻ⁱᵏᴸ);
                 cmp_shp=SInt{K}(cmp_shp), cmp_out=SInt{Kout}(cmp_out), cmp_in=SInt{Kin}(cmp_in), α, overwrite))

# Concrete implementation
function apply_curl!(G::AbsArrNumber{K₊₁},  # output field; G[i,j,k,w] is w-component of G at (i,j,k)
                     F::AbsArrNumber{K₊₁},  # input field; F[i,j,k,w] is w-component of F at (i,j,k)
                     isfwd::SBool{K},  # isfwd[w] = true|false: create ∂w by forward|backward difference
                     ∆l⁻¹::NTuple{K,AbsVecNumber},  # ∆l⁻¹[w]: inverse of distances between grid planes in x-direction
                     isbloch::SBool{K},  # boundary conditions in x, y, z
                     e⁻ⁱᵏᴸ::SNumber{K};  # Bloch phase factor in x, y, z
                     cmp_shp::SInt{K}=SVec(ntuple(identity, Val(K))),
                     cmp_out::SInt{Kout}=SVec(ntuple(identity, size(G,K₊₁))),
                     cmp_in::SInt{Kin}=SVec(ntuple(identity, size(G,K₊₁))),
                     α::Number=1,  # scale factor to multiply to result before adding it to G: G += α ∇×F
                     overwrite::Bool=true  # false to add result to output storage Gᵥ
                     ) where {K,K₊₁,Kout,Kin}
    @assert K₊₁ == K+1

    for ind_nv = 1:Kout
        nv = cmp_out[ind_nv]  # Cartesian compotent of output field
        Gᵥ = selectdim(G, K₊₁, ind_nv)  # v-component of output field

        overwrite && (Gᵥ .= 0)
        for ind_nu = 1:Kin
            nu = cmp_in[ind_nu]  # Cartesian compotent of input field
            parity = CURL_BLK[nv,nu]  # CURL_BLK defined in matrix/curl.jl

            iszero(parity) && continue

            nw = 6 - nv - nu  # direction of differentiation; 6 = 1 + 2 + 3
            is_nw = cmp_shp.==nw
            sum(is_nw)==1 || @error "cmp_shp = $cmp_shp does not have one and only one nw = $nw"
            ind_nw = findfirst(is_nw)

            Fᵤ = selectdim(F, K₊₁, ind_nu)  # u-component of input field
            apply_∂!(Gᵥ, Fᵤ, ind_nw, isfwd[ind_nw], ∆l⁻¹[ind_nw], isbloch[ind_nw], e⁻ⁱᵏᴸ[ind_nw], α=parity*α, overwrite=false)  # Gᵥ += α (±∂Fᵤ/∂w)
        end
    end

    return nothing
end
