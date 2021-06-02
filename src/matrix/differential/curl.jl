export create_curl

# Wrapper to create the discrete curl by default
create_curl(isfwd::AbsVecBool,  # isfwd[w] = true|false: ∂w is forward|backward difference
            N::AbsVecInteger,  # size of grid
            ∆l⁻¹::NTuple{K,Number}=ntuple(k->1.0,length(isfwd)),  # ∆l⁻¹[w]: inverse of uniform distance between grid planes in x-direction
            isbloch::AbsVecBool=fill(true,K),  # boundary conditions in x, y, z
            e⁻ⁱᵏᴸ::AbsVecNumber=ones(K);  # Bloch phase factor in x, y, z
            cmp_shp::AbsVecInteger=1:K,
            cmp_out::AbsVecInteger=1:K,
            cmp_in::AbsVecInteger=1:K,
            order_cmpfirst::Bool=true  # true to use Cartesian-component-major ordering for more tightly banded matrix
            ) where {K} =
    # I should not cast e⁻ⁱᵏᴸ into a complex vector, because then the entire curl matrix
    # becomes a complex matrix.  Sometimes I want to keep it real (e.g., when no PML and
    # Bloch phase factors are used).  In fact, this is the reason why I accept e⁻ⁱᵏᴸ instead
    # of constructing it from k and L as exp.(-im .* k .* L), which is always complex even
    # if k = 0.
    #
    # I should not cast ∆l⁻¹ to a vector of any specific type (e.g., Float, CFloat), either,
    # because sometimes I would want to even create an integral curl operator.
    create_curl(isfwd, fill.(∆l⁻¹,(N...,)), isbloch, e⁻ⁱᵏᴸ; cmp_shp, cmp_out, cmp_in, order_cmpfirst)

# Wrapper to convert AbstractVector's to SVec's
create_curl(isfwd::AbsVecBool,  # isfwd[w] = true|false: ∂w is forward|backward difference
            ∆l⁻¹::NTuple{K,AbsVecNumber},  # ∆l⁻¹[w]: inverse of uniform distance between grid planes in x-direction
            isbloch::AbsVecBool=fill(true,K),  # boundary conditions in x, y, z
            e⁻ⁱᵏᴸ::AbsVecNumber=ones(K);  # Bloch phase factor in x, y, z
            cmp_shp::AbsVecInteger=1:K,
            cmp_out::AbsVecInteger=1:K,
            cmp_in::AbsVecInteger=1:K,
            order_cmpfirst::Bool=true  # true to use Cartesian-component-major ordering for more tightly banded matrix
            ) where {K} =
    (Kout = length(cmp_out); Kin = length(cmp_in);
     create_curl(SVec{K}(isfwd), ∆l⁻¹, SVec{K}(isbloch), SVec{K}(e⁻ⁱᵏᴸ);
                 cmp_shp=SInt{K}(cmp_shp), cmp_out=SInt{Kout}(cmp_out), cmp_in=SInt{Kin}(cmp_in), order_cmpfirst))

# ∇×F = x̂ (∂y Fz - ∂z Fy) + ŷ (∂z Fx - ∂x Fz) + ẑ (∂x Fy - ∂y Fx)
#
# In matrix form,
# ⎡ 0  -∂z  ∂y⎤
# ⎢ ∂z  0  -∂x⎥
# ⎣-∂y  ∂x  0 ⎦
#
# CURL_BLK takes only the signs of the blocks:
const CURL_BLK = SSInt{3,9}(0,  1, -1,  # 1st column (not row)
                           -1,  0,  1,  # 2nd column (not row)
                            1, -1,  0)  # 3rd column (not row)

function create_curl(isfwd::SBool{K},  # isfwd[w] = true|false: ∂w is forward|backward difference
                     ∆l⁻¹::NTuple{K,AbsVecNumber},  # ∆l⁻¹[w]: inverse of distances between grid planes in w-direction
                     isbloch::SBool{K},  # isblock[w]: boundary conditions in w-direction
                     e⁻ⁱᵏᴸ::SNumber{K};  # e⁻ⁱᵏᴸ[w]: Bloch phase factor in w-direction
                     cmp_shp::SInt{K}=SVec(ntuple(identity, Val(K))),
                     cmp_out::SInt{Kout}=SVec(ntuple(identity, Val(K))),
                     cmp_in::SInt{Kin}=SVec(ntuple(identity, Val(K))),
                     order_cmpfirst::Bool=true  # true to use Cartesian-component-major ordering for more tightly banded matrix
                     ) where {K,Kout,Kin}  # {space dimension, output field dimension, input field dimension}
    T = promote_type(eltype.(∆l⁻¹)..., eltype(e⁻ⁱᵏᴸ))  # eltype(eltype(∆l⁻¹)) can be Any if ∆l⁻¹ is inhomogeneous
    N = SVec(length.(∆l⁻¹))
    M = prod(N)

    num_blk = sum(abs, CURL_BLK[cmp_out,cmp_in])  # no allocations

    # Below, create_∂info() is called num_blk times in the double for loops, and each call
    # returns I, J, V of length 2M, so we preallocate Itot, Jtot, Vtot of length num_blk⋅2M.
    Itot = VecInt(undef, num_blk * 2M)
    Jtot = VecInt(undef, num_blk * 2M)
    Vtot = Vector{T}(undef, num_blk * 2M)

    ind_blk = 0  # index of matrix block
    for ind_nv = 1:Kout
        nv = cmp_out[ind_nv]  # Cartesian compotent of output field
        istr, ioff = order_cmpfirst ? (Kout, ind_nv-Kout) : (1, M*(ind_nv-1))  # (row stride, row offset)
        for ind_nu = 1:Kin
            nu = cmp_in[ind_nu]  # Cantesian component of input field

            parity = CURL_BLK[nv,nu]
            iszero(parity) && continue  # if zero block, skip to next nu

            jstr, joff = order_cmpfirst ? (Kin, ind_nu-Kin) : (1, M*(ind_nu-1))  # (column stride, column offset)

            nw = 6 - nv - nu  # direction of differentiation; 6 = 1 + 2 + 3
            is_nw = cmp_shp.==nw
            sum(is_nw)==1 || @error "cmp_shp = $cmp_shp does not have one and only one nw = $nw"
            ind_nw = findfirst(is_nw)
            I, J, V = create_∂info(ind_nw, isfwd[ind_nw], N, ∆l⁻¹[ind_nw], isbloch[ind_nw], e⁻ⁱᵏᴸ[ind_nw])

            @. I = istr * I + ioff
            @. J = jstr * J + joff
            V .*= parity

            # For some reason, using .= below is slower because it uses 1 allocatiotn.  On
            # the other hand, using = does not use allocation and therefore faster.
            indₛ, indₑ = ind_blk*2M + 1, (ind_blk+1)*2M
            Itot[indₛ:indₑ] = I
            Jtot[indₛ:indₑ] = J
            Vtot[indₛ:indₑ] = V
            ind_blk += 1
        end
    end

    return dropzeros!(sparse(Itot, Jtot, Vtot, Kout*M, Kin*M))
end
