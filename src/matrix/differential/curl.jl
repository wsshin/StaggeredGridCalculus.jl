export create_curl

# Wrapper to create the discrete curl by default
create_curl(isfwd::AbsVecBool,  # isfwd[w] = true|false: ∂w is forward|backward difference
            N::AbsVecInteger,  # size of grid
            ∆l⁻¹::NTuple{K,Number}=ntuple(k->1.0,length(isfwd)),  # ∆l⁻¹[w]: inverse of uniform distance between grid planes in x-direction
            isbloch::AbsVecBool=fill(true,K),  # boundary conditions in x, y, z
            e⁻ⁱᵏᴸ::AbsVecNumber=ones(Int,K);  # Bloch phase factor in x, y, z
            dir_diff::AbsVecInteger=1:K,
            cmp_out::AbsVecInteger=1:K,
            cmp_in::AbsVecInteger=1:K,
            cmpfirst::Bool=true  # true to use Cartesian-component-major ordering for more tightly banded matrix
            ) where {K} =
    # I should not cast e⁻ⁱᵏᴸ into a complex vector, because then the entire curl matrix
    # becomes a complex matrix.  Sometimes I want to keep it real (e.g., when no PML and
    # Bloch phase factors are used).  In fact, this is the reason why I accept e⁻ⁱᵏᴸ instead
    # of constructing it from k and L as exp.(-im .* k .* L), which is always complex even
    # if k = 0.
    #
    # I should not cast ∆l⁻¹ to a vector of any specific type (e.g., Float, ComplexF), either,
    # because sometimes I would want to even create an integral curl operator.
    create_curl(isfwd, fill.(∆l⁻¹,(N...,)), isbloch, e⁻ⁱᵏᴸ; dir_diff, cmp_out, cmp_in, cmpfirst)

# Wrapper to convert AbstractVector's to SVec's
create_curl(isfwd::AbsVecBool,  # isfwd[w] = true|false: ∂w is forward|backward difference
            ∆l⁻¹::NTuple{K,AbsVecNumber},  # ∆l⁻¹[w]: inverse of uniform distance between grid planes in x-direction
            isbloch::AbsVecBool=fill(true,K),  # boundary conditions in x, y, z
            e⁻ⁱᵏᴸ::AbsVecNumber=ones(Int,K);  # Bloch phase factor in x, y, z
            dir_diff::AbsVecInteger=1:K,
            cmp_out::AbsVecInteger=1:K,
            cmp_in::AbsVecInteger=1:K,
            cmpfirst::Bool=true  # true to use Cartesian-component-major ordering for more tightly banded matrix
            ) where {K} =
    (Kout = length(cmp_out); Kin = length(cmp_in);
     create_curl(SVec{K}(isfwd), ∆l⁻¹, SVec{K}(isbloch), SVec{K}(e⁻ⁱᵏᴸ);
                 dir_diff=SInt{K}(dir_diff), cmp_out=SInt{Kout}(cmp_out), cmp_in=SInt{Kin}(cmp_in), cmpfirst))

# ∇×F = x̂ (∂y Fz - ∂z Fy) + ŷ (∂z Fx - ∂x Fz) + ẑ (∂x Fy - ∂y Fx)
#
# In matrix form,
# ⎡ 0  -∂z  ∂y⎤
# ⎢ ∂z  0  -∂x⎥
# ⎣-∂y  ∂x  0 ⎦
#
# CURL_BLK takes only the signs of the blocks:
const CURL_BLK = S²Int{3,9}(0,  1, -1,  # 1st column (not row)
                           -1,  0,  1,  # 2nd column (not row)
                            1, -1,  0)  # 3rd column (not row)

# - Provide cmp_in to specify the input vector field components.
# - Provide cmp_out to specify the output vector field componets.
# - For a given element of cmp_in and given element of cmp_out, the direction of
#  differentiation is decided.  For example, the z-component of the curl is ∂xFy - ∂yFx.
#  Therefore, for cmp_out[i] = z and cmp_in[j] = x, the direction of differentiation is y.
#  In doing so, use CURL_BLK.  Using CURL_BLK is necessary, because we are dealing with 3D
#  curl.  Even if we deal with 1D or 2D, we are taking the components of 3D curl rather than
#  taking generalized curl; see Agenda > StaggeredGridCalculus > [Dec/09/2023] Update.
# - Provide dir_diff in order to figure out which component of ∆l is ∆y.  Even though we
#  deal with 3D curl, we may pass ∆ of only 1D or 2D (more on this below), so we need to
#  specify which components of ∆l correspond to x, y, z.

# In implementing create_curl(), there are two options for choosing ∆l⁻¹ and dir_diff:
# - Pass the full ∆l⁻¹ for all directions, even if some of ∆x⁻¹, ∆y⁻¹, ∆z⁻¹ are not used,
# and indicate the used ones by dir_diff.  In this case, the number K of the Cartesian
# directions of ∆l⁻¹ may not be necessarily the K of dir_diff.
# - Pass ∆l⁻¹ containing only the used ones of ∆x⁻¹, ∆y⁻¹, ∆z⁻¹; even in this case, we still
# need dir_diff, because when ∆l⁻¹ has, e.g., two Cartesian components, we need to indicate
# whether they are (x,y), (y,z), or (z,x).  In this case, ∆l⁻¹ and dir_diff  have the same K.
#
# We choose the second option, because in MaxwellGuidedMode, when the cross section is, e.g.,
# on the yz-plane, we don't even have ∆x to create the full ∆l⁻¹.  Therefore, we use the
# same K for ∆l⁻¹ and dir_diff.
#
# Also, this means that the order of components is not necessarily xyz.  If we deal with the
# yz-plane, the two components are y and z, so the first component is not necessarily the x-
# component.
#
# If ∆l⁻¹ = ((∆yprim⁻¹, ∆zprim⁻¹), (∆ydual⁻¹, ∆zdual⁻¹)), we pass dir_diff = [2,3].  This
# is important for getting ind_nw from the direction of differentiation nw = 6 - nv - nu...
# Actually not sure: when nu and nv are decided, we can figure out dir_diff in CURL_BLK.
function create_curl(
    isfwd::SBool{K},  # isfwd[w] = true|false: ∂w is forward|backward difference
    ∆l⁻¹::NTuple{K,AbsVecNumber},  # ∆l⁻¹[w]: inverse of distances between grid planes in w-direction
    isbloch::SBool{K},  # isblock[w]: boundary conditions in w-direction
    e⁻ⁱᵏᴸ::SNumber{K};  # e⁻ⁱᵏᴸ[w]: Bloch phase factor in w-direction
    dir_diff::SInt{K}=SVec(ntuple(identity, Val(K))),  # Cartesian directions of differentiation (axes of space containing shapes)
    cmp_out::SInt{Kout}=SVec(ntuple(identity, Val(K))),  # Cartesian directions of output field components; not necessarily dir_diff, because, e.g., Ez can be defined on xy-shape
    cmp_in::SInt{Kin}=SVec(ntuple(identity, Val(K))),  # Cartesian directions of input field; not necessarily dir_diff, because, e.g., Ez can be defined on xy-shape
    cmpfirst::Bool=true,  # true to use Cartesian-component-major ordering for more tightly banded matrix
    # unlike create_divg() and create_grad(), create_curl() does not have kwarg `compact`; see Agenda > StaggeredGridCalculus > Dec/09/2023] Update
) where {K,Kout,Kin}  # {space dimension, output field dimension, input field dimension}
    T = promote_type(eltype.(∆l⁻¹)..., eltype(e⁻ⁱᵏᴸ))  # eltype(eltype(∆l⁻¹)) can be Any if ∆l⁻¹ is inhomogeneous
    N = SVec(length.(∆l⁻¹))
    M = prod(N)

    num_blk = sum(abs, CURL_BLK[cmp_out,cmp_in])  # no allocations

    # Below, create_∂info() is called num_blk times in the double for loops, and each call
    # returns I, J, V of length 2M (one diagonal and one off-diagonal), so we preallocate
    # Itot, Jtot, Vtot of length num_blk⋅2M.
    Itot = VecInt(undef, num_blk * 2M)
    Jtot = VecInt(undef, num_blk * 2M)
    Vtot = Vector{T}(undef, num_blk * 2M)

    ind_blk = 0  # index of matrix block
    for (ind_nv, nv) = enumerate(cmp_out)  # enumerate Kout-directions of output field
        istr, ioff = cmpfirst ? (Kout, ind_nv-Kout) : (1, (ind_nv-1)M)  # (row stride, row offset)

        for (ind_nu, nu) = enumerate(cmp_in)  # enumerate Kin-directions input field
            parity = CURL_BLK[nv,nu]
            iszero(parity) && continue  # if zero block, skip to next nu

            jstr, joff = cmpfirst ? (Kin, ind_nu-Kin) : (1, (ind_nu-1)M)  # (column stride, column offset)

            nw = 6 - nv - nu  # direction of differentiation; 6 = 1 + 2 + 3
            is_nw = dir_diff.==nw
            sum(is_nw)==1 || @error "dir_diff = $dir_diff does not have one and only one nw = $nw"
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
