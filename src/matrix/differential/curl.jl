export create_curl

# Wrapper to create the discrete curl by default
create_curl(isfwd::AbsVecBool,  # isfwd[w] = true|false: ∂w is forward|backward difference
            N::AbsVecInteger,  # size of grid
            ∆l⁻¹::NTuple{K,Number}=ntuple(k->1.0,length(isfwd)),  # ∆l⁻¹[w]: inverse of uniform distance between grid planes in x-direction
            isbloch::AbsVecBool=fill(true,K),  # boundary conditions in x, y, z
            e⁻ⁱᵏᴸ::AbsVecNumber=ones(Int,K);  # Bloch phase factor in x, y, z
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
    create_curl(isfwd, fill.(∆l⁻¹,(N...,)), isbloch, e⁻ⁱᵏᴸ; cmp_out, cmp_in, cmpfirst)

# Wrapper to convert AbstractVector's to SVec's
create_curl(isfwd::AbsVecBool,  # isfwd[w] = true|false: ∂w is forward|backward difference
            ∆l⁻¹::NTuple{K,AbsVecNumber},  # ∆l⁻¹[w]: inverse of uniform distance between grid planes in x-direction
            isbloch::AbsVecBool=fill(true,K),  # boundary conditions in x, y, z
            e⁻ⁱᵏᴸ::AbsVecNumber=ones(Int,K);  # Bloch phase factor in x, y, z
            cmp_out::AbsVecInteger=1:K,
            cmp_in::AbsVecInteger=1:K,
            cmpfirst::Bool=true  # true to use Cartesian-component-major ordering for more tightly banded matrix
            ) where {K} =
    (Kout = length(cmp_out); Kin = length(cmp_in);
     create_curl(SVec{K}(isfwd), ∆l⁻¹, SVec{K}(isbloch), SVec{K}(e⁻ⁱᵏᴸ);
                 cmp_out=SInt{Kout}(cmp_out), cmp_in=SInt{Kin}(cmp_in), cmpfirst))

# - The directions of differentiations are 1:K, where K is the number of entries of ∆l⁻¹.
#
# - Provide cmp_in to specify the input vector field components; can contain directions > K.
#
# - Provide cmp_out to specify the output vector field componets; can contain directions > K.
#
# - For a given element of cmp_in and given element of cmp_out, the direction of
# differentiation is decided.  For example, the 3rd-component of the curl is ∂₁F₂ - ∂₂F₁.
# Therefore, for cmp_out[i] = 3 and cmp_in[j] = 1, the direction of differentiation is 2.
#
# - We deal with 3D curl.  Even when we deal with 1D or 2D, we are taking the components of
# 3D curl rather than taking generalized curl; see Agenda > StaggeredGridCalculus >
# [Dec/09/2023] Update.
#
# - This is why create_curl() does not support compact: we need to specify cmp_in and and
# cmp_out of the 3D curl explicitly, and the size of the curl matrix is automatically
# decided by them.  On the other hand, create_grad(), for example,
#
function create_curl(
    isfwd::SBool{K},  # isfwd[w] = true|false: ∂w is forward|backward difference
    ∆l⁻¹::NTuple{K,AbsVecNumber},  # ∆l⁻¹[w]: inverse of distances between grid planes in w-direction; assume ordered by right-hand rule
    isbloch::SBool{K},  # isblock[w]: boundary conditions in w-direction
    e⁻ⁱᵏᴸ::SNumber{K};  # e⁻ⁱᵏᴸ[w]: Bloch phase factor in w-direction
    cmp_out::SInt{Kout}=SVec(ntuple(identity, Val(K))),  # Cartesian directions of output field components; not necessarily 1:K, because, e.g., Ez can be defined on xy-shape
    cmp_in::SInt{Kin}=SVec(ntuple(identity, Val(K))),  # Cartesian directions of input field; not necessarily 1:K, because, e.g., Ez can be defined on xy-shape
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
    for (ind_nv, nv) = enumerate(cmp_out)  # nv is direction of output field; ind_nv is used for calculating row indices in sparse matrix
        # Why should we use ind_nv instead of nv?  Consider cmpfirst=false.  Even if nv = 3,
        # it could be the only (and the first) output direction.  Then it should be the
        # first row block, corresponding to ind_nv = 1.
        istr, ioff = cmpfirst ? (Kout, ind_nv-Kout) : (1, (ind_nv-1)M)  # (row stride, row offset)

        for (ind_nu, nu) = enumerate(cmp_in)  # nu is direction of input field; ind_nu is used for calculating column indices in sparse matrix
            # ∇×F = 1̂ (∂₂F₃ - ∂₃F₂) + 2̂ (∂₃F₁ - ∂₁F₃) + 3̂ (∂₁F₂ - ∂₂F₁)
            #
            # In matrix form,
            # ⎡ 0  -∂₃  ∂₂⎤
            # ⎢ ∂₃  0  -∂₁⎥
            # ⎣-∂₂  ∂₁  0 ⎦
            nv==nu && continue  # if zero block, skip to next nu
            parity = mod(nv-nu+1, 3) - 1  # sign of (nv,nu)-entry of above matrix

            jstr, joff = cmpfirst ? (Kin, ind_nu-Kin) : (1, (ind_nu-1)M)  # (column stride, column offset)

            # nw is the index of vector N.  N is needed by create_∂info().  If K = 2,
            # N = [Nx,Ny] and nw = 1 for x and nw = 2 for y.
            nw = 6 - nv - nu  # direction of differentiation; 6 = 1 + 2 + 3
            @argcheck 1≤nw≤K  # direction of differentiation should be one of 1:K

            I, J, V = create_∂info(nw, isfwd[nw], N, ∆l⁻¹[nw], isbloch[nw], e⁻ⁱᵏᴸ[nw])

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
