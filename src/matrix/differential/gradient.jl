# Assumes the space dimension and field dimension are the same.  In other words, when the
# space coordinate indices are (i,j,k), then the field has three vector components.
# Therefore, for the output field array G[i,j,k,w], we assume w = 1:3.
export create_grad

# Wrapper to create the discrete gradient by default
#
# Note that when the positional arguments are SVector's, this function may call the concrete
# implementation directly instead of the next wrapper function taking ∆l⁻¹ even if the
# keyword arguments are not SVector's, because methods are dispatched based only on the
# types of positional arguments.  Therefore, we convert the keyword arguments into SVector's.
create_grad(isfwd::AbsVecBool,  # isfwd[w] = true|false: ∂w is forward|backward difference
            N::AbsVecInteger,  # size of grid
            ∆l⁻¹::Tuple{Vararg{Number}}=ntuple(x->1.0,length(isfwd)),  # ∆l⁻¹[w]: inverse of uniform distance between grid planes in w-direction
            isbloch::AbsVecBool=fill(true,length(isfwd)),  # boundary conditions in x, y, z
            e⁻ⁱᵏᴸ::AbsVecNumber=ones(Int,length(isfwd));  # Bloch phase factor in x, y, z
            dims::AbsVecInteger=1:length(isfwd),  # dimensions into which gradient is performed; size of matrix does not depend on this
            cmpfirst::Bool=true,  # true to use Cartesian-component-major ordering for more tightly banded matrix
            compact::Bool=true,  # true to create compact, tall matrix; false to create square matrix
            ) =
    (K = length(isfwd); create_grad(isfwd, fill.(∆l⁻¹,(N...,)), isbloch, e⁻ⁱᵏᴸ; dims, cmpfirst, compact))

# Wrapper to convert AbstractVector's to SVec's
create_grad(isfwd::AbsVecBool,  # isfwd[w] = true|false: ∂w is forward|backward difference
            ∆l⁻¹::NTuple{K,AbsVecNumber},  # ∆l⁻¹[w]: inverse of distances between grid planes in w-direction
            isbloch::AbsVecBool=fill(true,K),  # boundary conditions in x, y, z
            e⁻ⁱᵏᴸ::AbsVecNumber=ones(Int,K);  # Bloch phase factor in x, y, z
            dims::AbsVecInteger=1:K,  # dimensions into which gradient is performed; size of matrix does not depend on this
            cmpfirst::Bool=true,  # true to use Cartesian-component-major ordering for more tightly banded matrix
            compact::Bool=true,  # true to create compact, tall matrix; false to create square matrix
            ) where {K} =
    # I should not cast e⁻ⁱᵏᴸ into a complex vector, because then the entire curl matrix
    # becomes a complex matrix.  Sometimes I want to keep it real (e.g., when no PML and
    # Bloch phase factors are used).  In fact, this is the reason why I accept e⁻ⁱᵏᴸ instead
    # of constructing it from k and L as exp.(-im .* k .* L), which is always complex even
    # if k = 0.
    #
    # I should not cast ∆l⁻¹ to a vector of any specific type (e.g., Float, ComplexF), either,
    # because sometimes I would want to even create an integral curl operator.
    create_grad(SBool{K}(isfwd), ∆l⁻¹, SBool{K}(isbloch), SVec{K}(e⁻ⁱᵏᴸ); dims=SInt{length(dims)}(dims), cmpfirst, compact)

# - Unlike create_curl(), create_grad() does not require cmp_out, because it is the same as
# dir_diff (now removed, but deduced from ∆l⁻¹).
#
# - For compact = false, we create a full matrix (rather than a block-column matrix) that
# takes the vector field, such as (Hx,Hy,Hz), and takes the gradient of its one specific
# Cartesian component.  Technically, the input field component could be none of the
# directions of differentiation.  For example, consider taking the gradient of Hz in the
# xy-plane.  Then the full matrix to create is a 2×3 block matrix.  We don't consider such
# cases for simplicity, because we should be able to pass the information of "3" in that
# case, which cannot be contained in the scalar cmp_in.  This is usually OK, because even in
# the considered case that occurs in the waveguide mode calculation, the matrix equation is
# set up for (Hx, Hy), and Hz is derived from the two.
#
# - In the future, if I really need to generate 2×3 block matrices, consider updating the
# interface to pass cmp_in = [false, false, true].  More generally, I should be able to
# specify the dimension Kin of the input, dimension Kout of the output, cmp_in (single
# natural number ≤ Kin), and cmp_out (Kout-vector).  When compact=false, this will generate
# an operator that takes a Kin-dimensional input vector field and generates Kout-dimensional
# output vector field.
#
# - For now, I think it should be fine to assume that Kin = Kout = K of ∆l⁻¹.  The gradient
# is almost always paired with the divergence.  The dimension (e.g., 2D) of the output
# vector field of the gradient is the dimension of the input vector field of the divergence,
# which is usually the same as the dimension of any vector field of the problem (e.g., 2D).
# example case).
#
# - To check the correctness of the implementation, consider the case of dir_diff = [2,3].
function create_grad(isfwd::SBool{K},  # isfwd[w] = true|false: ∂w is forward|backward difference
                     ∆l⁻¹::NTuple{K,AbsVecNumber},  # ∆l⁻¹[w]: inverse of distances between grid planes in w-direction
                     isbloch::SBool{K},  # boundary conditions in K dimensions
                     e⁻ⁱᵏᴸ::SNumber{K};  # Bloch phase factors in K dimensions
                     cmp_in::Integer,  # single Cartesian direction (not directions) of input field; no effect if compact = true
                     cmpfirst::Bool=true,  # true to use Cartesian-component-major ordering for more tightly banded matrix
                     compact::Bool=true,  # true to create compact, tall matrix; false to create square matrix
                     ) where {K}
    T = promote_type(eltype.(∆l⁻¹)..., eltype(e⁻ⁱᵏᴸ))  # eltype(eltype(∆l⁻¹)) can be Any if ∆l⁻¹ is inhomogeneous
    N = SVec(length.(∆l⁻¹))
    M = prod(N)
    KM = K * M  # resulting matrix: KM×M for compact=true; KM×KM for compact=false

    # Below, create_∂info() is called K times in the for loop, and each call returns I, J, V
    # of length 2M (one diagonal and one off-diagonal), so we preallocate Itot, Jtot, Vtot
    # of length K×2M = 2KM.
    Itot = VecInt(undef, 2KM)
    Jtot = VecInt(undef, 2KM)
    Vtot = Vector{T}(undef, 2KM)

    for nw = 1:K  # direction of differentiation (= direction of output)
        istr, ioff = cmpfirst ? (K, nw-K) : (1, (nw-1)M)  # (row stride, row offset)

        I, J, V = create_∂info(nw, isfwd[nw], N, ∆l⁻¹[nw], isbloch[nw], e⁻ⁱᵏᴸ[nw])

        @. I = istr * I + ioff

        if !compact
            # Note that cmp_in is used only when compact=false.
            jstr, joff = cmpfirst ? (K, cmp_in-K) : (1, (cmp_in-1)M)  # (column stride, column offset)
            @. J = jstr * J + joff
        end

        # For some reason, using .= below is slower because it uses 1 allocatiotn.  On the
        # other hand, using = does not use allocation and therefore faster.
        indₛ, indₑ = (ind_nw-1)*2M + 1, ind_nw*2M
        Itot[indₛ:indₑ] = I
        Jtot[indₛ:indₑ] = J
        Vtot[indₛ:indₑ] = V
    end

    return compact ?
        dropzeros!(sparse(Itot, Jtot, Vtot, KM, M)) :
        dropzeros!(sparse(Itot, Jtot, Vtot, KM, KM))
end
