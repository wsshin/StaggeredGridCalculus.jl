# Assumes the space dimension and field dimension are the same.  In other words, when the
# space coordinate indices are (i,j,k), then the field has three vector components.
# Therefore, for the input field array F[i,j,k,w], we assume w = 1:3.
export create_divg

# Wrapper to create the discrete divergence by default
#
# Note that when the positional arguments are SVector's, this function may call the concrete
# implementation directly instead of the next wrapper function taking ∆l⁻¹ even if the
# keyword arguments are not SVector's, because methods are dispatched based only on the
# types of positional arguments.  Therefore, we convert the keyword arguments into SVector's.
create_divg(isfwd::AbsVecBool,  # isfwd[w] = true|false: ∂w is forward|backward difference
            N::AbsVecInteger,  # size of grid
            ∆l⁻¹::Tuple{Vararg{Number}}=ntuple(x->1.0,length(N)),  # ∆l⁻¹[w]: inverse of uniform distance between grid planes in w-direction
            isbloch::AbsVecBool=fill(true,length(isfwd)),  # boundary conditions in x, y, z
            e⁻ⁱᵏᴸ::AbsVecNumber=ones(length(isfwd));  # Bloch phase factor in x, y, z
            order_cmpfirst::Bool=true) =  # true to use Cartesian-component-major ordering for more tightly banded matrix
    (K = length(isfwd); create_divg(isfwd, fill.(∆l⁻¹,(N...,)), isbloch, e⁻ⁱᵏᴸ; order_cmpfirst))

# Wrapper to convert AbstractVector's to SVec's
create_divg(isfwd::AbsVecBool,  # isfwd[w] = true|false: ∂w is forward|backward difference
            ∆l⁻¹::NTuple{K,AbsVecNumber},  # ∆l⁻¹[w]: inverse of distances between grid planes in w-direction
            isbloch::AbsVecBool=fill(true,K),  # boundary conditions in x, y, z
            e⁻ⁱᵏᴸ::AbsVecNumber=ones(K);  # Bloch phase factor in x, y, z
            order_cmpfirst::Bool=true  # true to use Cartesian-component-major ordering for more tightly banded matrix
            ) where {K} =
    # I should not cast e⁻ⁱᵏᴸ into a complex vector, because then the entire curl matrix
    # becomes a complex matrix.  Sometimes I want to keep it real (e.g., when no PML and
    # Bloch phase factors are used).  In fact, this is the reason why I accept e⁻ⁱᵏᴸ instead
    # of constructing it from k and L as exp.(-im .* k .* L), which is always complex even
    # if k = 0.
    #
    # I should not cast ∆l⁻¹ to a vector of any specific type (e.g., Float, ComplexF), either,
    # because sometimes I would want to even create an integral curl operator.
    create_divg(SBool{K}(isfwd), ∆l⁻¹, SBool{K}(isbloch), SVec{K}(e⁻ⁱᵏᴸ); order_cmpfirst)

function create_divg(isfwd::SBool{K},  # isfwd[w] = true|false: ∂w is forward|backward difference
                     ∆l⁻¹::NTuple{K,AbsVecNumber},  # ∆l⁻¹[w]: inverse of distances between grid planes in w-direction
                     isbloch::SBool{K},  # boundary conditions in K dimensions
                     e⁻ⁱᵏᴸ::SNumber{K};  # Bloch phase factors in K dimensions
                     order_cmpfirst::Bool=true  # true to use Cartesian-component-major ordering for more tightly banded matrix
                     ) where {K}
    T = promote_type(eltype.(∆l⁻¹)..., eltype(e⁻ⁱᵏᴸ))  # eltype(eltype(∆l⁻¹)) can be Any if ∆l⁻¹ is inhomogeneous
    N = SVec(length.(∆l⁻¹))
    M = prod(N)
    KM = K * M

    # Below, create_∂info() is called K times in the double for loops, and each call returns
    # I, J, V of length 2M, so we preallocate Itot, Jtot, Vtot of length K×2M = 2KM.
    Itot = VecInt(undef, 2KM)
    Jtot = VecInt(undef, 2KM)
    Vtot = Vector{T}(undef, 2KM)

    for nw = 1:K  # direction of differentiation
        I, J, V = create_∂info(nw, isfwd[nw], N, ∆l⁻¹[nw], isbloch[nw], e⁻ⁱᵏᴸ[nw])

        nu = nw  # component of input field to feed to ∂w (column index of matrix block)
        jstr, joff = order_cmpfirst ? (K, nu-K) : (1, M*(nu-1))  # (column stride, column offset)
        @. J = jstr * J + joff

        # For some reason, using .= below is slower because it uses 1 allocatiotn.  On the
        # other hand, using = does not use allocation and therefore faster.
        indₛ, indₑ = (nw-1)*2M + 1, nw*2M
        Itot[indₛ:indₑ] = I
        Jtot[indₛ:indₑ] = J
        Vtot[indₛ:indₑ] = V
    end

    return dropzeros!(sparse(Itot, Jtot, Vtot, M, KM))
end
