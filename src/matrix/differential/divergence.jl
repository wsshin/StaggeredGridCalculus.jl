export create_divg

create_divg(isfwd::AbsVecBool,  # isfwd[w] = true|false: create ∂w by forward|backward difference
            N::AbsVecInteger,  # size of grid
            ∆l::Tuple{Vararg{AbsVecNumber}}=ones.((N...,)),  # ∆l[w]: distances between grid planes in x-direction
            isbloch::AbsVecBool=fill(true,length(N)),  # boundary conditions in x, y, z
            e⁻ⁱᵏᴸ::AbsVecNumber=ones(length(N));  # Bloch phase factor in x, y, z
            permute∂::AbsVecInteger=1:length(N),  # permuted order of partial derivatives
            scale∂::AbsVecNumber=ones(length(N)), # scale factors to multiply to partial derivatives (after permuted)
            order_compfirst::Bool=true) =  # true to use Cartesian-component-major ordering for more tightly banded matrix
    # I should not cast e⁻ⁱᵏᴸ into a complex vector, because then the entire curl matrix
    # becomes a complex matrix.  Sometimes I want to keep it real (e.g., when no PML and
    # Bloch phase factors are used).  In fact, this is the reason why I accept e⁻ⁱᵏᴸ instead
    # of constructing it from k and L as exp.(-im .* k .* L), which is always complex even
    # if k = 0.
    #
    # I should not cast ∆l to a vector of any specific type (e.g., Float, CFloat), either,
    # because sometimes I would want to even create an integral curl operator.
    (K = length(N); create_divg(SVector{K}(isfwd), SVector{K,Int}(N), ∆l, SVector{K}(isbloch), SVector{K}(e⁻ⁱᵏᴸ),
                                permute∂=SVector{K}(permute∂), scale∂=SVector{K}(scale∂), order_compfirst=order_compfirst))

function create_divg(isfwd::SBool{K},  # isfwd[w] = true|false: create ∂w by forward|backward difference
                     N::SInt{K},  # size of grid
                     ∆l::NTuple{K,AbsVecNumber},  # ∆l[w]: distances between grid planes in x-direction
                     isbloch::SBool{K},  # boundary conditions in K dimensions
                     e⁻ⁱᵏᴸ::SNumber{K};  # Bloch phase factors in K dimensions
                     permute∂::SInt{K}=SVector(ntuple(identity, Val(K))),  # permuted order of partial derivatives
                     scale∂::SNumber{K}=SVector(ntuple(k->1, Val(K))), # scale factors to multiply to partial derivatives (after permuted)
                     order_compfirst::Bool=true  # true to use Cartesian-component-major ordering for more tightly banded matrix
                     ) where {K}
    T = promote_type(eltype.(∆l)..., eltype(e⁻ⁱᵏᴸ))  # eltype(eltype(∆l)) can be Any if ∆l is inhomogeneous
    M = prod(N)
    KM = K * M

    # Below, create_∂info() is called K times in the double for loops, and each call returns
    # I, J, V of length 2M, so we preallocate Itot, Jtot, Vtot of length K×2M = 2KM.
    Itot = VecInt(undef, 2KM)
    Jtot = VecInt(undef, 2KM)
    Vtot = Vector{T}(undef, 2KM)

    for nblk = 1:K  # index of matrix block
        nw = permute∂[nblk]  # Cartesian compotent of input field
        I, J, V = create_∂info(nw, isfwd[nw], N, ∆l[nw], isbloch[nw], e⁻ⁱᵏᴸ[nw])

        jstr, joff = order_compfirst ? (3, nblk-3) : (1, M*(nblk-1))  # (column stride, column offset)
        @. J = jstr * J + joff
        V .*= scale∂[nblk]

        # For some reason, using .= below is slower because it uses 1 allocatiotn.  On the
        # other hand, using = does not use allocation and therefore faster.
        indₛ, indₑ = (nblk-1)*2M + 1, nblk*2M
        Itot[indₛ:indₑ] = I
        Jtot[indₛ:indₑ] = J
        Vtot[indₛ:indₑ] = V
    end

    return dropzeros!(sparse(Itot, Jtot, Vtot, M, KM))
end
