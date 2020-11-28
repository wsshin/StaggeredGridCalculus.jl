export create_grad

create_grad(isfwd::AbsVecBool,  # isfwd[w] = true|false: create ∂w by forward|backward difference
            N::AbsVecInteger,  # size of grid
            ∆l⁻¹::Tuple{Vararg{AbsVecNumber}}=ones.((N...,)),  # ∆l⁻¹[w]: inverse of distances between grid planes in w-direction
            isbloch::AbsVecBool=fill(true,length(N)),  # boundary conditions in x, y, z
            e⁻ⁱᵏᴸ::AbsVecNumber=ones(length(N));  # Bloch phase factor in x, y, z
            permute∂::AbsVecInteger=1:length(N),  # permute∂[w]: location of ∂w block
            scale∂::AbsVecNumber=ones(length(N)),  # scale∂[w]: scale factor to multiply to ∂w
            order_cmpfirst::Bool=true) =  # true to use Cartesian-component-major ordering for more tightly banded matrix
    # I should not cast e⁻ⁱᵏᴸ into a complex vector, because then the entire curl matrix
    # becomes a complex matrix.  Sometimes I want to keep it real (e.g., when no PML and
    # Bloch phase factors are used).  In fact, this is the reason why I accept e⁻ⁱᵏᴸ instead
    # of constructing it from k and L as exp.(-im .* k .* L), which is always complex even
    # if k = 0.
    #
    # I should not cast ∆l⁻¹ to a vector of any specific type (e.g., Float, CFloat), either,
    # because sometimes I would want to even create an integral curl operator.
    (K = length(N); create_grad(SVector{K}(isfwd), SVector{K,Int}(N), ∆l⁻¹, SVector{K}(isbloch), SVector{K}(e⁻ⁱᵏᴸ),
                                permute∂=SVector{K}(permute∂), scale∂=SVector{K}(scale∂), order_cmpfirst=order_cmpfirst))

function create_grad(isfwd::SBool{K},  # isfwd[w] = true|false: create ∂w by forward|backward difference
                     N::SInt{K},  # size of grid
                     ∆l⁻¹::NTuple{K,AbsVecNumber},  # ∆l⁻¹[w]: inverse of distances between grid planes in w-direction
                     isbloch::SBool{K},  # boundary conditions in K dimensions
                     e⁻ⁱᵏᴸ::SNumber{K};  # Bloch phase factors in K dimensions
                     permute∂::SInt{K}=SVector(ntuple(identity, Val(K))),  # permute∂[w]: location of ∂w block
                     scale∂::SNumber{K}=SVector(ntuple(k->1.0, Val(K))),  # scale∂[w]: scale factor to multiply to ∂w
                     order_cmpfirst::Bool=true  # true to use Cartesian-component-major ordering for more tightly banded matrix
                     ) where {K}
    T = promote_type(eltype.(∆l⁻¹)..., eltype(e⁻ⁱᵏᴸ))  # eltype(eltype(∆l⁻¹)) can be Any if ∆l⁻¹ is inhomogeneous
    M = prod(N)
    KM = K * M

    # Below, create_∂info() is called K times in the double for loops, and each call returns
    # I, J, V of length 2M, so we preallocate Itot, Jtot, Vtot of length K×2M = 2KM.
    Itot = VecInt(undef, 2KM)
    Jtot = VecInt(undef, 2KM)
    Vtot = Vector{T}(undef, 2KM)

    for nw = 1:K  # direction of differentiation
        I, J, V = create_∂info(nw, isfwd[nw], N, ∆l⁻¹[nw], isbloch[nw], e⁻ⁱᵏᴸ[nw])

        nv = permute∂[nw]  # index of output field (index of matrix block)
        istr, ioff = order_cmpfirst ? (K, nv-K) : (1, M*(nv-1))  # (row stride, row offset)
        @. I = istr * I + ioff

        V .*= scale∂[nw]

        # For some reason, using .= below is slower because it uses 1 allocatiotn.  On the
        # other hand, using = does not use allocation and therefore faster.
        indₛ, indₑ = (nw-1)*2M + 1, nw*2M
        Itot[indₛ:indₑ] = I
        Jtot[indₛ:indₑ] = J
        Vtot[indₛ:indₑ] = V
    end

    return dropzeros!(sparse(Itot, Jtot, Vtot, KM, M))
end
