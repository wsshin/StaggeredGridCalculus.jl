export create_grad

create_grad(isfwd::AbsVecBool,  # isfwd[v] = true|false: create ∂v by forward|backward difference
            N::AbsVecInteger,  # size of grid
            ∆l::Tuple{Vararg{AbsVecNumber}}=ones.((N...,)),  # ∆l[v]: distances between grid planes in v-direction
            isbloch::AbsVecBool=fill(true,length(N)),  # boundary conditions in x, y, z
            e⁻ⁱᵏᴸ::AbsVecNumber=ones(length(N));  # Bloch phase factor in x, y, z
            permute∂::AbsVecInteger=1:length(N),  # permute∂[v]: location of ∂v block
            scale∂::AbsVecNumber=ones(length(N)),  # scale∂[v]: scale factor to multiply to ∂v
            order_cmpfirst::Bool=true) =  # true to use Cartesian-component-major ordering for more tightly banded matrix
    # I should not cast e⁻ⁱᵏᴸ into a complex vector, because then the entire curl matrix
    # becomes a complex matrix.  Sometimes I want to keep it real (e.g., when no PML and
    # Bloch phase factors are used).  In fact, this is the reason why I accept e⁻ⁱᵏᴸ instead
    # of constructing it from k and L as exp.(-im .* k .* L), which is always complex even
    # if k = 0.
    #
    # I should not cast ∆l to a vector of any specific type (e.g., Float, CFloat), either,
    # because sometimes I would want to even create an integral curl operator.
    (K = length(N); create_grad(SVector{K}(isfwd), SVector{K,Int}(N), ∆l, SVector{K}(isbloch), SVector{K}(e⁻ⁱᵏᴸ),
                                permute∂=SVector{K}(permute∂), scale∂=SVector{K}(scale∂), order_cmpfirst=order_cmpfirst))

function create_grad(isfwd::SBool{K},  # isfwd[v] = true|false: create ∂v by forward|backward difference
                     N::SInt{K},  # size of grid
                     ∆l::NTuple{K,AbsVecNumber},  # ∆l[v]: distances between grid planes in v-direction
                     isbloch::SBool{K},  # boundary conditions in K dimensions
                     e⁻ⁱᵏᴸ::SNumber{K};  # Bloch phase factors in K dimensions
                     permute∂::SInt{K}=SVector(ntuple(identity, Val(K))),  # permute∂[v]: location of ∂v block
                     scale∂::SNumber{K}=SVector(ntuple(k->1, Val(K))),  # scale∂[w]: scale factor to multiply to ∂v
                     order_cmpfirst::Bool=true  # true to use Cartesian-component-major ordering for more tightly banded matrix
                     ) where {K}
    T = promote_type(eltype.(∆l)..., eltype(e⁻ⁱᵏᴸ))  # eltype(eltype(∆l)) can be Any if ∆l is inhomogeneous
    M = prod(N)
    KM = K * M

    # Below, create_∂info() is called K times in the double for loops, and each call returns
    # I, J, V of length 2M, so we preallocate Itot, Jtot, Vtot of length K×2M = 2KM.
    Itot = VecInt(undef, 2KM)
    Jtot = VecInt(undef, 2KM)
    Vtot = Vector{T}(undef, 2KM)

    for nv = 1:K  # direction of differentiation
        I, J, V = create_∂info(nv, isfwd[nv], N, ∆l[nv], isbloch[nv], e⁻ⁱᵏᴸ[nv])

        nblk = permute∂[nv]  # index of matrix block
        istr, ioff = order_cmpfirst ? (K, nblk-K) : (1, M*(nblk-1))  # (row stride, row offset)
        @. I = istr * I + ioff

        V .*= scale∂[nv]

        # For some reason, using .= below is slower because it uses 1 allocatiotn.  On the
        # other hand, using = does not use allocation and therefore faster.
        indₛ, indₑ = (nv-1)*2M + 1, nv*2M
        Itot[indₛ:indₑ] = I
        Jtot[indₛ:indₑ] = J
        Vtot[indₛ:indₑ] = V
    end

    return dropzeros!(sparse(Itot, Jtot, Vtot, KM, M))
end
