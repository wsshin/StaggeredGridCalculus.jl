export create_curl

create_curl(isfwd::AbsVecBool,  # isfwd[w] = true|false: create ∂w by forward|backward difference
            N::AbsVecInteger,  # size of grid
            ∆l::Tuple3{AbsVecNumber}=ones.((N...,)),  # ∆l[w]: distances between grid planes in x-direction
            isbloch::AbsVecBool=fill(true,length(N)),  # boundary conditions in x, y, z
            e⁻ⁱᵏᴸ::AbsVecNumber=ones(length(N));  # Bloch phase factor in x, y, z
            order_cmpfirst::Bool=true) =  # true to use Cartesian-component-major ordering for more tightly banded matrix
    # I should not cast e⁻ⁱᵏᴸ into a complex vector, because then the entire curl matrix
    # becomes a complex matrix.  Sometimes I want to keep it real (e.g., when no PML and
    # Bloch phase factors are used).  In fact, this is the reason why I accept e⁻ⁱᵏᴸ instead
    # of constructing it from k and L as exp.(-im .* k .* L), which is always complex even
    # if k = 0.
    #
    # I should not cast ∆l to a vector of any specific type (e.g., Float, CFloat), either,
    # because sometimes I would want to even create an integral curl operator.
    (K = length(N); create_curl(SVector{K}(isfwd), SVector{K,Int}(N), ∆l, SVector{K}(isbloch), SVector{K}(e⁻ⁱᵏᴸ), order_cmpfirst=order_cmpfirst))

function create_curl(isfwd::SBool{3},  # isfwd[w] = true|false: create ∂w by forward|backward difference
                     N::SInt{3},  # size of grid
                     ∆l::Tuple3{AbsVecNumber},  # ∆l[w]: distances between grid planes in x-direction
                     isbloch::SBool{3},  # boundary conditions in x, y, z
                     e⁻ⁱᵏᴸ::SNumber{3};  # Bloch phase factor in x, y, z
                     order_cmpfirst::Bool=true)  # true to use Cartesian-component-major ordering for more tightly banded matrix
    T = promote_type(eltype.(∆l)..., eltype(e⁻ⁱᵏᴸ))  # eltype(eltype(∆l)) can be Any if ∆l is inhomogeneous
    M = prod(N)

    # Below, create_∂info() is called 3×2 = 6 times in the double for loops, and each call
    # returns I, J, V of length 2M, so we preallocate Itot, Jtot, Vtot of length 6×2M = 12M.
    Itot = VecInt(undef, 12M)
    Jtot = VecInt(undef, 12M)
    Vtot = Vector{T}(undef, 12M)

    indblk = 0  # index of matrix block
    for nv = 1:3  # Cartesian compotent of output field
        istr, ioff = order_cmpfirst ? (3, nv-3) : (1, M*(nv-1))  # (row stride, row offset)
        parity = 1
        for nw = next2(nv)  # direction of differentiation
            nw′ = 6 - nv - nw  # Cantesian component of input field; 6 = nX + nY + nZ
            jstr, joff = order_cmpfirst ? (3, nw′-3) : (1, M*(nw′-1))  # (column stride, column offset)
            I, J, V = create_∂info(nw, isfwd[nw], N, ∆l[nw], isbloch[nw], e⁻ⁱᵏᴸ[nw])

            @. I = istr * I + ioff
            @. J = jstr * J + joff
            V .*= parity

            # For some reason, using .= below is slower because it uses 1 allocatiotn.  On the
            # other hand, using = does not use allocation and therefore faster.
            indₛ, indₑ = indblk*2M + 1, (indblk+1)*2M
            Itot[indₛ:indₑ] = I
            Jtot[indₛ:indₑ] = J
            Vtot[indₛ:indₑ] = V
            indblk += 1

            parity = -1
        end
    end

    return dropzeros!(sparse(Itot, Jtot, Vtot, 3M, 3M))
end
