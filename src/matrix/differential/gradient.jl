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
create_grad(isfwd::AbstractVector{Bool},  # isfwd[w] = true|false: ∂w is forward|backward difference
            N::AbstractVector{<:Integer},  # size of grid
            ∆l⁻¹::NTuple{K,Number}=ntuple(x->1,length(isfwd)),  # ∆l⁻¹[w]: inverse of uniform distance between grid planes in w-direction
            isbloch::AbstractVector{Bool}=ones(Bool,K),  # boundary conditions in x, y, z
            e⁻ⁱᵏᴸ::AbstractVector{<:Number}=ones(Int,K);  # Bloch phase factor in x, y, z
            kwargs...
            ) where {K} =
    create_grad(isfwd, fill.(∆l⁻¹,(N...,)), isbloch, e⁻ⁱᵏᴸ; kwargs...)

# Wrapper to convert AbstractVector's to SVector's
create_grad(isfwd::AbstractVector{Bool},  # isfwd[w] = true|false: ∂w is forward|backward difference
            ∆l⁻¹::NTuple{K,AbstractVector{<:Number}},  # ∆l⁻¹[w]: inverse of distances between grid planes in w-direction
            isbloch::AbstractVector{Bool}=ones(Bool,K),  # boundary conditions in x, y, z
            e⁻ⁱᵏᴸ::AbstractVector{<:Number}=ones(Int,K);  # Bloch phase factor in x, y, z
            kwargs...
            ) where {K} =
    # I should not cast e⁻ⁱᵏᴸ into a complex vector, because then the entire curl matrix
    # becomes a complex matrix.  Sometimes I want to keep it real (e.g., when no PML and
    # Bloch phase factors are used).  In fact, this is the reason why I accept e⁻ⁱᵏᴸ instead
    # of constructing it from k and L as exp.(-im .* k .* L), which is always complex even
    # if k = 0.
    #
    # I should not cast ∆l⁻¹ to a vector of any specific type (e.g., Float64, ComplexF64), either,
    # because sometimes I would want to even create an integral curl operator.
    create_grad(SVector{K}(isfwd), ∆l⁻¹, SVector{K}(isbloch), SVector{K}(e⁻ⁱᵏᴸ); kwargs...)

# - Unlike create_curl(), create_grad() does not require cmp_out, because it is the same as
# dir_diff (now removed, but deduced from ∆l⁻¹).
#
# - For compact = false (now implemented by cmp_in > 0), we create a full matrix (rather
# than a block-column matrix) that takes the vector field, such as (Hx,Hy,Hz), and takes the
# gradient of its one specific Cartesian component.  Technically, the input field component
# could be none of the directions of differentiation.  For example, consider taking the
# gradient of Hz in the xy-plane.  Then the full matrix to create is a 2×3 block matrix.  We
# don't consider such cases for simplicity, because we should be able to pass the
# information of "3" in that case, which cannot be contained in the scalar cmp_in.  This is
# usually OK, because even in the considered case that occurs in the waveguide mode
# calculation, the matrix equation is set up for (Hx, Hy), and Hz is derived from the two.
#
# - In the future, if I really need to generate 2×3 block matrices, consider updating the
# interface to pass cmp_in = [false, false, true].  More generally, I should be able to
# specify the dimension Kin of the input, dimension Kout of the output, cmp_in (single
# natural number ≤ Kin), and cmp_out (Kout-vector).  When compact=false, this will generate
# an operator that takes a Kin-dimensional input vector field and generates Kout-dimensional
# output vector field.
#
# - For now, I think it should be fine to assume that Kin = Kout = K of ∆l⁻¹ as mentioned
# above.  The gradient is almost always paired with the divergence.  The dimension (e.g., 2D)
# of the output vector field of the gradient is the dimension of the input vector field of
# the divergence, which is usually the same as the dimension of any vector field of the
# problem (e.g., 2D). example case).
#
# - To check the correctness of the implementation, consider the case of dir_diff = [2,3].
function create_grad(isfwd::SVector{K,Bool},  # isfwd[w] = true|false: ∂w is forward|backward difference
                     ∆l⁻¹::NTuple{K,AbstractVector{<:Number}},  # ∆l⁻¹[w]: inverse of distances between grid planes in w-direction
                     isbloch::SVector{K,Bool},  # boundary conditions in K dimensions
                     e⁻ⁱᵏᴸ::SVector{K,<:Number};  # Bloch phase factors in K dimensions
                     cmp_in::Integer=0,  # 0 creates compact thin matrix acting on scalar function (not vector field); ∈[1,K] creates square matrix acting on cmp_in-th Cartesian direction of input vector field
                     cmp_out::Integer=0,  # 0 creates usual gradient operator; ∈[1,K] populates entries for only one output component without changing matrix shape
                     cmpfirst::Bool=true,  # true to use Cartesian-component-major ordering for more tightly banded matrix
                     ) where {K}
    @argcheck 0 ≤ cmp_in ≤ K
    @argcheck 0 ≤ cmp_out ≤ K

    T = promote_type(eltype.(∆l⁻¹)..., eltype(e⁻ⁱᵏᴸ))  # eltype(eltype(∆l⁻¹)) can be Any if ∆l⁻¹ is inhomogeneous
    N = SVector(length.(∆l⁻¹))
    M = prod(N)
    KM = K * M  # resulting matrix: KM×M for cmp_in=0 (compact); KM×KM for cmp_in>0 (square)

    # Below, create_∂info() is called N_blk times in the for loop, and each call returns I, J, V
    # of length 2M (one diagonal and one off-diagonal), so we preallocate Itot, Jtot, Vtot
    # of length 2M × N_blk.
    cmps_out = iszero(cmp_out) ? (1:K) : (cmp_out:cmp_out)
    N_blk = length(cmps_out)  # number of nonzeo matrix blocks

    Itot = Vector{Int}(undef, 2M * N_blk)
    Jtot = Vector{Int}(undef, 2M * N_blk)
    Vtot = Vector{T}(undef, 2M * N_blk)

    ind_blk = 0  # index of matrix block
    for nw = cmps_out  # direction of differentiation (= direction of output)
        I, J, V = create_∂info(nw, isfwd[nw], N, ∆l⁻¹[nw], isbloch[nw], e⁻ⁱᵏᴸ[nw])

        istr, ioff = cmpfirst ? (K, nw-K) : (1, (nw-1)M)  # (row stride, row offset)
        @. I = istr * I + ioff

        if !iszero(cmp_in)
            jstr, joff = cmpfirst ? (K, cmp_in-K) : (1, (cmp_in-1)M)  # (column stride, column offset)
            @. J = jstr * J + joff
        end

        # For some reason, using .= below is slower because it uses 1 allocatiotn.  On the
        # other hand, using = does not use allocation and therefore faster.
        indₛ, indₑ = ind_blk*2M + 1, (ind_blk+1)*2M  # should be ind_nw instead of nw like in create_curl(), but here ind_nw = nw
        Itot[indₛ:indₑ] = I
        Jtot[indₛ:indₑ] = J
        Vtot[indₛ:indₑ] = V
        ind_blk += 1
    end

    return iszero(cmp_in) ?
        dropzeros!(sparse(Itot, Jtot, Vtot, KM, M)) :
        dropzeros!(sparse(Itot, Jtot, Vtot, KM, KM))
end
