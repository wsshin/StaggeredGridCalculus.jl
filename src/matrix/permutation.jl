export create_πcmp

# Unlike curl, divergence, gradient for which the field dimension Kf and the space dimension
# K are the same, here we assume they can be different.

# Wrapper to convert AbstractVector's to SVec's.
create_πcmp(N::AbsVecInteger,
            permute::AbsVecInteger;  # permute[w]: component of output field where w-component of input field is placed; e.g., permute = [3,1,2] transforms [F1,F2,F3] to [F2,F3,F1]
            scale::AbsVecNumber=ones(length(permute)),  # scale[w]: scale factor to w-component of input field
            order_cmpfirst::Bool=true) =  # true to use Cartesian-component-major ordering for more tightly banded matrix
    (K = length(N); Kf = length(permute); create_πcmp(SInt{K}(N), SInt{Kf}(permute), scale=SVec{Kf}(scale), order_cmpfirst=order_cmpfirst))

# Create the permutation matrix that permutes the order of Cartesian components of a vector
# field.
function create_πcmp(N::SInt{K},  # size of grid
                     permute::SInt{Kf};  # permute[w]: component of output field where w-component of input field is placed; e.g., permute = [3,1,2] transforms [F1,F2,F3] to [F2,F3,F1]
                     scale::SVec{Kf,T}=SFloat(ntuple(k->1.0, Val(Kf))),  # scale[w]: scale factor to w-component of input field
                     order_cmpfirst::Bool=true  # true to use Cartesian-component-major ordering for more tightly banded matrix
                     ) where {K,Kf,T}
    M = prod(N)
    KfM = Kf * M

    Itot = VecInt(undef, KfM)
    Jtot = VecInt(undef, KfM)
    Vtot = Vector{T}(undef, KfM)

    for nw = 1:Kf  # component of input
        nv = permute[nw]  # component of output where w-component of input is placed

        istr, ioff = order_cmpfirst ? (Kf, nv-Kf) : (1, M*(nv-1))  # (row stride, row offset)
        jstr, joff = order_cmpfirst ? (Kf, nw-Kf) : (1, M*(nw-1))  # (column stride, column offset)

        indₛ, indₑ = (nw-1)*M + 1, nw*M
        @. Itot[indₛ:indₑ] = istr * (1:M) + ioff
        @. Jtot[indₛ:indₑ] = jstr * (1:M) + joff
        @. Vtot[indₛ:indₑ] = scale[nw]
    end

    return dropzeros!(sparse(Itot, Jtot, Vtot, KfM, KfM))  # dropzeros!() are in case some scale[nw] is zero
end
