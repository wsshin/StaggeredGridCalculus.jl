export create_πcmp, create_inv_πcmp

# Unlike curl, divergence, gradient for which the field dimension Kf and the space dimension
# K are the same, here we assume they can be different.

# Wrapper to convert AbstractVector's to SVec's.
#
# If N and permute are both SInt, do not expect this wrapper to be called even if scale is
# typed AbsVecNumber, because keyword arguments don't affect the dispatch procedure.
create_πcmp(N::AbsVecInteger,  # size of grid
            cmpperm::AbsVecInteger;  # e.g., cmpperm = [3,1,2] transforms [F1,F2,F3] to [F3,F1,F2]
            scale::AbsVecNumber=ones(Int,length(cmpperm)),  # scale[w]: scale factor to w-component of input field
            cmpfirst::Bool=true) =  # true to use Cartesian-component-major ordering for more tightly banded matrix
    (K = length(N); Kf = length(cmpperm); create_πcmp(SInt{K}(N), SInt{Kf}(cmpperm), scale=SVec{Kf}(scale); cmpfirst))

# Create the permutation matrix that permutes the order of Cartesian components of a vector
# field.
function create_πcmp(N::SInt{K},  # size of grid
                     cmpperm::SInt{Kf};  # e.g., cmpperm = [3,1,2] transforms [F1,F2,F3] to [F3,F1,F2]
                     scale::SVec{Kf,T}=SInt(ntuple(k->1, Val(Kf))),  # scale[w]: scale factor to w-component of input field
                     cmpfirst::Bool=true  # true to use Cartesian-component-major ordering for more tightly banded matrix
                     ) where {K,Kf,T<:Number}
    M = prod(N)
    KfM = Kf * M

    Itot = VecInt(undef, KfM)
    Jtot = VecInt(undef, KfM)
    Vtot = Vector{T}(undef, KfM)

    # Below, at the nv-componet of the output comes the nw-component of the input.
    for nv = 1:Kf  # component of output
        nw = cmpperm[nv]  # component of input

        istr, ioff = cmpfirst ? (Kf, nv-Kf) : (1, (nv-1)M)  # (row stride, row offset)
        jstr, joff = cmpfirst ? (Kf, nw-Kf) : (1, (nw-1)M)  # (column stride, column offset)

        indₛ, indₑ = (nv-1)*M + 1, nv*M
        @. Itot[indₛ:indₑ] = istr * (1:M) + ioff
        @. Jtot[indₛ:indₑ] = jstr * (1:M) + joff
        @. Vtot[indₛ:indₑ] = scale[nw]
    end

    return dropzeros!(sparse(Itot, Jtot, Vtot, KfM, KfM))  # dropzeros!() are in case some scale[nw] is zero
end

# Take the same argument as create_πcmp(), but returns its inverse.
function create_invπcmp(N, cmpperm; scale=SInt(ntuple(k->1, Val(length(cmpperm)))), kwargs...)
    invscale = all(isone.(scale)) ? scale : 1 ./ scale

    return transpose(create_πcmp(N, cmpperm; scale=invscale, kwargs...))
end
