export create_πcmp, create_inv_πcmp

# Unlike curl, divergence, gradient for which the field dimension Kf and the space dimension
# K are the same, here we assume they can be different.

# I wanted to make cmpperm and cellperm keyword arguments with default values such that the
# users can specify only one of them depending on whether they want to permutate components
# or cells.  (If cmpperm and cellperm are regular, positional arguments, the one that comes
# later cannot be specified without specifing the one that comes earlier, even if we set
# their default values.)
#
# However, it turns out that unlike the default value 1:K of cellperm, the default value 1:Kf
# of cmpperm cannot be derived from N, because Kf is not necessarily K = length(N).  This
# means that even if we make cmpperm and cellperm keyword arguments, we cannot provide the
# default value for cmpperm, and therefore the users cannot specify only cellperm.
#
# This is why I made both cmpperm and cellperm positional arguments: if cmpperm always needs
# a user-specified value, making it a positional argument makes more sense.  If we make
# cmpperm a positional argument, it makes more sense to make cellperm a positional argument
# as well.

# Wrapper to convert AbstractVector's to SVec's.
#
# If N and permute are both SInt, do not expect this wrapper to be called even if scale is
# typed AbsVecNumber, because keyword arguments don't affect the dispatch procedure.
create_πcmp(N::AbsVecInteger,
            cmpperm::AbsVecInteger,  # e.g., cmpperm = [3,1,2] transforms [F1,F2,F3] to [F3,F1,F2]
            cellperm::AbsVecInteger=1:length(N);  # e.g., cellperm = [3,1,2] makes z-index fastest-varying and y-index slowest-varying
            scale::AbsVecNumber=ones(Int,length(cmpperm)),  # scale[w]: scale factor to w-component of input field
            cmpfirst::Bool=true) =  # true to use Cartesian-component-major ordering for more tightly banded matrix
    (K = length(N); Kf = length(cmpperm);
    create_πcmp(SInt{K}(N), SInt{Kf}(cmpperm), SInt{K}(cellperm), scale=SVec{Kf}(scale), cmpfirst=cmpfirst))

# Create the permutation matrix that permutes the order of Cartesian components of a vector
# field.
function create_πcmp(N::SInt{K},  # size of grid
                     cmpperm::SInt{Kf},  # e.g., cmpperm = [3,1,2] transforms [F1,F2,F3] to [F3,F1,F2]
                     cellperm::SInt{K}=SInt(ntuple(k->1, Val(K)));  # e.g., cellperm = [3,1,2] makes z-index fastest-varying and y-index slowest-varying
                     scale::SVec{Kf,T}=SInt(ntuple(k->1, Val(Kf))),  # scale[w]: scale factor to w-component of input field
                     cmpfirst::Bool=true  # true to use Cartesian-component-major ordering for more tightly banded matrix
                     ) where {K,Kf,T<:Number}
    M = prod(N)
    KfM = Kf * M

    Itot = VecInt(undef, KfM)
    Jtot = VecInt(undef, KfM)
    Vtot = Vector{T}(undef, KfM)

    # The goal of cellperm is to permute the elements of an input vector, whose indices are
    # linearized cell indices in the order of (x,y,z), into an output vector whose indices
    # are linearized cell indices in the order of (cellperm[1], cellperm[2], cellperm[3]).
    #
    # If we want to permute an input 1:N to a permutation p, the ith element of the output
    # vector is p[i].  This means that the desired permutation matrix has 1's at (i, p[i]).
    # In other words, the desired permutation matrix is constructed sparse(1:N, p, ones(Int,N)).
    #
    # Based on this observation, we realize that the desired permutation matrix for the
    # linearized cell indices is constructed spare(out_LI, in_LI, ones(...)) below:
    out_LI = LinearIndices(N.data)   # i-indices (indices of output element); linear indices organized as K-dimensional array
    in_LI = PermutedDimsArray(out_LI, cellperm.data)  # j-indices (indices of input element)

    out_li = out_LI[:]
    in_li = in_LI[:]

    # Below, at the nv-componet of the output comes the nw-component of the input.
    for nv = 1:Kf  # component of output
        nw = cmpperm[nv]  # component of input

        istr, ioff = cmpfirst ? (Kf, nv-Kf) : (1, M*(nv-1))  # (row stride, row offset)
        jstr, joff = cmpfirst ? (Kf, nw-Kf) : (1, M*(nw-1))  # (column stride, column offset)

        indₛ, indₑ = (nv-1)*M + 1, nv*M
        @. Itot[indₛ:indₑ] = istr * out_li + ioff
        @. Jtot[indₛ:indₑ] = jstr * in_li + joff
        @. Vtot[indₛ:indₑ] = scale[nw]
    end

    return dropzeros!(sparse(Itot, Jtot, Vtot, KfM, KfM))  # dropzeros!() are in case some scale[nw] is zero
end

# Take the same argument as create_πcmp(), but returns its inverse.
function create_inv_πcmp(N, cmpperm, cellperm=SInt(ntuple(identity, length(N))); scale=SInt(ntuple(k->1, Val(length(cmpperm)))), kwargs...)
    invscale = all(isone.(scale)) ? scale : 1 ./ scale

    return transpose(create_πcmp(N, cmpperm, cellperm; scale=invscale, kwargs...))
end
