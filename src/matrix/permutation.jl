export create_πcmp, create_πcell, create_invπcmp, create_invπcell

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
                     scale::SVec{Kf,T}=@SVector(ones(Int,Val(Kf))),  # scale[w]: scale factor to w-component of input field
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
function create_invπcmp(N, cmpperm; scale=@SVector(ones(Int, Val(length(cmpperm)))), kwargs...)
    invscale = all(isone.(scale)) ? scale : 1 ./ scale

    return transpose(create_πcmp(N, cmpperm; scale=invscale, kwargs...))
end

create_πcell(
    N::AbsVecInteger,  # size of grid
    Kf::Integer,  # number of Cartesian components
    cellperm::AbsVecInteger;  # e.g., cellperm = [3,1,2] makes z-index fastest-varying and y-index slowest-varying
    cmpfirst::Bool=true  # true to use Cartesian-component-major ordering for more tightly banded matrix
) = (K = length(N); create_πcell(SInt{K}(N), Int(Kf), SInt{K}(cellperm); cmpfirst))

# Create the permutation matrix that permutes the order of Cartesian components of a vector
# field.
#
# Unlike component permutation, cell permutation does not require the scale keyword.  We may
# have different equations for different Cartesian components, so component permutation,
# which is used for constructing different equations for different Cartesian components,
# need scale.
#
# We certainly have different differentiations for different Cartesian directions, and may
# have different signs or scale constants for derivatives along different Cartesian
# directions.  However, that cannot be implemented by multiplying scales during cell
# permutation, because the same cell participates in derivatives along different directions,
# which means that different scales need to be multiplied to the same cells.
function create_πcell(
    N::SInt{K},  # size of grid
    Kf::Int,  # number of Cartesian components
    cellperm::SVector{K,Int};  # e.g., cellperm = [3,1,2] makes z-index fastest-varying and y-index slowest-varying
    cmpfirst::Bool=true  # true to use Cartesian-component-major ordering for more tightly banded matrix
) where {K}
    M = prod(N)
    KfM = Kf * M

    Itot = VecInt(undef, KfM)
    Jtot = VecInt(undef, KfM)
    Vtot = Vector{Bool}(undef, KfM)

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

    # Below, we don't have separate indices for the input and output Cartesian components,
    # because this function does not permute the order of Cartesian components between the
    # input and output.  In other words, if cmpfirst=false such that the (i,j) matrix block
    # of the 3×3 block matrix corresponds to the ith Cartesian component of the output and
    # jth Cartesian component of the input, then the matrix is block-diagonal.
    for ncmp = 1:Kf  # index of component
        str, off = cmpfirst ? (Kf, ncmp-Kf) : (1, (ncmp-1)M)  # (row and column stride, row and column offset)

        indₛ, indₑ = (ncmp-1)*M + 1, ncmp*M
        @. Itot[indₛ:indₑ] = str * out_li + off
        @. Jtot[indₛ:indₑ] = str * in_li + off
        @. Vtot[indₛ:indₑ] = true
    end

    return dropzeros!(sparse(Itot, Jtot, Vtot, KfM, KfM))  # dropzeros!() are in case some scale[nw] is zero
end

# Take the same argument as create_πcell(), but returns its inverse.
create_invπcell(N, Kf, cellperm; kwargs...) = transpose(create_πcell(N, Kf, cellperm; kwargs...))
