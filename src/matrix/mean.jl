export create_m, create_mean

# Assumption: we don't calculate averages for interpolated fields.  In other words, we
# calculate averages for the original fields defined on Yee's grid.  (There are exceptions;
# see below.)  This assumption simplifies implementation of the symmetry boundary condition
# in the averaging operators greatly.
#
# For example, suppose we use the primal grid planes as the E-field planes.  Then, for the
# symmetry boundary condition (which is the PEC boundary condition in this case), the fields
# defined on the domain boundaries are the tangential  E-fields and normal H-fields.  Both
# fields are zero on the symmetry boundaries.  Therefore, if we assume that we calculate
# averages only for those fields defined on Yee's grid, we do not have to construct
# separate averaging operators for the tangential E-field and normal H-field.
#
# See the comments in differential.jl as well.

# Note for SALT users
# The averaging scheme in SALT will be mostly used with the Bloch boundary condition, so the
# implementation of the impact of boundary conditions in the averaging scheme is actually
# not that important, because field averaging at the boundaries is trivial for the Bloch
# boundary condition.

# The forward and (-1) × backward averaging operators are not the transpose of each other
# when the symmetry boundary condition is used: one has element with the factor of 2
# multiplied while the other with the factor of zero.  Still, the symmetry boundary enforces
# a special structure on the subpixel smoothed material parameter tensor on the symmetry
# boundary.  This tensor of a special structure has zero entries where the non-transpose
# relationship between the forward and backward averaging matters, and eventually makes the
# effect of this non-transpose relationship does not propagate up to the symmetry of the
# final operator (i.e., the final operator is still symmetric even though the forward and
# (-1) × backward averaging operators are not the transpose of each other). See my notes
# entitled [Beginning of the part added on Aug/14/2018] in RN - Subpixel Smoothing.nb.

create_mean(isfwd::AbsVecBool,  # isfwd[w] = true|false for forward|backward averaging
            N::AbsVecInteger,  # size of grid
            isbloch::AbsVecBool=fill(true,length(N)),  # for length(N) = 3, boundary conditions in x, y, z
            e⁻ⁱᵏᴸ::AbsVecNumber=ones(length(N));  # for length(N) = 3, Bloch phase factor in x, y, z
            order_cmpfirst::Bool=true) =  # true to use Cartesian-component-major ordering for more tightly banded matrix
    (∆l = ones.((N...,)); ∆l′⁻¹ = ones.((N...,)); create_mean(isfwd, N, ∆l, ∆l′⁻¹, isbloch, e⁻ⁱᵏᴸ, order_cmpfirst=order_cmpfirst))

create_mean(isfwd::AbsVecBool,  # isfwd[w] = true|false for forward|backward averaging
            N::AbsVecInteger,  # size of grid
            ∆l::NTuple{K,AbsVecNumber},  # line segments to multiply with; vectors of length N
            ∆l′⁻¹::NTuple{K,AbsVecNumber},  # inverse of line segments to divide by; vectors of length N
            isbloch::AbsVecBool=fill(true,length(N)),  # for K = 3, boundary conditions in x, y, z
            e⁻ⁱᵏᴸ::AbsVecNumber=ones(length(N));  # for K = 3, Bloch phase factor in x, y, z
            order_cmpfirst::Bool=true  # true to use Cartesian-component-major ordering for more tightly banded matrix
            ) where {K} =
    (create_mean(SBool{K}(isfwd), SInt{K}(N), ∆l, ∆l′⁻¹, SBool{K}(isbloch), SVector{K}(e⁻ⁱᵏᴸ), order_cmpfirst=order_cmpfirst))

# Creates the field-averaging operator for all three Cartegian components.
#
# The function used to support setting nonzero matrix blocks at off-diagonal blocks, but the
# need for doing it is absorbed in MaxwellFDM.jl/src/param.jl/create_param_matrix() by
# putting the material parameters at off-diagonal blocks.  Therefore, the field-averaging
# operators always the diagonal blocks as nonzero blocks, as they perform averaging on the
# already generated input and output fields.  See RN - Subpixel Smoothing > [Update (May/13/2018)].
function create_mean(isfwd::SBool{K},  # isfwd[w] = true|false for forward|backward averaging
                     N::SInt{K},  # size of grid
                     ∆l::NTuple{K,AbsVecNumber},  # line segments to multiply with; vectors of length N
                     ∆l′⁻¹::NTuple{K,AbsVecNumber},  # inverse of line segments to divide by; vectors of length N
                     isbloch::SBool{K}=SVector(ntuple(k->true,K)),  # for K = 3, boundary conditions in x, y, z
                     e⁻ⁱᵏᴸ::SNumber{K}=SVector(ntuple(k->1.0,K));  # for K = 3, Bloch phase factor in x, y, z
                     order_cmpfirst::Bool=true  # true to use Cartesian-component-major ordering for more tightly banded matrix
                     ) where {K}
    T = promote_type(eltype.(∆l)..., eltype.(∆l′⁻¹)..., eltype(e⁻ⁱᵏᴸ))  # eltype(eltype(∆l)) can be Any if ∆l is inhomogeneous
    M = prod(N)
    KM = K * M

    Itot = VecInt(undef, 2KM)
    Jtot = VecInt(undef, 2KM)
    Vtot = Vector{T}(undef, 2KM)

    indblk = 0  # index of matrix block
    for nv = 1:K  # Cartesian compotents of output field
        I, J, V = create_minfo(nv, isfwd[nv], N, ∆l[nv], ∆l′⁻¹[nv], isbloch[nv], e⁻ⁱᵏᴸ[nv])  # averaging along nv-direction

        istr, ioff = order_cmpfirst ? (K, nv-K) : (1, M*(nv-1))  # (row stride, row offset)
        nw = nv  # Cartesian component of input field; same as output field's, because we set diagonal blocks
        jstr, joff = order_cmpfirst ? (K, nw-K) : (1, M*(nw-1))  # (column stride, column offset)

        @. I = istr * I + ioff
        @. J = jstr * J + joff

        # For some reason, using .= below is slower because it uses 1 allocatiotn.  On the
        # other hand, using = does not use allocation and therefore faster.
        indₛ, indₑ = indblk*2M + 1, (indblk+1)*2M  # each I, J, V is length-2M
        Itot[indₛ:indₑ] = I
        Jtot[indₛ:indₑ] = J
        Vtot[indₛ:indₑ] = V
        indblk += 1
    end

    return dropzeros!(sparse(Itot, Jtot, Vtot, KM, KM))  # KM×KM matrix with 2 entries per row (so 2KM entries for I, J, V)
end


## Field-averaging operators ##
#
# This creates the averaging operator for a single Cartesian component.  For the operator
# for all three Cartesian components, use create_mean.
#
# Construction of these operators are similar to that of difference operators.  However,
# unlike the difference operators that are primarilly used for curl and hence differentiates
# the fields along the direction normal to the fields, the field-averaging operators average
# fields along the field direction.  As a result, backward (rather than forward) averaging
# is for primal fields.
create_m(nw::Integer,  # for length(N) = 3, 1|2|3 for averaging along x|y|z
         isfwd::Bool,  # true|false for forward|backward averaging
         N::AbsVecInteger,  # size of grid
         isbloch::Bool=true,  # boundary condition in w-direction
         e⁻ⁱᵏᴸ::Number=1.0) =  # Bloch phase factor
    (K = length(N); ∆w = ones(N[nw]); ∆w′⁻¹ = ones(N[nw]); create_m(nw, isfwd, SInt{K}(N), ∆w, ∆w′⁻¹, isbloch, e⁻ⁱᵏᴸ))

create_m(nw::Integer,  # for length(N) = 3, 1|2|3 for averaging along x|y|z
         isfwd::Bool,  # true|false for forward|backward averaging
         N::AbsVecInteger,  # size of grid
         ∆w::AbsVecNumber,  # line segments to multiply with; vector of length N[nw]
         ∆w′⁻¹::AbsVecNumber,  # inverse of line segments to divide by; vector of length N[nw]
         isbloch::Bool=true,  # boundary condition in w-direction
         e⁻ⁱᵏᴸ::Number=1.0) =  # Bloch phase factor
    (K = length(N); M = prod(N); dropzeros!(sparse(create_minfo(nw, isfwd, SInt{K}(N), ∆w, ∆w′⁻¹, isbloch, e⁻ⁱᵏᴸ)..., M, M)))


function create_minfo(nw::Integer,  # for K = 3, 1|2|3 for averaging along x|y|z
                      isfwd::Bool,  # true|false for forward|backward averaging
                      N::SInt{K},  # size of grid
                      ∆w::AbsVecNumber,  # line segments to multiply with; vector of length N[nw]
                      ∆w′⁻¹::AbsVecNumber,  # inverse of line segments to divide by; vector of length N[nw]
                      isbloch::Bool,  # boundary condition in w-direction
                      e⁻ⁱᵏᴸ::Number  # Bloch phase factor
                      ) where {K}
    M = prod(N)
    Nw = N[nw]
    ŵ = SVector(ntuple(identity,Val(K))) .== nw  # [0,true,0] for w == y
    ns = isfwd ? 1.0 : -1.0  # number for sign

    # Below, when constructing I, J's, V's, note that a tuple of array subscripts (i,j,k)
    # indicates a row index of the final matrix.  In other words, the entries with the same
    # subscripts correspond to the same *output* field.  So, when setting I[i,j,k], J₀[i,j,k],
    # Jₛ[i,j,k], V₀[i,j,k], Vₛ[i,j,k], we must know that they are for the entries in the
    # same row index (or the same output field).  Specifically,
    #
    # - I is the row index itself, so it is the identity map: I[i,j,k] = LinearIndices(N)[i,j,k].
    #
    # - J₀[i,j,k] is the column index of the diagonal entry in the row subscripted (i,j,k).
    # Because the row and column indices are the same for the diagonal entries, J₀ = I and
    # not created separately.
    #
    # - Jₛ[i,j,k] is the column index of the off-diagonal entry in the row subscripted
    # (i,j,k).
    #
    # - V₀[i,j,k] is the value of the diagonal entry in the row subscripted (i,j,k).
    #
    # - Vₛ[i,j,k] is the value fo the off-diagonal entry in the row subscripted (i,j,k).

    I = reshape(collect(1:M), N.data)

    Jₛ = reshape(collect(1:M), N.data)
    shifts = -ns * ŵ  # shift vector for sign; [0,-1,0] for w == y and ns = +1
    Jₛ = circshift(Jₛ, shifts.data)

    # Align ∆w and ∆w′⁻¹ in the w-direction.
    vec1 =  @SVector ones(Int,K)
    sizew = @. !ŵ * vec1 + ŵ * N  # [1,Ny,1] for w == y
    ∆W = reshape(∆w, sizew.data)
    ∆W′⁻¹ = reshape(∆w′⁻¹, sizew.data)

    # Construct the values of the diagonal and off-diagonal nonzero entries of the matrix
    # for the periodic (≠ Bloch) boundary condition.  (These values will be updated for
    # specific boundary conditions later.)
    #
    # To figure out the entries of V₀ and Vₛ, it is most convenient to figure out the entire
    # 3D arrays V₀ and Vₛ, instead of entry by entry.  In doing so, note that the entries
    # with the same subscripts correspond to the same *output* field (= same row index).
    T = promote_type(eltype(∆w), eltype(∆w′⁻¹), eltype(e⁻ⁱᵏᴸ))
    V₀ = fill(T(0.5), N.data) .* ∆W .* ∆W′⁻¹  # values of diagonal entries

    Vₛ = fill(T(0.5), N.data) .* ∆W  # values of off-diagonal entries (division later)
    Vₛ = circshift(Vₛ, shifts.data)
    Vₛ .*= ∆W′⁻¹

    # Modify V₀ and Vₛ according to the boundary condition.  What we do is basically to take
    # the operator for the periodic boundary condition (constructed above) as a template and
    # then modify it for the actual boundary condition, either by applying the Bloch phase
    # factors for the Bloch boundary condition or by multiplying 0 or 2 to some diagonal and
    # off-diagonal entries for the symmetry boundary condition.  Because this modification
    # is due to the boundary condition, the modification needs to be made only at the array
    # entries with the first or last subscript of the Cartesion direction in which averaging
    # is taken.
    #
    # The outline of the procedure is as follows:
    #
    # 1. Detertmine the rows in which the modification should be made.  This is done by
    # figuring out the array subscripts (i,j,k) of the output average array entry to which
    # the input boundary fields contribute.
    #
    # 2. In each of these rows, determine the column in which the modification should be
    # made.  Because each row has only two nonzero entries (one diagoal and one off-diagonal),
    # this step is to determine whether the modification should be made at the diagonal
    # entry or the off-diagonal entry in the row currently concerned.  If the (i,j,k) of the
    # input fields contributing to the currently concerned output averages are the same as
    # the (i,j,k) of the output averages, the modification is made in V₀ (diagonal).  If
    # they are different, the modification is made in Vₛ (off-diagonal).
    #
    # 3. In the determined rows, modify either V₀ or Vₛ depending on the above step.  This
    # is to modify V₀[i,j,k] or Vₛ[i,j,k], where (i,j,k) is the array subscripts specifying
    # the determined row.
    #
    # For the final sparsity patterns of the operators, see my notes entitled [Beginning of
    # the part added on Aug/14/2018] in RN - Subpixel Smoothing.nb.
    #
    # Below, V[Base.setindex(axes(V), iw:iw, nw)...] mimics the implementation of slicedim
    # and means V[:,iw:iw,:] for w = y.  (axes(V) = (1:Nx,1:Ny,1:Nz) and setindex(t,v,i)
    # does t[i] = v, so Base.setindex(axes(V),iw:iw,nw) creates (1:Nx,iw:iw,1:Nz) for w = y.)
    # The use of iw:iw instead of iw is to support 1D arrays.  If V is 1D, then V[iw] is a
    # scalar and the dot equal on V[iw] fails.
    if isbloch
        # Application of the aformentioned procedure:
        #
        # 0. Goal
        # Our goal is to create the ghost fields on one boundary by multiplying the Bloch
        # phase factors to the nonghost fields on the opposite boundary.  Note that w is the
        # direction of averaging.
        #
        # 1. Determination of the indices of the rows to modify
        # For isfwd = true, we are taking forward averaging between primal grid planes.  The
        # ghost input fields are at the positive-end boundary, and they contritube to the
        # averages at the last dual grid points, so the rows to modify are subscripted (i,j,k)
        # with iw ≡ (i,j,k)[w] = Nw.
        #
        # For isfwd = false, we are taking backward averaging between dual grid planes.  The
        # ghost input fields are before the negative-end boundary, and they contribute to
        # the averages at the first primal grid points, so the rows to modify are
        # subscripted (i,j,k) with iw ≡ (i,j,k)[w] = 1.
        #
        # 2. Determination of the column in each row to modify
        # For isfwd = true, the w-subscript of the ghost input field is greater by 1 than
        # that of the output average.
        #
        # For isfwd = false, the w-subscript of the ghost input field is less by 1 than that
        # of the output average.
        #
        # The nonghost input fields are brought to these ghost fields (from the opposite
        # boundary), an they are different from the input fields subscripted the same as the
        # output averages.  Therefore, the matrix entries to modify by multiplying the Bloch
        # phase factors with are off-diagonal entries in Vₛ.
        #
        # 3. Modification of the values in V
        # For isfwd = true, the nonghost input fields have a smaller w-subscript than the
        # ghost input fields that they are brought into.  Therefore, e⁻ⁱᵏᴸ must be
        # multiplied to the nonghost input fields to create the ghost input fields.
        #
        # For isfwd = false, the nonghost input fields have a larger w-subscript than the
        # ghost input fields that they are brought into.  Therefore, e⁺ⁱᵏᴸ must be
        # multiplied to the nonghost input fields to create the ghost input fields.
        iw = isfwd ? Nw : 1
        Vₛ[Base.setindex(axes(Vₛ), iw:iw, nw)...] .*= e⁻ⁱᵏᴸ^ns
    else  # symmetry bounndary
        # A. isfwd = true (forward averaging)
        #
        # A-0. Goal
        # Forward averaging is taken between primal grid planes.  Because the domain
        # boundaries are always primal grid planes, some input fields (either tangential or
        # normal) are on the domain boundaries.  Those input fields are zero for the
        # symmetry boundary condition and do not contribute to the output averages.  Our
        # goal is to make sure that the forward averaging operator properly ignore those
        # fields that should be zero.
        #
        # A-1. Determination of the indices of the the rows to modify
        # The negative-end-boundary input fields contribute to the w-directional averages
        # evaluated at the first (nonghost) dual grid points in the w-direction.  The
        # positive-end-boundary (ghost) input fields contribute to the w-directional
        # averages evaluated at the last dual grid points in the w-direction.  Therefore,
        # the array subscripts (i,j,k) we need to consider are the ones with iw ≡ (i,j,k)[w] = 1, Nw.
        #
        # A-2. Determination of the column in each row to modify
        # The negative-end-boundary input fields have subscripts the same as the averages
        # they contribute to, so for iw = 1 we modify the diagonal entries stored in V₀.
        # The positive-end-boundary input fields have w-subscript greater by 1 than the
        # averages they contribute to, so for iw = Nw we modify the off-diagonal entries
        # stored in Vₛ.
        #
        # A-3. Modification of the values in V
        # The negative-end-boundary input fields must be zeroed, so for (i,j,k) with iw = 1
        # we set V₀[i,j,k] = 0.
        # The positive-end-boundary input fields must be also zeroed, so for (i,j,k) with iw = Nw
        # we set Vₛ[i,j,k] = 0.  Note that the positive-end-boundary input fields are ghost
        # fields that are brought from the negative-end-boundary (nonghost) input fields.
        #
        #
        # B. isfwd = false (backward averaging)
        #
        # B-0. Goal
        # Backward averaging is taken between dual grid planes.  The domain boundaries are
        # always primal grid planes.  Therefore, no input fields (either tangential or
        # normal) are on the domain boundaries, and thus no input fields needs to be zeroed.
        #
        # However, the field at dual grid points are symmetric (rather than anti-symmetric)
        # around the symmetry boundaries (so that their interpolated values at the symmetry
        # boundary are nonzero). Therefore, the average between the two is double the
        # contribution from one input field, leading to multiplication of a factor of 2 at
        # the symmetry boundaries.
        #
        # On the positive-end boundary, the average of the dual field is not evaluated,
        # so we don't have to do anything to realize the positivte-end symmetry boundary.
        # On the other hand, on the negative-end boundary the average of the dual field is
        # evaluated.  We have to make sure that the resulting average is double the
        # contribution from one input field there.
        #
        # B-1. Determination of the indices of the the rows to modify
        # The output averages at the negative-end boundary are subscripted (i,j,k) with
        # iw ≡ (i,j,k)[w] = 1.
        #
        # B-2. Determination of the column in each row to modify
        # The output averages at the negative-end boundary is taken between the first
        # (nonghost) dual grid point and the ghost point before that point.  We can realize
        # the symmetry boundary by doubling the effect of the input field at the first
        # dual grid point and zeroing the effect of the input field at the ghost point.
        #
        # The input field at the first dual grid point is subscripted the same as the output
        # average, so its effect is described in a diagonal entry in V₀.  The input field at
        # the ghost point is subscripted 1 less in the w-direction, so its effect is
        # described in an off-diagonal entry in Vₛ.
        #
        # B-3. Modification of the values in V
        # We double V₀[i,j,k] and zero Vₛ[i,j,k] for (i,j,k) with iw = 1.
        #
        # The above procedure A and B are written compactly below without separating the
        # isfwd = true and false cases.


        # Replace some diagonal entries (V₀) with 0 or 2.
        iw = 1
        val = isfwd ? 0 : 2
        V₀[Base.setindex(axes(V₀), iw:iw, nw)...] .*= val

        # Replace some off-diagonal entries (Vₛ) with 0.
        iw = isfwd ? Nw : 1
        Vₛ[Base.setindex(axes(Vₛ), iw:iw, nw)...] .= 0

        # Note that the forward and backward averaging operators are not the transpose of
        # each other because they have different diagonals.  This may seem to make the final
        # material parameter matrix (with the input and output field averaging operators
        # multiplied) nonsymmetric.  However, zeros in the material parameter tensor
        # enforced by the symmetry boundary condition nullifies the possibility of
        # nonsymmetry.  See my notes entitled [Beginning of the part added on Aug/14/2018]
        # in RN - Subpixel Smoothing.nb.  See also kottke_input_accurate in smoothing.jl.
    end

    i = reshape(I, M)
    jₛ = reshape(Jₛ, M)
    v₀ = reshape(V₀, M)
    vₛ = reshape(Vₛ, M)

    Is = [i; i]  # row indices of [diagonal; off-diagonal] entries
    Js = [i; jₛ]  # column indices of [diagonal; off-diagonal] entries
    Vs = [v₀; vₛ]  # matrix entries of [diagonal, off-diagonal] entris

    return Is, Js, Vs
end
