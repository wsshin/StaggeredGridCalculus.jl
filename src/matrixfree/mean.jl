# Average fields along the field directions (i.e., Fw along the w-direction).  See the
# description of matrix/mean.jl/create_m̂().  Technically, apply_m̂!() can be used to average
# fields normal to the direction of averaging, but it is not used that way in apply_mean!().

# Assumes the space dimension and field dimension are the same.  In other words, when the
# space coordinate indices are (i,j,k), then the field has three vector components.
# Therefore, for the input field array F[i,j,k,w], we assume w = 1:3.

# The functions calculate the average and add to the output array, instead of replacing the
# values stored in the output array.  Therefore, if the derivative values themselves are
# desired, pass the output array initialized with zeros.

# Inside @threads, avoid defining β, which requires another let block to avoid unnecessary
# allocations.

export apply_m̂!, apply_mean!

# Wrapper for converting AbstractVector's to SVector's for arithmetic averaging
# This corresponds to the wrappers to create discrete versions of differential operators,
# but the nonzero enries in the matrix are 0.5 rather than 1.0.
apply_mean!(G::AbsArrNumber,  # output field; G[i,j,k,w] is w-component of G at (i,j,k)
            F::AbsArrNumber,  # input field; G[i,j,k,w] is w-component of G at (i,j,k)
            ::Val{OP},  # Val(:(=)) or Val(:(+=)): set (=) or add (+=) operator to use
            isfwd::AbsVecBool,  # isfwd[w] = true|false for forward|backward averaging
            isbloch::AbsVecBool=fill(true,length(isfwd)),  # boundary conditions in x, y, z
            e⁻ⁱᵏᴸ::AbsVecNumber=ones(length(isfwd));  # Bloch phase factor in x, y, z
            α::Number=1.0  # scale factor to multiply to result before adding it to G: G += α mean(F)
            ) where {OP} =
    (K = length(isfwd); apply_mean!(G, F, Val(OP), SBool{K}(isfwd), SBool{K}(isbloch), SVector{K}(e⁻ⁱᵏᴸ), α=α))

# Wrapper for converting AbstractVector's to SVector's for weighted averaging
apply_mean!(G::AbsArrNumber,  # output field; G[i,j,k,w] is w-component of G at (i,j,k)
            F::AbsArrNumber,  # input field; G[i,j,k,w] is w-component of G at (i,j,k)
            ::Val{OP},  # Val(:(=)) or Val(:(+=)): set (=) or add (+=) operator to use
            isfwd::AbsVecBool,  # isfwd[w] = true|false for forward|backward averaging
            ∆l::Tuple{Vararg{AbsVecNumber}},  # line segments to multiply with; vectors of length N
            ∆l′⁻¹::Tuple{Vararg{AbsVecNumber}},  # inverse of line segments to divide by; vectors of length N
            isbloch::AbsVecBool=fill(true,length(isfwd)),  # boundary conditions in x, y, z
            e⁻ⁱᵏᴸ::AbsVecNumber=ones(length(isfwd));  # Bloch phase factor in x, y, z
            α::Number=1.0  # scale factor to multiply to result before adding it to G: G += α mean(F)
            ) where {OP} =
    (K = length(isfwd); apply_mean!(G, F, Val(OP), SBool{K}(isfwd), ∆l, ∆l′⁻¹, SBool{K}(isbloch), SVector{K}(e⁻ⁱᵏᴸ), α=α))

# Concrete implementation for arithmetic averaging
# Initially this was implemented by passing a tuple of ones() os ∆l and ∆l′⁻¹, but because
# arithmetic averaging is used frequently, the optimized version is implemented separately.
function apply_mean!(G::AbsArrNumber{K₊₁},  # output field; G[i,j,k,w] is w-component of G at (i,j,k)
                     F::AbsArrNumber{K₊₁},  # input field; G[i,j,k,w] is w-component of G at (i,j,k)
                     ::Val{OP},  # Val(:(=)) or Val(:(+=)): set (=) or add (+=) operator to use
                     isfwd::SBool{K},  # isfwd[w] = true|false for forward|backward averaging
                     isbloch::SBool{K},  # boundary conditions in x, y, z
                     e⁻ⁱᵏᴸ::SNumber{K};  # Bloch phase factor in x, y, z
                     α::Number=1.0  # scale factor to multiply to result before adding it to G: G += α mean(F)
                     ) where {K,K₊₁,OP}  #  K is space dimension; K₊₁ = K + 1
    @assert K₊₁==K+1
    for nw = 1:K  # direction of averaging
        nv = nw
        Gv = selectdim(G, K₊₁, nv)  # nv-th component of output field

        nu = nw  # component of input field to feed to w-directional averaging
        Fu = selectdim(F, K₊₁, nu)  # nu-th component of input field

        apply_m̂!(Gv, Fu, Val(OP), nv, isfwd[nv], isbloch[nv], e⁻ⁱᵏᴸ[nv], α=α)
    end
end

# Concrete implementation for weighted averaging
# For the implementation details, see the comments in matrix/mean.jl.
function apply_mean!(G::AbsArrNumber{K₊₁},  # output field; G[i,j,k,w] is w-component of G at (i,j,k)
                     F::AbsArrNumber{K₊₁},  # input field; G[i,j,k,w] is w-component of G at (i,j,k)
                     ::Val{OP},  # Val(:(=)) or Val(:(+=)): set (=) or add (+=) operator to use
                     isfwd::SBool{K},  # isfwd[w] = true|false for forward|backward averaging
                     ∆l::NTuple{K,AbsVecNumber},  # line segments to multiply with; vectors of length N
                     ∆l′⁻¹::NTuple{K,AbsVecNumber},  # inverse of line segments to divide by; vectors of length N
                     isbloch::SBool{K},  # boundary conditions in x, y, z
                     e⁻ⁱᵏᴸ::SNumber{K};  # Bloch phase factor in x, y, z
                     α::Number=1.0  # scale factor to multiply to result before adding it to G: G += α mean(F)
                     ) where {K,K₊₁,OP}  #  K is space dimension; K₊₁ = K + 1
    @assert K₊₁==K+1
    for nw = 1:K  # direction of averaging
        nv = nw
        Gv = selectdim(G, K₊₁, nv)  # nv-th component of output field

        nu = nw  # component of input field to feed to w-directional averaging
        Fu = selectdim(F, K₊₁, nu)  # nu-th component of input field

        apply_m̂!(Gv, Fu, Val(OP), nv, isfwd[nv], ∆l[nv], ∆l′⁻¹[nv], isbloch[nv], e⁻ⁱᵏᴸ[nv], α=α)
    end
end

## Field-averaging operators "m̂" (as used in de Moerloose and de Zutter)
#
# This applies the averaging operator for a single Cartesian component.  For the operator
# for all three Cartesian components, use apply_mean!.

# Below, I set the default value for e⁻ⁱᵏᴸ below, even though I don't usually set default
# values for the arguments in concrete implementation (except for keyword arguments, which
# require default values).  This is to make the arguments of apply_m̂!() to have default
# values when the corresponding arguments of create_m̂() do.

# Arithmetic m̂ for 3D
function apply_m̂!(Gv::AbsArrNumber{3},  # v-component of output field (v = x, y, z)
                  Fu::AbsArrNumber{3},  # u-component of input field (u = x, y, z)
                  ::Val{OP},  # Val(:(=)) or Val(:(+=)): set (=) or add (+=) operator to use
                  nw::Integer,  # 1|2|3 for averaging along x|y|z
                  isfwd::Bool,  # true|false for forward|backward averaging
                  isbloch::Bool,  # boundary condition in w-direction
                  e⁻ⁱᵏᴸ::Number=1.0;  # Bloch phase factor
                  α::Number=1.0  # scale factor to multiply to result before adding it to Gv: Gv += α m(Fu)
                  ) where {OP}
    @assert size(Gv)==size(Fu)

    Nx, Ny, Nz = size(Fu)
    α2 = 0.5α

    # Make sure not to include branches inside for loops.
    if isfwd
        if nw == 1  # w = x
            if isbloch
                # 1. At locations except for the positive end of the x-direction
                @threads for k = 1:Nz
                    for j = 1:Ny, i = 1:Nx-1
                        @inbounds set_or_add!(Gv, (i,j,k), α2 * (Fu[i+1,j,k] + Fu[i,j,k]), Val(OP))
                    end
                end

                # 2. At the positive end of the x-direction (where the boundary fields are
                # taken from the negative-end boundary)
                for k = 1:Nz, j = 1:Ny
                    @inbounds set_or_add!(Gv, (Nx,j,k), α2 * (e⁻ⁱᵏᴸ*Fu[1,j,k] + Fu[Nx,j,k]), Val(OP))  # Fu[Nx+1,j,k] = exp(-i kx Lx) * Fu[1,j,k]
                end
            else  # symmetry boundary
                # 1. At the locations except for the positive and negative ends of the
                # x-direction
                @threads for k = 1:Nz
                    for j = 1:Ny, i = 2:Nx-1
                        @inbounds set_or_add!(Gv, (i,j,k), α2 * (Fu[i+1,j,k] + Fu[i,j,k]), Val(OP))
                    end
                end

                # 2. At the negative end of the x-direction (where the boundary fields are
                # assumed zero)
                for k = 1:Nz, j = 1:Ny
                    @inbounds set_or_add!(Gv, (1,j,k), α2 * Fu[2,j,k], Val(OP))  # Fu[1,j,k] == 0
                end

                # 3. At the positive end of the x-direction (where the boundary fields are
                # assumed zero)
                for k = 1:Nz, j = 1:Ny
                    @inbounds set_or_add!(Gv, (Nx,j,k), α2 * Fu[Nx,j,k], Val(OP))  # Fu[Nx+1,j,k] == 0
                end
            end
        elseif nw == 2  # w = y
            if isbloch
                # 1. At locations except for the positive end of the y-direction
                @threads for k = 1:Nz
                    for j = 1:Ny-1, i = 1:Nx
                        @inbounds set_or_add!(Gv, (i,j,k), α2 * (Fu[i,j+1,k] + Fu[i,j,k]), Val(OP))
                    end
                end

                # 2. At the positive end of the y-direction (where the boundary fields are
                # taken from the negative-end boundary)
                for k = 1:Nz, i = 1:Nx
                    @inbounds set_or_add!(Gv, (i,Ny,k), α2 * (e⁻ⁱᵏᴸ*Fu[i,1,k] + Fu[i,Ny,k]), Val(OP))  # Fu[i,Ny+1,k] = exp(-i ky Ly) * Fu[i,1,k]
                end
            else  # symmetry boundary
                # 1. At the locations except for the positive and negative ends of the
                # y-direction
                @threads for k = 1:Nz
                    for j = 2:Ny-1, i = 1:Nx
                        @inbounds set_or_add!(Gv, (i,j,k), α2 * (Fu[i,j+1,k] + Fu[i,j,k]), Val(OP))
                    end
                end

                # 2. At the negative end of the y-direction (where the boundary fields are
                # assumed zero)
                for k = 1:Nz, i = 1:Nx
                    @inbounds set_or_add!(Gv, (i,1,k), α2 * Fu[i,2,k], Val(OP))  # Fu[i,1,k] == 0
                end

                # 3. At the positive end of the y-direction (where the boundary fields are
                # assumed zero)
                for k = 1:Nz, i = 1:Nx
                    @inbounds set_or_add!(Gv, (i,Ny,k), α2 * Fu[i,Ny,k], Val(OP))  # Fu[i,Ny+1,k] == 0
                end
            end
        else  # nw == 3; w = z
            if isbloch
                # 1. At locations except for the positive end of the z-direction
                @threads for k = 1:Nz-1
                    for j = 1:Ny, i = 1:Nx
                        @inbounds set_or_add!(Gv, (i,j,k), α2 * (Fu[i,j,k+1] + Fu[i,j,k]), Val(OP))
                    end
                end

                # 2. At the positive end of the z-direction (where the boundary fields are
                # taken from the negative-end boundary)
                for j = 1:Ny, i = 1:Nx
                    @inbounds set_or_add!(Gv, (i,j,Nz), α2 * (e⁻ⁱᵏᴸ*Fu[i,j,1] + Fu[i,j,Nz]), Val(OP))  # Fu[i,j,Nz+1] = exp(-i kz Lz) * Fu[i,j,1]
                end
            else  # symmetry boundary
                # 1. At the locations except for the positive and negative ends of the
                # z-direction
                @threads for k = 2:Nz-1
                    for j = 1:Ny, i = 1:Nx
                        @inbounds set_or_add!(Gv, (i,j,k), α2 * (Fu[i,j,k+1] + Fu[i,j,k]), Val(OP))
                    end
                end

                # 2. At the negative end of the z-direction (where the boundary fields are
                # assumed zero)
                for j = 1:Ny, i = 1:Nx
                    @inbounds set_or_add!(Gv, (i,j,1), α2 * Fu[i,j,2], Val(OP))  # Fu[i,j,1] == 0
                end

                # 3. At the positive end of the z-direction (where the boundary fields are
                # assumed zero)
                for j = 1:Ny, i = 1:Nx
                    @inbounds set_or_add!(Gv, (i,j,Nz), α2 * Fu[i,j,Nz], Val(OP))  # Fu[i,j,Nz+1] == 0
                end
            end
        end  # if nw == ...
    else  # backward averaging
        if nw == 1  # w = x
            # 1. At the locations except for the negative end of the x-direction; unlike for
            # the forward difference, for the backward difference this part of the code is
            # common for both the Bloch and symmetry boundary conditions.
            @threads for k = 1:Nz
                for j = 1:Ny, i = 2:Nx  # not i = 2:Nx-1
                    @inbounds set_or_add!(Gv, (i,j,k), α2 * (Fu[i,j,k] + Fu[i-1,j,k]), Val(OP))
                end
            end

            # 2. At the negative end of the x-direction (where the boundary fields are taken
            # from the positive-end boundary for the Bloch boundary condition)
            if isbloch
                for k = 1:Nz, j = 1:Ny
                    @inbounds set_or_add!(Gv, (1,j,k), α2 * (Fu[1,j,k] + Fu[Nx,j,k]/e⁻ⁱᵏᴸ), Val(OP))  # Fu[0,j,k] = Fu[Nx,j,k] / exp(-i kx Lx)
                end
            else  # symmetry boundary
                for k = 1:Nz, j = 1:Ny
                    @inbounds set_or_add!(Gv, (1,j,k), α * Fu[1,j,k], Val(OP))  # Fu[0,j,k] = Fu[1,j,k]; use α instead of α2 as field is doubled
                end
            end
        elseif nw == 2  # w = y
            # 1. At the locations except for the negative end of the y-direction; unlike for
            # the forward difference, for the backward difference this part of the code is
            # common for both the Bloch and symmetry boundary conditions.
            @threads for k = 1:Nz
                for j = 2:Ny, i = 1:Nx  # not j = 2:Ny-1
                    @inbounds set_or_add!(Gv, (i,j,k), α2 * (Fu[i,j,k] + Fu[i,j-1,k]), Val(OP))
                end
            end

            # 2. At the negative end of the y-direction (where the boundary fields are taken
            # from the positive-end boundary for the Bloch boundary condition)
            if isbloch
                for k = 1:Nz, i = 1:Nx
                    @inbounds set_or_add!(Gv, (i,1,k), α2 * (Fu[i,1,k] + Fu[i,Ny,k]/e⁻ⁱᵏᴸ), Val(OP))  # Fu[i,0,k] = Fu[0,Ny,k] / exp(-i ky Ly)
                end
            else  # symmetry boundary
                for k = 1:Nz, i = 1:Nx
                    @inbounds set_or_add!(Gv, (i,1,k), α * Fu[i,1,k], Val(OP))  # Fu[i,0,k] = Fu[i,1,k]; use α instead of α2 as field is doubled
                end
            end
        else  # nw == 3; w = z
            # 1. At the locations except for the negative end of the z-direction; unlike for
            # the forward difference, for the backward difference this part of the code is
            # common for both the Bloch and symmetry boundary conditions.
            @threads for k = 2:Nz
                for j = 1:Ny, i = 1:Nx
                    @inbounds set_or_add!(Gv, (i,j,k), α2 * (Fu[i,j,k] + Fu[i,j,k-1]), Val(OP))
                end
            end

            # 2. At the negative end of the z-direction (where the boundary fields are taken
            # from the positive-end boundary for the Bloch boundary condition)
            if isbloch
                for j = 1:Ny, i = 1:Nx
                    @inbounds set_or_add!(Gv, (i,j,1), α2 * (Fu[i,j,1] + Fu[i,j,Nz]/e⁻ⁱᵏᴸ), Val(OP))  # Fu[i,j,0] = Fu[i,j,Nz] / exp(-i kz Lz)
                end
            else  # symmetry boundary
                for j = 1:Ny, i = 1:Nx
                    @inbounds set_or_add!(Gv, (i,j,1), α * Fu[i,j,1], Val(OP))  # Fu[i,j,0] = Fu[i,j,Nz]; use α instead of α2 as field is doubled
                end
            end
        end  # if nw == ...
    end  # if isfwd

    return nothing
end

# Arithmetic m̂ for 2D
function apply_m̂!(Gv::AbsArrNumber{2},  # v-component of output field (v = x, y)
                  Fu::AbsArrNumber{2},  # u-component of input field (u = x, y)
                  ::Val{OP},  # Val(:(=)) or Val(:(+=)): set (=) or add (+=) operator to use
                  nw::Integer,  # 1|2 for averaging along x|y
                  isfwd::Bool,  # true|false for forward|backward averaging
                  isbloch::Bool,  # boundary condition in w-direction
                  e⁻ⁱᵏᴸ::Number=1.0;  # Bloch phase factor
                  α::Number=1.0  # scale factor to multiply to result before adding it to Gv: Gv += α m(Fu)
                  ) where {OP}
    @assert size(Gv)==size(Fu)

    Nx, Ny = size(Fu)
    α2 = 0.5 * α

    # Make sure not to include branches inside for loops.
    if isfwd
        if nw == 1  # w = x
            if isbloch
                # 1. At locations except for the positive end of the x-direction
                @threads for j = 1:Ny
                    for i = 1:Nx-1
                        @inbounds set_or_add!(Gv, (i,j), α2 * (Fu[i+1,j] + Fu[i,j]), Val(OP))
                    end
                end

                # 2. At the positive end of the x-direction (where the boundary fields are
                # taken from the negative-end boundary)
                for j = 1:Ny
                    @inbounds set_or_add!(Gv, (Nx,j), α2 * (e⁻ⁱᵏᴸ*Fu[1,j] + Fu[Nx,j]), Val(OP))  # Fu[Nx+1,j] = exp(-i kx Lx) * Fu[1,j]
                end
            else  # symmetry boundary
                # 1. At the locations except for the positive and negative ends of the
                # x-direction
                @threads for j = 1:Ny
                    for i = 2:Nx-1
                        @inbounds set_or_add!(Gv, (i,j), α2 * (Fu[i+1,j] + Fu[i,j]), Val(OP))
                    end
                end

                # 2. At the negative end of the x-direction (where the boundary fields are
                # assumed zero)
                for j = 1:Ny
                    @inbounds set_or_add!(Gv, (1,j), α2 * Fu[2,j], Val(OP))  # Fu[1,j] == 0
                end

                # 3. At the positive end of the x-direction (where the boundary fields are
                # assumed zero)
                for j = 1:Ny
                    @inbounds set_or_add!(Gv, (Nx,j), α2 * Fu[Nx,j], Val(OP))  # Fu[Nx+1,j] == 0
                end
            end
        else  # nw == 2; w = y
            if isbloch
                # 1. At locations except for the positive end of the y-direction
                @threads for j = 1:Ny-1
                    for i = 1:Nx
                        @inbounds set_or_add!(Gv, (i,j), α2 * (Fu[i,j+1] + Fu[i,j]), Val(OP))
                    end
                end

                # 2. At the positive end of the y-direction (where the boundary fields are
                # taken from the negative-end boundary)
                for i = 1:Nx
                    @inbounds set_or_add!(Gv, (i,Ny), α2 * (e⁻ⁱᵏᴸ*Fu[i,1] + Fu[i,Ny]), Val(OP))  # Fu[i,Ny+1] = exp(-i ky Ly) * Fu[i,1]
                end
            else  # symmetry boundary
                # 1. At the locations except for the positive and negative ends of the
                # y-direction
                @threads for j = 2:Ny-1
                    for i = 1:Nx
                        @inbounds set_or_add!(Gv, (i,j), α2 * (Fu[i,j+1] + Fu[i,j]), Val(OP))
                    end
                end

                # 2. At the negative end of the y-direction (where the boundary fields are
                # assumed zero)
                for i = 1:Nx
                    @inbounds set_or_add!(Gv, (i,1), α2 * Fu[i,2], Val(OP))  # Fu[i,1] == 0
                end

                # 3. At the positive end of the y-direction (where the boundary fields are
                # assumed zero)
                for i = 1:Nx
                    @inbounds set_or_add!(Gv, (i,Ny), α2 * Fu[i,Ny], Val(OP))  # Fu[i,Ny+1] == 0
                end
            end
        end  # if nw == ...
    else  # backward averaging
        if nw == 1  # w = x
            # 1. At the locations except for the negative end of the x-direction; unlike for
            # the forward difference, for the backward difference this part of the code is
            # common for both the Bloch and symmetry boundary conditions.
            @threads for j = 1:Ny
                for i = 2:Nx  # not i = 2:Nx-1
                    @inbounds set_or_add!(Gv, (i,j), α2 * (Fu[i,j] + Fu[i-1,j]), Val(OP))
                end
            end

            # 2. At the negative end of the x-direction (where the boundary fields are taken
            # from the positive-end boundary for the Bloch boundary condition)
            if isbloch
                for j = 1:Ny
                    @inbounds set_or_add!(Gv, (1,j), α2 * (Fu[1,j] + Fu[Nx,j]/e⁻ⁱᵏᴸ), Val(OP))  # Fu[0,j] = Fu[Nx,j] / exp(-i kx Lx)
                end
            else  # symmetry boundary
                for j = 1:Ny
                    @inbounds set_or_add!(Gv, (1,j), α * Fu[1,j], Val(OP))  # Fu[0,j] = Fu[1,j]; use α instead of α2 as field is doubled
                end
            end
        elseif nw == 2  # w = y
            # 1. At the locations except for the negative end of the y-direction; unlike for
            # the forward difference, for the backward difference this part of the code is
            # common for both the Bloch and symmetry boundary conditions.
            @threads for j = 2:Ny
                for i = 1:Nx
                    @inbounds set_or_add!(Gv, (i,j), α2 * (Fu[i,j] + Fu[i,j-1]), Val(OP))
                end
            end

            # 2. At the negative end of the y-direction (where the boundary fields are taken
            # from the positive-end boundary for the Bloch boundary condition)
            if isbloch
                for i = 1:Nx
                    @inbounds set_or_add!(Gv, (i,1), α2 * (Fu[i,1] + Fu[i,Ny]/e⁻ⁱᵏᴸ), Val(OP))  # Fu[i,0] = Fu[0,Ny] / exp(-i ky Ly)
                end
            else  # symmetry boundary
                for i = 1:Nx
                    @inbounds set_or_add!(Gv, (i,1), α * Fu[i,1], Val(OP))  # Fu[i,0] = Fu[i,1]; use α instead of α2 as field is doubled
                end
            end
        end  # if nw == ...
    end  # if isfwd

    return nothing
end

# Arithmetic m̂ for 1D
function apply_m̂!(Gv::AbsArrNumber{1},  # v-component of output field (v = x, y)
                  Fu::AbsArrNumber{1},  # u-component of input field (u = x, y)
                  ::Val{OP},  # Val(:(=)) or Val(:(+=)): set (=) or add (+=) operator to use
                  nw::Integer,  # 1|2 for averaging along x|y
                  isfwd::Bool,  # true|false for forward|backward averaging
                  isbloch::Bool,  # boundary condition in w-direction
                  e⁻ⁱᵏᴸ::Number=1.0;  # Bloch phase factor
                  α::Number=1.0  # scale factor to multiply to result before adding it to Gv: Gv += α m(Fu)
                  ) where {OP}
    @assert size(Gv)==size(Fu)

    Nx = length(Fu)  # not size(Fu) unlike code for 2D and 3D
    α2 = 0.5 * α

    # Make sure not to include branches inside for loops.
    if isfwd
        if isbloch
            # 1. At locations except for the positive end of the x-direction
            @threads for i = 1:Nx-1
                @inbounds set_or_add!(Gv, (i,), α2 * (Fu[i+1] + Fu[i]), Val(OP))
            end

            # 2. At the positive end of the x-direction (where the boundary fields are taken
            # from the negative-end boundary)
            @inbounds set_or_add!(Gv, (Nx,), α2 * (e⁻ⁱᵏᴸ*Fu[1] + Fu[Nx]), Val(OP))  # Fu[Nx+1] = exp(-i kx Lx) * Fu[1]
        else  # symmetry boundary
            # 1. At the locations except for the positive and negative ends of the x-direction
            @threads for i = 2:Nx-1
                @inbounds set_or_add!(Gv, (i,), α2 * (Fu[i+1] + Fu[i]), Val(OP))
            end

            # 2. At the negative end of the x-direction (where the boundary fields are assumed
            # zero)
            @inbounds set_or_add!(Gv, (1,), α2 * Fu[2], Val(OP))  # Fu[1] == 0

            # 3. At the positive end of the x-direction (where the boundary fields are assumed
            # zero)
            @inbounds set_or_add!(Gv, (Nx,), α2 * Fu[Nx], Val(OP))  # Fu[Nx+1] == 0
        end
    else  # backward averaging
        # 1. At the locations except for the negative end of the x-direction; unlike for the
        # forward difference, for the backward difference this part of the code is common
        # for both the Bloch and symmetry boundary conditions.
        @threads for i = 2:Nx
            @inbounds set_or_add!(Gv, (i,), α2 * (Fu[i] + Fu[i-1]), Val(OP))
        end

        # 2. At the negative end of the x-direction (where the boundary fields are taken
        # from the positive-end boundary for the Bloch boundary condition)
        if isbloch
            @inbounds set_or_add!(Gv, (1,), α2 * (Fu[1] + Fu[Nx]/e⁻ⁱᵏᴸ), Val(OP))  # Fu[0] = Fu[Nx] / exp(-i kx Lx)
        else  # symmetry boundary
            @inbounds set_or_add!(Gv, (1,), α * Fu[1], Val(OP))  # Fu[0] = Fu[1]; use α instead of α2 as field is doubled
        end
    end  # if isfwd

    return nothing
end


# Weighted m̂ for 3D
function apply_m̂!(Gv::AbsArrNumber{3},  # v-component of output field (v = x, y, z)
                  Fu::AbsArrNumber{3},  # u-component of input field (u = x, y, z)
                  ::Val{OP},  # Val(:(=)) or Val(:(+=)): set (=) or add (+=) operator to use
                  nw::Integer,  # 1|2|3 for averaging along x|y|z
                  isfwd::Bool,  # true|false for forward|backward averaging
                  ∆w::AbsVecNumber,  # line segments to multiply with; vector of length N[nw]
                  ∆w′⁻¹::AbsVecNumber,  # inverse of line segments to divide by; vector of length N[nw]
                  isbloch::Bool,  # boundary condition in w-direction
                  e⁻ⁱᵏᴸ::Number=1.0;  # Bloch phase factor
                  α::Number=1.0  # scale factor to multiply to result before adding it to Gv: Gv += α m(Fu)
                  ) where {OP}
    @assert size(Gv)==size(Fu)
    @assert size(Fu,nw)==length(∆w)
    @assert length(∆w)==length(∆w′⁻¹)

    Nx, Ny, Nz = size(Fu)
    α2 = 0.5 * α

    # Make sure not to include branches inside for loops.
    if isfwd
        if nw == 1  # w = x
            if isbloch
                # 1. At locations except for the positive end of the x-direction
                @threads for k = 1:Nz
                    for j = 1:Ny, i = 1:Nx-1
                        @inbounds set_or_add!(Gv, (i,j,k), (α2 * ∆w′⁻¹[i]) * (∆w[i+1]*Fu[i+1,j,k] + ∆w[i]*Fu[i,j,k]), Val(OP))
                    end
                end

                # 2. At the positive end of the x-direction (where the boundary fields are
                # taken from the negative-end boundary)
                β = α2 * ∆w′⁻¹[Nx]
                for k = 1:Nz, j = 1:Ny
                    @inbounds set_or_add!(Gv, (Nx,j,k), β * (∆w[1]*e⁻ⁱᵏᴸ*Fu[1,j,k] + ∆w[Nx]*Fu[Nx,j,k]), Val(OP))  # Fu[Nx+1,j,k] = exp(-i kx Lx) * Fu[1,j,k]
                end
            else  # symmetry boundary
                # 1. At the locations except for the positive and negative ends of the
                # x-direction
                @threads for k = 1:Nz
                    for j = 1:Ny, i = 2:Nx-1
                        @inbounds set_or_add!(Gv, (i,j,k), (α2 * ∆w′⁻¹[i]) * (∆w[i+1]*Fu[i+1,j,k] + ∆w[i]*Fu[i,j,k]), Val(OP))
                    end
                end

                # 2. At the negative end of the x-direction (where the boundary fields are
                # assumed zero)
                β = α2 * ∆w′⁻¹[1]
                for k = 1:Nz, j = 1:Ny
                    @inbounds set_or_add!(Gv, (1,j,k), β * ∆w[2]*Fu[2,j,k], Val(OP))  # Fu[1,j,k] == 0
                end

                # 3. At the positive end of the x-direction (where the boundary fields are
                # assumed zero)
                β = α2 * ∆w′⁻¹[Nx]
                for k = 1:Nz, j = 1:Ny
                    @inbounds set_or_add!(Gv, (Nx,j,k), β * ∆w[Nx]*Fu[Nx,j,k], Val(OP))  # Fu[Nx+1,j,k] == 0
                end
            end
        elseif nw == 2  # w = y
            if isbloch
                # 1. At locations except for the positive end of the y-direction
                @threads for k = 1:Nz
                    for j = 1:Ny-1, i = 1:Nx
                        @inbounds set_or_add!(Gv, (i,j,k), (α2 * ∆w′⁻¹[j]) * (∆w[j+1]*Fu[i,j+1,k] + ∆w[j]*Fu[i,j,k]), Val(OP))
                    end
                end

                # 2. At the positive end of the y-direction (where the boundary fields are
                # taken from the negative-end boundary)
                β = α2 * ∆w′⁻¹[Ny]
                for k = 1:Nz, i = 1:Nx
                    @inbounds set_or_add!(Gv, (i,Ny,k), β * (∆w[1]*e⁻ⁱᵏᴸ*Fu[i,1,k] + ∆w[Ny]*Fu[i,Ny,k]), Val(OP))  # Fu[i,Ny+1,k] = exp(-i ky Ly) * Fu[i,1,k]
                end
            else  # symmetry boundary
                # 1. At the locations except for the positive and negative ends of the
                # y-direction
                @threads for k = 1:Nz
                    for j = 2:Ny-1, i = 1:Nx
                        @inbounds set_or_add!(Gv, (i,j,k), (α2 * ∆w′⁻¹[j]) * (∆w[j+1]*Fu[i,j+1,k] + ∆w[j]*Fu[i,j,k]), Val(OP))
                    end
                end

                # 2. At the negative end of the y-direction (where the boundary fields are
                # assumed zero)
                β = α2 * ∆w′⁻¹[1]
                for k = 1:Nz, i = 1:Nx
                    @inbounds set_or_add!(Gv, (i,1,k), β * ∆w[2]*Fu[i,2,k], Val(OP))  # Fu[i,1,k] == 0
                end

                # 3. At the positive end of the y-direction (where the boundary fields are
                # assumed zero)
                β = α2 * ∆w′⁻¹[Ny]
                for k = 1:Nz, i = 1:Nx
                    @inbounds set_or_add!(Gv, (i,Ny,k), β * ∆w[Ny]*Fu[i,Ny,k], Val(OP))  # Fu[i,Ny+1,k] == 0
                end
            end
        else  # nw == 3; w = z
            if isbloch
                # 1. At locations except for the positive end of the z-direction
                @threads for k = 1:Nz-1
                    for j = 1:Ny, i = 1:Nx
                        @inbounds set_or_add!(Gv, (i,j,k), (α2 * ∆w′⁻¹[k]) * (∆w[k+1]*Fu[i,j,k+1] + ∆w[k]*Fu[i,j,k]), Val(OP))
                    end
                end

                # 2. At the positive end of the z-direction (where the boundary fields are
                # taken from the negative-end boundary)
                β = α2 * ∆w′⁻¹[Nz]
                for j = 1:Ny, i = 1:Nx
                    @inbounds set_or_add!(Gv, (i,j,Nz), β * (∆w[1]*e⁻ⁱᵏᴸ*Fu[i,j,1] + ∆w[Nz]*Fu[i,j,Nz]), Val(OP))  # Fu[i,j,Nz+1] = exp(-i kz Lz) * Fu[i,j,1]
                end
            else  # symmetry boundary
                # 1. At the locations except for the positive and negative ends of the
                # z-direction
                @threads for k = 2:Nz-1
                    for j = 1:Ny, i = 1:Nx
                        @inbounds set_or_add!(Gv, (i,j,k), (α2 * ∆w′⁻¹[k]) * (∆w[k+1]*Fu[i,j,k+1] + ∆w[k]*Fu[i,j,k]), Val(OP))
                    end
                end

                # 2. At the negative end of the z-direction (where the boundary fields are
                # assumed zero)
                β = α2 * ∆w′⁻¹[1]
                for j = 1:Ny, i = 1:Nx
                    @inbounds set_or_add!(Gv, (i,j,1), β * ∆w[2]*Fu[i,j,2], Val(OP))  # Fu[i,j,1] == 0
                end

                # 3. At the positive end of the z-direction (where the boundary fields are
                # assumed zero)
                β = α2 * ∆w′⁻¹[Nz]
                for j = 1:Ny, i = 1:Nx
                    @inbounds set_or_add!(Gv, (i,j,Nz), β * ∆w[Nz]*Fu[i,j,Nz], Val(OP))  # Fu[i,j,Nz+1] == 0
                end
            end
        end  # if nw == ...
    else  # backward averaging
        if nw == 1  # w = x
            # 1. At the locations except for the negative end of the x-direction; unlike for
            # the forward difference, for the backward difference this part of the code is
            # common for both the Bloch and symmetry boundary conditions.
            @threads for k = 1:Nz
                for j = 1:Ny, i = 2:Nx  # not i = 2:Nx-1
                    @inbounds set_or_add!(Gv, (i,j,k), (α2 * ∆w′⁻¹[i]) * (∆w[i]*Fu[i,j,k] + ∆w[i-1]*Fu[i-1,j,k]), Val(OP))
                end
            end

            # 2. At the negative end of the x-direction (where the boundary fields are taken
            # from the positive-end boundary for the Bloch boundary condition)
            if isbloch
                β = α2 * ∆w′⁻¹[1]
                for k = 1:Nz, j = 1:Ny
                    @inbounds set_or_add!(Gv, (1,j,k), β * (∆w[1]*Fu[1,j,k] + ∆w[Nx]*Fu[Nx,j,k]/e⁻ⁱᵏᴸ), Val(OP))  # Fu[0,j,k] = Fu[Nx,j,k] / exp(-i kx Lx)
                end
            else  # symmetry boundary
                β = α * ∆w′⁻¹[1]
                for k = 1:Nz, j = 1:Ny
                    @inbounds set_or_add!(Gv, (1,j,k), β * ∆w[1]*Fu[1,j,k], Val(OP))  # Fu[0,j,k] = Fu[1,j,k]
                end
            end
        elseif nw == 2  # w = y
            # 1. At the locations except for the negative end of the y-direction; unlike for
            # the forward difference, for the backward difference this part of the code is
            # common for both the Bloch and symmetry boundary conditions.
            @threads for k = 1:Nz
                for j = 2:Ny, i = 1:Nx  # not j = 2:Ny-1
                    @inbounds set_or_add!(Gv, (i,j,k), (α2 * ∆w′⁻¹[j]) * (∆w[j]*Fu[i,j,k] + ∆w[j-1]*Fu[i,j-1,k]), Val(OP))
                end
            end

            # 2. At the negative end of the y-direction (where the boundary fields are taken
            # from the positive-end boundary for the Bloch boundary condition)
            if isbloch
                β = α2 * ∆w′⁻¹[1]
                for k = 1:Nz, i = 1:Nx
                    @inbounds set_or_add!(Gv, (i,1,k), β * (∆w[1]*Fu[i,1,k] + ∆w[Ny]*Fu[i,Ny,k]/e⁻ⁱᵏᴸ), Val(OP))  # Fu[i,0,k] = Fu[0,Ny,k] / exp(-i ky Ly)
                end
            else  # symmetry boundary
                β = α * ∆w′⁻¹[1]
                for k = 1:Nz, i = 1:Nx
                    @inbounds set_or_add!(Gv, (i,1,k), β * ∆w[1]*Fu[i,1,k], Val(OP))  # Fu[i,0,k] = Fu[i,1,k]
                end
            end
        else  # nw == 3; w = z
            # 1. At the locations except for the negative end of the z-direction; unlike for
            # the forward difference, for the backward difference this part of the code is
            # common for both the Bloch and symmetry boundary conditions.
            @threads for k = 2:Nz
                for j = 1:Ny, i = 1:Nx
                    @inbounds set_or_add!(Gv, (i,j,k), (α2 * ∆w′⁻¹[k]) * (∆w[k]*Fu[i,j,k] + ∆w[k-1]*Fu[i,j,k-1]), Val(OP))
                end
            end

            # 2. At the negative end of the z-direction (where the boundary fields are taken
            # from the positive-end boundary for the Bloch boundary condition)
            if isbloch
                β = α2 * ∆w′⁻¹[1]
                for j = 1:Ny, i = 1:Nx
                    @inbounds set_or_add!(Gv, (i,j,1), β * (∆w[1]*Fu[i,j,1] + ∆w[Nz]*Fu[i,j,Nz]/e⁻ⁱᵏᴸ), Val(OP))  # Fu[i,j,0] = Fu[i,j,Nz] / exp(-i kz Lz)
                end
            else  # symmetry boundary
                β = α * ∆w′⁻¹[1]
                for j = 1:Ny, i = 1:Nx
                    @inbounds set_or_add!(Gv, (i,j,1), β * ∆w[1]*Fu[i,j,1], Val(OP))  # Fu[i,j,0] = Fu[i,j,Nz]
                end
            end
        end  # if nw == ...
    end  # if isfwd

    return nothing
end

# Weighted m̂ for 2D
function apply_m̂!(Gv::AbsArrNumber{2},  # v-component of output field (v = x, y)
                  Fu::AbsArrNumber{2},  # u-component of input field (u = x, y)
                  ::Val{OP},  # Val(:(=)) or Val(:(+=)): set (=) or add (+=) operator to use
                  nw::Integer,  # 1|2 for averaging along x|y
                  isfwd::Bool,  # true|false for forward|backward averaging
                  ∆w::AbsVecNumber,  # line segments to multiply with; vector of length N[nw]
                  ∆w′⁻¹::AbsVecNumber,  # inverse of line segments to divide by; vector of length N[nw]
                  isbloch::Bool,  # boundary condition in w-direction
                  e⁻ⁱᵏᴸ::Number=1.0;  # Bloch phase factor
                  α::Number=1.0  # scale factor to multiply to result before adding it to Gv: Gv += α m(Fu)
                  ) where {OP}
    @assert size(Gv)==size(Fu)
    @assert size(Fu,nw)==length(∆w)
    @assert length(∆w)==length(∆w′⁻¹)

    Nx, Ny = size(Fu)
    α2 = 0.5 * α

    # Make sure not to include branches inside for loops.
    if isfwd
        if nw == 1  # w = x
            if isbloch
                # 1. At locations except for the positive end of the x-direction
                @threads for j = 1:Ny
                    for i = 1:Nx-1
                        @inbounds set_or_add!(Gv, (i,j), (α2 * ∆w′⁻¹[i]) * (∆w[i+1]*Fu[i+1,j] + ∆w[i]*Fu[i,j]), Val(OP))
                    end
                end

                # 2. At the positive end of the x-direction (where the boundary fields are
                # taken from the negative-end boundary)
                β = α2 * ∆w′⁻¹[Nx]
                for j = 1:Ny
                    @inbounds set_or_add!(Gv, (Nx,j), β * (∆w[1]*e⁻ⁱᵏᴸ*Fu[1,j] + ∆w[Nx]*Fu[Nx,j]), Val(OP))  # Fu[Nx+1,j] = exp(-i kx Lx) * Fu[1,j]
                end
            else  # symmetry boundary
                # 1. At the locations except for the positive and negative ends of the
                # x-direction
                @threads for j = 1:Ny
                    for i = 2:Nx-1
                        @inbounds set_or_add!(Gv, (i,j), (α2 * ∆w′⁻¹[i]) * (∆w[i+1]*Fu[i+1,j] + ∆w[i]*Fu[i,j]), Val(OP))
                    end
                end

                # 2. At the negative end of the x-direction (where the boundary fields are
                # assumed zero)
                β = α2 * ∆w′⁻¹[1]
                for j = 1:Ny
                    @inbounds set_or_add!(Gv, (1,j), β * ∆w[2]*Fu[2,j], Val(OP))  # Fu[1,j] == 0
                end

                # 3. At the positive end of the x-direction (where the boundary fields are
                # assumed zero)
                β = α2 * ∆w′⁻¹[Nx]
                for j = 1:Ny
                    @inbounds set_or_add!(Gv, (Nx,j), β * ∆w[Nx]*Fu[Nx,j], Val(OP))  # Fu[Nx+1,j] == 0
                end
            end
        else  # nw == 2; w = y
            if isbloch
                # 1. At locations except for the positive end of the y-direction
                @threads for j = 1:Ny-1
                    for i = 1:Nx
                        @inbounds set_or_add!(Gv, (i,j), (α2 * ∆w′⁻¹[j]) * (∆w[j+1]*Fu[i,j+1] + ∆w[j]*Fu[i,j]), Val(OP))
                    end
                end

                # 2. At the positive end of the y-direction (where the boundary fields are
                # taken from the negative-end boundary)
                β = α2 * ∆w′⁻¹[Ny]
                for i = 1:Nx
                    @inbounds set_or_add!(Gv, (i,Ny), β * (∆w[1]*e⁻ⁱᵏᴸ*Fu[i,1] + ∆w[Ny]*Fu[i,Ny]), Val(OP))  # Fu[i,Ny+1] = exp(-i ky Ly) * Fu[i,1]
                end
            else  # symmetry boundary
                # 1. At the locations except for the positive and negative ends of the
                # y-direction
                @threads for j = 2:Ny-1
                    for i = 1:Nx
                        @inbounds set_or_add!(Gv, (i,j), (α2 * ∆w′⁻¹[j]) * (∆w[j+1]*Fu[i,j+1] + ∆w[j]*Fu[i,j]), Val(OP))
                    end
                end

                # 2. At the negative end of the y-direction (where the boundary fields are
                # assumed zero)
                β = α2 * ∆w′⁻¹[1]
                for i = 1:Nx
                    @inbounds set_or_add!(Gv, (i,1), β * ∆w[2]*Fu[i,2], Val(OP))  # Fu[i,1] == 0
                end

                # 3. At the positive end of the y-direction (where the boundary fields are
                # assumed zero)
                β = α2 * ∆w′⁻¹[Ny]
                for i = 1:Nx
                    @inbounds set_or_add!(Gv, (i,Ny), β * ∆w[Ny]*Fu[i,Ny], Val(OP))  # Fu[i,Ny+1] == 0
                end
            end
        end  # if nw == ...
    else  # backward averaging
        if nw == 1  # w = x
            # 1. At the locations except for the negative end of the x-direction; unlike for
            # the forward difference, for the backward difference this part of the code is
            # common for both the Bloch and symmetry boundary conditions.
            @threads for j = 1:Ny
                for i = 2:Nx  # not i = 2:Nx-1
                    @inbounds set_or_add!(Gv, (i,j), (α2 * ∆w′⁻¹[i]) * (∆w[i]*Fu[i,j] + ∆w[i-1]*Fu[i-1,j]), Val(OP))
                end
            end

            # 2. At the negative end of the x-direction (where the boundary fields are taken
            # from the positive-end boundary for the Bloch boundary condition)
            if isbloch
                β = α2 * ∆w′⁻¹[1]
                for j = 1:Ny
                    @inbounds set_or_add!(Gv, (1,j), β * (∆w[1]*Fu[1,j] + ∆w[Nx]*Fu[Nx,j]/e⁻ⁱᵏᴸ), Val(OP))  # Fu[0,j] = Fu[Nx,j] / exp(-i kx Lx)
                end
            else  # symmetry boundary
                β = α * ∆w′⁻¹[1]
                for j = 1:Ny
                    @inbounds set_or_add!(Gv, (1,j), β * ∆w[1]*Fu[1,j], Val(OP))  # Fu[0,j] = Fu[1,j]
                end
            end
        elseif nw == 2  # w = y
            # 1. At the locations except for the negative end of the y-direction; unlike for
            # the forward difference, for the backward difference this part of the code is
            # common for both the Bloch and symmetry boundary conditions.
            @threads for j = 2:Ny
                for i = 1:Nx
                    @inbounds set_or_add!(Gv, (i,j), (α2 * ∆w′⁻¹[j]) * (∆w[j]*Fu[i,j] + ∆w[j-1]*Fu[i,j-1]), Val(OP))
                end
            end

            # 2. At the negative end of the y-direction (where the boundary fields are taken
            # from the positive-end boundary for the Bloch boundary condition)
            if isbloch
                β = α2 * ∆w′⁻¹[1]
                for i = 1:Nx
                    @inbounds set_or_add!(Gv, (i,1), β * (∆w[1]*Fu[i,1] + ∆w[Ny]*Fu[i,Ny]/e⁻ⁱᵏᴸ), Val(OP))  # Fu[i,0] = Fu[0,Ny] / exp(-i ky Ly)
                end
            else  # symmetry boundary
                β = α * ∆w′⁻¹[1]
                for i = 1:Nx
                    @inbounds set_or_add!(Gv, (i,1), β * ∆w[1]*Fu[i,1], Val(OP))  # Fu[i,0] = Fu[i,1]
                end
            end
        end  # if nw == ...
    end  # if isfwd

    return nothing
end

# Weighted m̂ for 1D
function apply_m̂!(Gv::AbsArrNumber{1},  # v-component of output field (v = x, y)
                  Fu::AbsArrNumber{1},  # u-component of input field (u = x, y)
                  ::Val{OP},  # Val(:(=)) or Val(:(+=)): set (=) or add (+=) operator to use
                  nw::Integer,  # 1|2 for averaging along x|y
                  isfwd::Bool,  # true|false for forward|backward averaging
                  ∆w::AbsVecNumber,  # line segments to multiply with; vector of length N[nw]
                  ∆w′⁻¹::AbsVecNumber,  # inverse of line segments to divide by; vector of length N[nw]
                  isbloch::Bool,  # boundary condition in w-direction
                  e⁻ⁱᵏᴸ::Number=1.0;  # Bloch phase factor
                  α::Number=1.0  # scale factor to multiply to result before adding it to Gv: Gv += α m(Fu)
                  ) where {OP}
    @assert size(Gv)==size(Fu)
    @assert size(Fu,nw)==length(∆w)
    @assert length(∆w)==length(∆w′⁻¹)

    Nx = length(Fu)  # not size(Fu) unlike code for 2D and 3D
    α2 = 0.5 * α

    # Make sure not to include branches inside for loops.
    if isfwd
        if isbloch
            # 1. At locations except for the positive end of the x-direction
            @threads for i = 1:Nx-1
                @inbounds set_or_add!(Gv, (i,), (α2 * ∆w′⁻¹[i]) * (∆w[i+1]*Fu[i+1] + ∆w[i]*Fu[i]), Val(OP))
            end

            # 2. At the positive end of the x-direction (where the boundary fields are taken
            # from the negative-end boundary)
            β = α2 * ∆w′⁻¹[Nx]
            @inbounds set_or_add!(Gv, (Nx,), β * (∆w[1]*e⁻ⁱᵏᴸ*Fu[1] + ∆w[Nx]*Fu[Nx]), Val(OP))  # Fu[Nx+1] = exp(-i kx Lx) * Fu[1]
        else  # symmetry boundary
            # 1. At the locations except for the positive and negative ends of the x-direction
            @threads for i = 2:Nx-1
                @inbounds set_or_add!(Gv, (i,), (α2 * ∆w′⁻¹[i]) * (∆w[i+1]*Fu[i+1] + ∆w[i]*Fu[i]), Val(OP))
            end

            # 2. At the negative end of the x-direction (where the boundary fields are assumed
            # zero)
            β = α2 * ∆w′⁻¹[1]
            @inbounds set_or_add!(Gv, (1,), β * ∆w[2]*Fu[2], Val(OP))  # Fu[1] == 0

            # 3. At the positive end of the x-direction (where the boundary fields are assumed
            # zero)
            β = α2 * ∆w′⁻¹[Nx]
            @inbounds set_or_add!(Gv, (Nx,), β * ∆w[Nx]*Fu[Nx], Val(OP))  # Fu[Nx+1] == 0
        end
    else  # backward averaging
        # 1. At the locations except for the negative end of the x-direction; unlike for the
        # forward difference, for the backward difference this part of the code is common
        # for both the Bloch and symmetry boundary conditions.
        @threads for i = 2:Nx
            @inbounds set_or_add!(Gv, (i,), (α2 * ∆w′⁻¹[i]) * (∆w[i]*Fu[i] + ∆w[i-1]*Fu[i-1]), Val(OP))
        end

        # 2. At the negative end of the x-direction (where the boundary fields are taken
        # from the positive-end boundary for the Bloch boundary condition)
        if isbloch
            β = α2 * ∆w′⁻¹[1]
            @inbounds set_or_add!(Gv, (1,), β * (∆w[1]*Fu[1] + ∆w[Nx]*Fu[Nx]/e⁻ⁱᵏᴸ), Val(OP))  # Fu[0] = Fu[Nx] / exp(-i kx Lx)
        else  # symmetry boundary
            β = α * ∆w′⁻¹[1]
            @inbounds set_or_add!(Gv, (1,), β * ∆w[1]*Fu[1], Val(OP))  # Fu[0] = Fu[1]
        end
    end  # if isfwd

    return nothing
end
