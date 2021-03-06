# The functions add the calculated values to the existing values of the output array.
# Therefore, if the derivative values themselves are desired, pass the output array
# initialized with zeros.

export apply_∂!

# Inside @spawn, avoid defining β, which requires another let block to avoid unnecessary
# allocations.

# Wrapper to apply discrete partial derivative by default
apply_∂!(Gv::AbsArrNumber,  # v-component of output field (v = x, y, z in 3D)
         Fu::AbsArrNumber,  # u-component of input field (u = x, y, z in 3D)
         ::Val{OP},  # Val(:(=)) or Val(:(+=)): set (=) or add (+=) operator to use
         nw::Integer,  # 1|2|3 for x|y|z in 3D
         isfwd::Bool,  # true|false for forward|backward difference
         ∆w⁻¹::Number=1.0,  # inverse of spatial discretization
         isbloch::Bool=true,  # boundary condition in w-direction
         e⁻ⁱᵏᴸ::Number=1.0;  # Bloch phase factor
         α::Number=1.0  # scale factor to multiply to result before adding it to Gv: Gv += α ∂Fu/∂w
         ) where {OP} =
    (N = size(Gv); apply_∂!(Gv, Fu, Val(OP), nw, isfwd, fill(∆w⁻¹, N[nw]), isbloch, e⁻ⁱᵏᴸ, α=α))  # fill: create vector of ∆w⁻¹

# Unlike matrix/partial.jl, there is no need to create a wrapper to convert AbstractVector's
# to SVec's, because the concrete implementation of apply_∂! below does not take any
# SVec's.  I only need to set the default value for e⁻ⁱᵏᴸ below, even though I don't
# usually set default values for the arguments in concrete implementation (except for
# keyword arguments, which require default values).

# The field arrays Fu (and Gv) represents a K-D array of a specific Cartesian component of the
# field, and indexed as Fu[i,j,k], where (i,j,k) is the grid cell location.

# Concrete apply_∂! for 3D
function apply_∂!(Gv::AbsArrNumber{3},  # v-component of output field (v = x, y, z)
                  Fu::AbsArrNumber{3},  # u-component of input field (u = x, y, z)
                  ::Val{OP},  # Val(:(=)) or Val(:(+=)): set (=) or add (+=) operator to use
                  nw::Integer,  # 1|2|3 for x|y|z
                  isfwd::Bool,  # true|false for forward|backward difference
                  ∆w⁻¹::AbsVecNumber,  # inverse of spatial discretization; vector of length N[nw]
                  isbloch::Bool,  # boundary condition in w-direction
                  e⁻ⁱᵏᴸ::Number=1.0;  # Bloch phase factor: L = Lw
                  α::Number=1.0  # scale factor to multiply to result before adding it to Gv: Gv += α ∂Fu/∂w
                  ) where {OP}
    @assert size(Gv)==size(Fu)
    @assert 1≤nw≤3
    @assert size(Fu,nw)==length(∆w⁻¹)

    Nx, Ny, Nz = size(Fu)

    # Make sure not to include branches inside for loops.
    if isfwd
        if nw == 1
            if isbloch
                # 1. At locations except for the positive end of the x-direction
                @threads for k = 1:Nz
                    for j = 1:Ny, i = 1:Nx-1
                        @inbounds set_or_add!(Gv, (i,j,k), (α * ∆w⁻¹[i]) * (Fu[i+1,j,k] - Fu[i,j,k]), Val(OP))
                    end
                end

                # 2. At the positive end of the x-direction (where the boundary fields are
                # taken from the negative-end boundary)
                β = α * ∆w⁻¹[Nx]
                for k = 1:Nz, j = 1:Ny
                    @inbounds set_or_add!(Gv, (Nx,j,k), β * (e⁻ⁱᵏᴸ*Fu[1,j,k] - Fu[Nx,j,k]), Val(OP))  # Fu[Nx+1,j,k] = exp(-i kx Lx) * Fu[1,j,k]
                end
            else  # symmetry boundary
                # 1. At the locations except for the positive and negative ends of the
                # x-direction
                @threads for k = 1:Nz
                    for j = 1:Ny, i = 2:Nx-1
                        @inbounds set_or_add!(Gv, (i,j,k), (α * ∆w⁻¹[i]) * (Fu[i+1,j,k] - Fu[i,j,k]), Val(OP))
                    end
                end

                # 2. At the negative end of the x-direction (where the boundary fields are
                # assumed zero)
                β = α * ∆w⁻¹[1]
                for k = 1:Nz, j = 1:Ny
                    @inbounds set_or_add!(Gv, (1,j,k), β * Fu[2,j,k], Val(OP))  # Fu[1,j,k] == 0
                end

                # 3. At the positive end of the x-direction (where the boundary fields are
                # assumed zero)
                β = -α * ∆w⁻¹[Nx]
                for k = 1:Nz, j = 1:Ny
                    @inbounds set_or_add!(Gv, (Nx,j,k), β * Fu[Nx,j,k], Val(OP))  # Fu[Nx+1,j,k] == 0
                end
            end
        elseif nw == 2
            if isbloch
                # 1. At locations except for the positive end of the y-direction
                @threads for k = 1:Nz
                    for j = 1:Ny-1, i = 1:Nx
                        @inbounds set_or_add!(Gv, (i,j,k), (α * ∆w⁻¹[j]) * (Fu[i,j+1,k] - Fu[i,j,k]), Val(OP))
                    end
                end

                # 2. At the positive end of the y-direction (where the boundary fields are
                # taken from the negative-end boundary)
                β = α * ∆w⁻¹[Ny]
                for k = 1:Nz, i = 1:Nx
                    @inbounds set_or_add!(Gv, (i,Ny,k), β * (e⁻ⁱᵏᴸ*Fu[i,1,k] - Fu[i,Ny,k]), Val(OP))  # Fu[i,Ny+1,k] = exp(-i ky Ly) * Fu[i,1,k]
                end
            else  # symmetry boundary
                # 1. At the locations except for the positive and negative ends of the
                # y-direction
                @threads for k = 1:Nz
                    for j = 2:Ny-1, i = 1:Nx
                        @inbounds set_or_add!(Gv, (i,j,k), (α * ∆w⁻¹[j]) * (Fu[i,j+1,k] - Fu[i,j,k]), Val(OP))
                    end
                end

                # 2. At the negative end of the y-direction (where the boundary fields are
                # assumed zero)
                β = α * ∆w⁻¹[1]
                for k = 1:Nz, i = 1:Nx
                    @inbounds set_or_add!(Gv, (i,1,k), β * Fu[i,2,k], Val(OP))  # Fu[i,1,k] == 0
                end

                # 3. At the positive end of the y-direction (where the boundary fields are
                # assumed zero)
                β = -α * ∆w⁻¹[Ny]
                for k = 1:Nz, i = 1:Nx
                    @inbounds set_or_add!(Gv, (i,Ny,k), β * Fu[i,Ny,k], Val(OP))  # Fu[i,Ny+1,k] == 0
                end
            end
        else  # nw == 3
            if isbloch
                # 1. At locations except for the positive end of the z-direction
                @threads for k = 1:Nz-1
                    for j = 1:Ny, i = 1:Nx
                        @inbounds set_or_add!(Gv, (i,j,k), (α * ∆w⁻¹[k]) * (Fu[i,j,k+1] - Fu[i,j,k]), Val(OP))
                    end
                end

                # 2. At the positive end of the z-direction (where the boundary fields are
                # taken from the negative-end boundary)
                β = α * ∆w⁻¹[Nz]
                for j = 1:Ny, i = 1:Nx
                    @inbounds set_or_add!(Gv, (i,j,Nz), β * (e⁻ⁱᵏᴸ*Fu[i,j,1] - Fu[i,j,Nz]), Val(OP))  # Fu[i,j,Nz+1] = exp(-i kz Lz) * Fu[i,j,1]
                end
            else  # symmetry boundary
                # 1. At the locations except for the positive and negative ends of the
                # z-direction
                @threads for k = 2:Nz-1
                    for j = 1:Ny, i = 1:Nx
                        @inbounds set_or_add!(Gv, (i,j,k), (α * ∆w⁻¹[k]) * (Fu[i,j,k+1] - Fu[i,j,k]), Val(OP))
                    end
                end

                # 2. At the negative end of the z-direction (where the boundary fields are
                # assumed zero)
                β = α * ∆w⁻¹[1]
                for j = 1:Ny, i = 1:Nx
                    @inbounds set_or_add!(Gv, (i,j,1), β * Fu[i,j,2], Val(OP))  # Fu[i,j,1] == 0
                end

                # 3. At the positive end of the z-direction (where the boundary fields are
                # assumed zero)
                β = -α * ∆w⁻¹[Nz]
                for j = 1:Ny, i = 1:Nx
                    @inbounds set_or_add!(Gv, (i,j,Nz), β * Fu[i,j,Nz], Val(OP))  # Fu[i,j,Nz+1] == 0
                end
            end
        end  # if nw == ...
    else  # backward difference
        if nw == 1
            # 1. At the locations except for the negative end of the x-direction; unlike for
            # the forward difference, for the backward difference this part of the code is
            # common for both the Bloch and symmetry boundary conditions.
            @threads for k = 1:Nz
                for j = 1:Ny, i = 2:Nx  # not i = 2:Nx-1
                    @inbounds set_or_add!(Gv, (i,j,k), (α * ∆w⁻¹[i]) * (Fu[i,j,k] - Fu[i-1,j,k]), Val(OP))
                end
            end

            # 2. At the negative end of the x-direction (where the boundary fields are taken
            # from the positive-end boundary for the Bloch boundary condition)
            if isbloch
                β = α * ∆w⁻¹[1]
                for k = 1:Nz, j = 1:Ny
                    @inbounds set_or_add!(Gv, (1,j,k), β * (Fu[1,j,k] - Fu[Nx,j,k]/e⁻ⁱᵏᴸ), Val(OP))  # Fu[0,j,k] = Fu[Nx,j,k] / exp(-i kx Lx)
                end
            else  # symmetry boundary
                for k = 1:Nz, j = 1:Ny
                    @inbounds set_or_add!(Gv, (1,j,k), 0, Val(OP))  # derivatives of dual fields at symmetry boundary are zero
                end
            end
        elseif nw == 2
            # 1. At the locations except for the negative end of the y-direction; unlike for
            # the forward difference, for the backward difference this part of the code is
            # common for both the Bloch and symmetry boundary conditions.
            @threads for k = 1:Nz
                for j = 2:Ny, i = 1:Nx  # not j = 2:Ny-1
                    @inbounds set_or_add!(Gv, (i,j,k), (α * ∆w⁻¹[j]) * (Fu[i,j,k] - Fu[i,j-1,k]), Val(OP))
                end
            end

            # 2. At the negative end of the y-direction (where the boundary fields are taken
            # from the positive-end boundary for the Bloch boundary condition)
            if isbloch
                β = α * ∆w⁻¹[1]
                for k = 1:Nz, i = 1:Nx
                    @inbounds set_or_add!(Gv, (i,1,k), β * (Fu[i,1,k] - Fu[i,Ny,k]/e⁻ⁱᵏᴸ), Val(OP))  # Fu[i,0,k] = Fu[i,Ny,k] / exp(-i ky Ly)
                end
            else  # symmetry boundary
                for k = 1:Nz, i = 1:Nx
                    @inbounds set_or_add!(Gv, (i,1,k), 0, Val(OP))  # derivatives of dual fields at symmetry boundary are zero
                end
            end
        else  # nw == 3
            # 1. At the locations except for the negative end of the z-direction; unlike for
            # the forward difference, for the backward difference this part of the code is
            # common for both the Bloch and symmetry boundary conditions.
            @threads for k = 2:Nz
                for j = 1:Ny, i = 1:Nx
                    @inbounds set_or_add!(Gv, (i,j,k), (α * ∆w⁻¹[k]) * (Fu[i,j,k] - Fu[i,j,k-1]), Val(OP))
                end
            end

            # 2. At the negative end of the z-direction (where the boundary fields are taken
            # from the positive-end boundary for the Bloch boundary condition)
            if isbloch
                β = α * ∆w⁻¹[1]
                for j = 1:Ny, i = 1:Nx
                    @inbounds set_or_add!(Gv, (i,j,1), β * (Fu[i,j,1] - Fu[i,j,Nz]/e⁻ⁱᵏᴸ), Val(OP))  # Fu[i,j,0] = Fu[i,j,Nz] / exp(-i kz Lz)
                end
            else  # symmetry boundary
                for j = 1:Ny, i = 1:Nx
                    @inbounds set_or_add!(Gv, (i,j,1), 0, Val(OP))  # derivatives of dual fields at symmetry boundary are zero
                end
            end
        end  # if nw == ...
    end  # if isfwd

    return nothing
end


# Concrete apply_∂! for 2D
function apply_∂!(Gv::AbsArrNumber{2},  # v-component of output field (v = x, y)
                  Fu::AbsArrNumber{2},  # u-component of input field (u = x, y)
                  ::Val{OP},  # Val(:(=)) or Val(:(+=)): set (=) or add (+=) operator to use
                  nw::Integer,  # 1|2 for x|y
                  isfwd::Bool,  # true|false for forward|backward difference
                  ∆w⁻¹::AbsVecNumber,  # inverse of spatial discretization; vector of length N[nw]
                  isbloch::Bool,  # boundary condition in w-direction
                  e⁻ⁱᵏᴸ::Number=1.0;  # Bloch phase factor: L = Lw
                  α::Number=1.0  # scale factor to multiply to result before adding it to Gv: Gv += α ∂Fu/∂w
                  ) where {OP}
    @assert size(Gv)==size(Fu)
    @assert 1≤nw≤2
    @assert size(Fu,nw)==length(∆w⁻¹)

    Nx, Ny = size(Fu)

    # Make sure not to include branches inside for loops.
    if isfwd
        if nw == 1
            if isbloch
                # 1. At locations except for the positive end of the x-direction
                @threads for j = 1:Ny
                    for i = 1:Nx-1
                        @inbounds set_or_add!(Gv, (i,j), (α * ∆w⁻¹[i]) * (Fu[i+1,j] - Fu[i,j]), Val(OP))
                    end
                end

                # 2. At the positive end of the x-direction (where the boundary fields are
                # taken from the negative-end boundary)
                β = α * ∆w⁻¹[Nx]
                for j = 1:Ny
                    @inbounds set_or_add!(Gv, (Nx,j), β * (e⁻ⁱᵏᴸ*Fu[1,j] - Fu[Nx,j]), Val(OP))  # Fu[Nx+1,j] = exp(-i kx Lx) * Fu[1,j]
                end
            else  # symmetry boundary
                # 1. At the locations except for the positive and negative ends of the
                # x-direction
                @threads for j = 1:Ny
                    for i = 2:Nx-1
                        @inbounds set_or_add!(Gv, (i,j), (α * ∆w⁻¹[i]) * (Fu[i+1,j] - Fu[i,j]), Val(OP))
                    end
                end

                # 2. At the negative end of the x-direction (where the boundary fields are
                # assumed zero)
                β = α * ∆w⁻¹[1]
                for j = 1:Ny
                    @inbounds set_or_add!(Gv, (1,j), β * Fu[2,j], Val(OP))  # Fu[1,j] == 0
                end

                # 3. At the positive end of the x-direction (where the boundary fields are
                # assumed zero)
                β = -α * ∆w⁻¹[Nx]
                for j = 1:Ny
                    @inbounds set_or_add!(Gv, (Nx,j), β * Fu[Nx,j], Val(OP))  # Fu[Nx+1,j] == 0
                end
            end
        else  # nw == 2
            if isbloch
                # 1. At locations except for the positive end of the y-direction
                @threads for j =1:Ny-1
                    for i = 1:Nx
                        @inbounds set_or_add!(Gv, (i,j), (α * ∆w⁻¹[j]) * (Fu[i,j+1] - Fu[i,j]), Val(OP))
                    end
                end

                # 2. At the positive end of the y-direction (where the boundary fields are
                # taken from the negative-end boundary)
                β = α * ∆w⁻¹[Ny]
                for i = 1:Nx
                    @inbounds set_or_add!(Gv, (i,Ny), β * (e⁻ⁱᵏᴸ*Fu[i,1] - Fu[i,Ny]), Val(OP))  # Fu[i,Ny+1] = exp(-i ky Ly) * Fu[i,1]
                end
            else  # symmetry boundary
                # 1. At the locations except for the positive and negative ends of the
                # y-direction
                @threads for j = 2:Ny-1
                    for i = 1:Nx
                        @inbounds set_or_add!(Gv, (i,j), (α * ∆w⁻¹[j]) * (Fu[i,j+1] - Fu[i,j]), Val(OP))
                    end
                end

                # 2. At the negative end of the y-direction (where the boundary fields are
                # assumed zero)
                β = α * ∆w⁻¹[1]
                for i = 1:Nx
                    @inbounds set_or_add!(Gv, (i,1), β * Fu[i,2], Val(OP))  # Fu[i,1] == 0
                end

                # 3. At the positive end of the y-direction (where the boundary fields are
                # assumed zero)
                β = -α * ∆w⁻¹[Ny]
                for i = 1:Nx
                    @inbounds set_or_add!(Gv, (i,Ny), β * Fu[i,Ny], Val(OP))  # Fu[i,Ny+1] == 0
                end
            end
        end  # if nw == ...
    else  # backward difference
        if nw == 1
            # 1. At the locations except for the negative end of the x-direction; unlike for
            # the forward difference, for the backward difference this part of the code is
            # common for both the Bloch and symmetry boundary conditions.
            @threads for j = 1:Ny
                for i = 2:Nx  # not i = 2:Nx-1
                    @inbounds set_or_add!(Gv, (i,j), (α * ∆w⁻¹[i]) * (Fu[i,j] - Fu[i-1,j]), Val(OP))
                end
            end

            # 2. At the negative end of the x-direction (where the boundary fields are taken
            # from the positive-end boundary)
            if isbloch
                β = α * ∆w⁻¹[1]
                for j = 1:Ny
                    @inbounds set_or_add!(Gv, (1,j), β * (Fu[1,j] - Fu[Nx,j]/e⁻ⁱᵏᴸ), Val(OP))  # Fu[0,j] = Fu[Nx,j] / exp(-i kx Lx)
                end
            else  # symmetry boundary
                for j = 1:Ny
                    @inbounds set_or_add!(Gv, (1,j), 0, Val(OP))  # derivatives of dual fields at symmetry boundary are zero
                end
            end
        else  # nw == 2
            # 1. At the locations except for the negative end of the y-direction; unlike for
            # the forward difference, for the backward difference this part of the code is
            # common for both the Bloch and symmetry boundary conditions.
            @threads for j = 2:Ny
                for i = 1:Nx
                    @inbounds set_or_add!(Gv, (i,j), (α * ∆w⁻¹[j]) * (Fu[i,j] - Fu[i,j-1]), Val(OP))
                end
            end

            # 2. At the negative end of the y-direction (where the boundary fields are taken
            # from the positive-end boundary for the Bloch boundary condition)
            if isbloch
                β = α * ∆w⁻¹[1]
                for i = 1:Nx
                    @inbounds set_or_add!(Gv, (i,1), β * (Fu[i,1] - Fu[i,Ny]/e⁻ⁱᵏᴸ), Val(OP))  # Fu[i,0] = Fu[i,Ny] / exp(-i ky Ly)
                end
            else  # symmetry boundary
                for i = 1:Nx
                    @inbounds set_or_add!(Gv, (i,1), 0, Val(OP))  # derivatives of dual fields at symmetry boundary are zero
                end
            end
        end  # if nw == ...
    end  # if isfwd

    return nothing
end

# Concrete apply_∂! for 1D
function apply_∂!(Gv::AbsArrNumber{1},  # v-component of output field (v = x)
                  Fu::AbsArrNumber{1},  # u-component of input field (u = x)
                  ::Val{OP},  # Val(:(=)) or Val(:(+=)): set (=) or add (+=) operator to use
                  nw::Integer,  # 1 for x
                  isfwd::Bool,  # true|false for forward|backward difference
                  ∆w⁻¹::AbsVecNumber,  # inverse of spatial discretization; vector of length N[nw]
                  isbloch::Bool,  # boundary condition in w-direction
                  e⁻ⁱᵏᴸ::Number=1.0;  # Bloch phase factor: L = Lw
                  α::Number=1.0  # scale factor to multiply to result before adding it to Gv: Gv += α ∂Fu/∂w
                  ) where {OP}
    @assert size(Gv)==size(Fu)
    @assert nw==1
    @assert size(Fu,nw)==length(∆w⁻¹)

    Nx = length(Fu)  # not size(Fu) unlike code for 2D and 3D

    # Make sure not to include branches inside for loops.
    if isfwd
        if isbloch
            # 1. At locations except for the positive end of the x-direction
            @threads for i = 1:Nx-1
                @inbounds set_or_add!(Gv, (i,), (α * ∆w⁻¹[i]) * (Fu[i+1] - Fu[i]), Val(OP))
            end

            # 2. At the positive end of the x-direction (where the boundary fields are taken
            # from the negative-end boundary)
            β = α * ∆w⁻¹[Nx]
            set_or_add!(Gv, (Nx,), β * (e⁻ⁱᵏᴸ*Fu[1] - Fu[Nx]), Val(OP))  # Fu[Nx+1] = exp(-i kx Lx) * Fu[1]
        else  # symmetry boundary
            # 1. At the locations except for the positive and negative ends of the x-direction
            @threads for i = 2:Nx-1
                @inbounds set_or_add!(Gv, (i,), (α * ∆w⁻¹[i]) * (Fu[i+1] - Fu[i]), Val(OP))
            end

            # 2. At the negative end of the x-direction (where the boundary fields are assumed
            # zero)
            β = α * ∆w⁻¹[1]
            set_or_add!(Gv, (1,), β * Fu[2], Val(OP))  # Fu[1] == 0

            # 3. At the positive end of the y-direction (where the boundary fields are assumed
            # zero)
            β = -α * ∆w⁻¹[Nx]
            set_or_add!(Gv, (Nx,), β * Fu[Nx], Val(OP))  # Fu[Nx+1] == 0
        end
    else  # backward difference
        # 1. At the locations except for the negative end of the x-direction; unlike for the
        # forward difference, for the backward difference this part of the code is common
        # for both the Bloch and symmetry boundary conditions.
        @threads for i = 2:Nx
            @inbounds set_or_add!(Gv, (i,), (α * ∆w⁻¹[i]) * (Fu[i] - Fu[i-1]), Val(OP))
        end

        # 2. At the negative end of the x-direction (where the boundary fields are taken
        # from the positive-end boundary for the Bloch boundary condition)
        if isbloch
            β = α * ∆w⁻¹[1]
            set_or_add!(Gv, (1,), β * (Fu[1] - Fu[Nx]/e⁻ⁱᵏᴸ), Val(OP))  # Fu[0] = Fu[Nx] / exp(-i kx Lx)
        else  # symmetry boundary
            set_or_add!(Gv, (1,), 0, Val(OP))  # derivatives of dual fields at symmetry boundary are zero
        end
    end  # if isfwd

    return nothing
end

# # The following is the implementation for arbitrary space dimensions, but it is slow.
# function apply_∂!(Gv::AbsArrNumber{K},  # v-component of output field (v = x, y, z in 3D)
#                   Fu::AbsArrNumber{K},  # u-component of input field (u = x, y, z in 3D)
#                   nw::Integer,  # 1|2|3 for x|y|z in 3D
#                   isfwd::Bool,  # true|false for forward|backward difference
#                   ∆w⁻¹::AbsVecNumber,  # spatial discretization; vector of length N[nw]
#                   isbloch::Bool,  # boundary condition in w-direction
#                   e⁻ⁱᵏᴸ::Number=1.0;  # Bloch phase factor: L = Lw
#                   α::Number=1.0  # scale factor to multiply to result before adding it to Gv: Gv += α ∂Fu/∂w
#                   ) where {K}  # space dimension (field dimension Kf does not show up as we deal with single component)
#     @assert size(Gv)==size(Fu)
#     @assert 1≤nw≤K
#     @assert size(Fu,nw)==length(∆w⁻¹)
#
#     # Make sure not to include branches inside for loops.
#     ciₛ₀ = (ntuple(k->1, Val(K)))  # start indices; (1,1,1) in 3D
#     ciₑ₀ = (size(Fu))  # end indices; (Nx,Ny,Nz) in 3D
#     δci = (ntuple(k->k==nw, Val(K)))  # 1 at w-component; 0 elsewhere
#     ∆ci = (ciₑ₀.I .* δci.I) - δci  # Nw-1 at w-component; 0 elsewhere
#     if isfwd  # forward difference
#         if isbloch  # Bloch boundary condition
#             # At locations except for the positive end of the w-direction
#             ciₛ = ciₛ₀
#             ciₑ = ciₑ₀ - δci  # w-component of end indices is decreased from Nw to Nw-1
#             CI = CartesianIndices(map(:, ciₛ.I, ciₑ.I))
#             for ci = CI
#                 @inbounds Gv[ci] += (α * ∆w⁻¹[ci[nw]]) * (Fu[ci+δci] - Fu[ci])
#             end
#
#             # At the positive end of the w-direction (where the boundary fields are taken
#             # from the negative-end boundary)
#             ciₛ = ciₛ₀ + ∆ci  # Nw at w-component; 1 elsewhere
#             ciₑ = ciₑ₀  # (Nx,Ny,Nz) in 3D
#             CI = CartesianIndices(map(:, ciₛ.I, ciₑ.I))
#             β = α * ∆w⁻¹[ciₑ[nw]]
#             for ci = CI
#                 @inbounds Gv[ci] += β * (e⁻ⁱᵏᴸ*Fu[ci-∆ci] - Fu[ci])  # for w = x, Fu[Nx+1,j,k] = exp(-i kx Lx) * Fu[1,j,k]
#             end
#         else  # symmetry boundary condition
#             # At the locations except for the positive and negative ends of the w-direction
#             ciₛ = ciₛ₀ + δci  # w-component of start indices is increased from 1 to 2
#             ciₑ = ciₑ₀ - δci  # w-component of end indices is decreased from Nw to Nw-1
#             CI = CartesianIndices(map(:, ciₛ.I, ciₑ.I))
#             for ci = CI
#                 @inbounds Gv[ci] += (α * ∆w⁻¹[ci[nw]]) * (Fu[ci+δci] - Fu[ci])
#             end
#
#             # At the negative end of the w-direction (where the boundary fields are assumed zero)
#             ciₛ = ciₛ₀  # (1,1,1) in 3D
#             ciₑ = ciₑ₀ - ∆ci  # w-component of end indices is decreased from Nw to 1
#             β = α * ∆w⁻¹[1]
#             CI = CartesianIndices(map(:, ciₛ.I, ciₑ.I))
#             for ci = CI
#                 @inbounds Gv[ci] += β * Fu[ci+δci]  # note Fu[ci] = 0
#             end
#
#             # At the positive end of the w-direction (where the boundary fields are assumed zero)
#             ciₛ = ciₛ₀ + ∆ci  # w-component of end indices is increased from 1 to Nw
#             ciₑ = ciₑ₀  # (Nx,Ny,Nz) in 3D
#             β = α * ∆w⁻¹[ciₑ[nw]]
#             CI = CartesianIndices(map(:, ciₛ.I, ciₑ.I))
#             for ci = CI
#                 @inbounds Gv[ci] -= β * Fu[ci]  # Fu[ci+δci] == 0
#             end
#         end  # if isbloch
#     else  # backward difference
#         # At the locations except for the negative end of the w-direction; unlike for the
#         # forward difference, for the backward difference this part of the code is common
#         # for both the Bloch and symmetry boundary conditions.
#         ciₛ = ciₛ₀ + δci  # w-component of start indices is increased from 1 to 2
#         ciₑ = ciₑ₀
#         CI = CartesianIndices(map(:, ciₛ.I, ciₑ.I))
#         for ci = CI
#             @inbounds Gv[ci] += (α * ∆w⁻¹[ci[nw]]) * (Fu[ci] - Fu[ci-δci])
#         end
#
#         if isbloch  # Bloch boundary condition
#             # At the negative end of the w-direction (where the boundary fields are taken
#             # from the positive-end boundary)
#             ciₛ = ciₛ₀  # (1,1,1) in 3D
#             ciₑ = ciₑ₀ - ∆ci  #  1 at w-component; N elsewhere
#             CI = CartesianIndices(map(:, ciₛ.I, ciₑ.I))
#             β = α * ∆w⁻¹[1]
#             for ci = CI
#                 @inbounds Gv[ci] += β * (Fu[ci] - Fu[ci+∆ci]/e⁻ⁱᵏᴸ)  # for w = x, Fu[0,j,k] = Fu[Nx,j,k] / exp(-i kx Lx)
#             end
#         else  # symmetry boundary condition
#             # Do nothing, because the derivative of dual fields at the symmetry boundary is
#             # zero.
#         end  # if isbloch
#     end  # if isfwd
#
#     return nothing
# end
