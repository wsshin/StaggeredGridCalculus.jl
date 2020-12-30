# Average fields along the field directions (i.e., Fw along the w-direction).  See the
# description of matrix/mean.jl/create_m().  Technically, apply_m!() can be used to average
# fields normal to the direction of averaging, but it is not used that way in apply_mean!().

# Assumes the space dimension and field dimension are the same.  In other words, when the
# space coordinate indices are (i,j,k), then the field has three vector components.
# Therefore, for the input field array F[i,j,k,w], we assume w = 1:3.

# The functions calculate the average and add to the output array, instead of replacing the
# values stored in the output array.  Therefore, if the derivative values themselves are
# desired, pass the output array initialized with zeros.

# Inside @spawn, avoid defining β, which requires another let block to avoid unnecessary
# allocations.

export apply_m!, apply_mean!

# Wrapper for arithmetic averaging
apply_mean!(G::AbsArrNumber,  # output field; G[i,j,k,w] is w-component of G at (i,j,k) in 3D
            F::AbsArrNumber,  # input field; F[i,j,k,w] is w-component of G at (i,j,k)
            ::Val{OP},  # Val(:(=)) or Val(:(+=)): set (=) or add (+=) operator to use
            isfwd::AbsVecBool,  # isfwd[w] = true|false for forward|backward averaging
            isbloch::AbsVecBool=fill(true,length(isfwd)),  # boundary conditions in x, y, z
            e⁻ⁱᵏᴸ::AbsVecNumber=ones(length(isfwd));  # Bloch phase factor in x, y, z
            α::Number=1.0  # scale factor to multiply to result before adding it to G: G += α ∇×F
            ) where {OP} =
    (N = size(F)[1:end-1]; ∆l = ones.((N...,)); ∆l⁻¹ = ones.((N...,)); apply_mean!(G, F, Val(OP), isfwd, ∆l, ∆l⁻¹, isbloch, e⁻ⁱᵏᴸ, α=α))

# Wrapper for converting AbstractVector's to SVector's
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
    (K = length(isfwd); apply_mean!(G, F, Val(OP), SVector{K}(isfwd), ∆l, ∆l′⁻¹, SVector{K}(isbloch), SVector{K}(e⁻ⁱᵏᴸ), α=α))

# Concrete implementation
# For the implementation details, see the comments in matrix/mean.jl.
function apply_mean!(G::AbsArrNumber{K₊₁},  # output field; G[i,j,k,w] is w-component of G at (i,j,k)
                     F::AbsArrNumber{K₊₁},  # input field; G[i,j,k,w] is w-component of G at (i,j,k)
                     ::Val{OP},  # Val(:(=)) or Val(:(+=)): set (=) or add (+=) operator to use
                     isfwd::SBool{K},  # isfwd[w] = true|false for forward|backward averaging
                     ∆l::NTuple{K,AbsVecNumber},  # line segments to multiply with; vectors of length N
                     ∆l′⁻¹::NTuple{K,AbsVecNumber},  # inverse of line segments to divide by; vectors of length N
                     isbloch::SBool{K},  # boundary conditions in x, y, z
                     e⁻ⁱᵏᴸ::SNumber{K};  # Bloch phase factor in x, y, z
                     α::Number=1  # scale factor to multiply to result before adding it to G: G += α mean(F)
                     ) where {K,K₊₁,OP}  #  K is space dimension; K₊₁ = K + 1
    @assert(K₊₁==K+1)
    n_bounds = calc_boundary_indices(size(G)[1:K])
    for nw = 1:K  # direction of averaging
        nv = nw
        Gv = selectdim(G, K₊₁, nv)  # nv-th component of output field

        nu = nw  # component of input field to feed to w-directional averaging
        Fu = selectdim(F, K₊₁, nu)  # nu-th component of input field

        apply_m!(Gv, Fu, Val(OP), nv, isfwd[nv], ∆l[nv], ∆l′⁻¹[nv], isbloch[nv], e⁻ⁱᵏᴸ[nv], n_bounds=n_bounds, α=α)
    end
end

## Field-averaging operators "m" (as used in de Moerloose and de Zutter)
#
# This applies the averaging operator for a single Cartesian component.  For the operator
# for all three Cartesian components, use apply_mean!.
apply_m!(Gv::AbsArrNumber,  # v-component of output field (v = x, y, z in 3D)
         Fu::AbsArrNumber,  # u-component of input field (u = x, y, z in 3D)
         ::Val{OP},  # Val(:(=)) or Val(:(+=)): set (=) or add (+=) operator to use
         nw::Integer,  # 1|2|3 for averaging along x|y|z in 3D
         isfwd::Bool,  # true|false for forward|backward averaging
         ∆w::Number,  # line segments to multiply with; vector of length N[nw]
         isbloch::Bool=true,  # boundary condition in w-direction
         e⁻ⁱᵏᴸ::Number=1;  # Bloch phase factor
         n_bounds::Tuple2{AbsVecInteger}=calc_boundary_indices(size(Gv)),  # (nₛ,nₑ): start and end indices of chunks in last dimension to be processed in parallel
         α::Number=1  # scale factor to multiply to result before adding it to Gv: Gv += α m(Fu)
         ) where {OP} =
    (N = size(Fu); ∆w_vec = fill(∆w, N[nw]); ∆w′⁻¹_vec = fill(1/∆w, N[nw]); apply_m!(Gv, Fu, Val(OP), nw, isfwd, ∆w_vec, ∆w′⁻¹_vec, isbloch, e⁻ⁱᵏᴸ, n_bounds=n_bounds, α=α))  # fill: create vector of ∆w

apply_m!(Gv::AbsArrNumber,  # v-component of output field (v = x, y, z in 3D)
         Fu::AbsArrNumber,  # u-component of input field (u = x, y, z in 3D)
         ::Val{OP},  # Val(:(=)) or Val(:(+=)): set (=) or add (+=) operator to use
         nw::Integer,  # 1|2|3 for averaging along x|y|z
         isfwd::Bool,  # true|false for forward|backward averaging
         ∆w::AbsVecNumber=ones(size(Fu)[nw]),  # line segments to multiply with; vector of length N[nw]
         ∆w′⁻¹::AbsVecNumber=ones(size(Fu)[nw]),  # inverse of line segments to divide by; vector of length N[nw]
         isbloch::Bool=true,  # boundary condition in w-direction
         e⁻ⁱᵏᴸ::Number=1;  # Bloch phase factor
         n_bounds::Tuple2{AbsVecInteger}=calc_boundary_indices(size(Gv)),  # (nₛ,nₑ): start and end indices of chunks in last dimension to be processed in parallel
         α::Number=1  # scale factor to multiply to result before adding it to Gv: Gv += α m(Fu)
         ) where {OP} =
    (N = size(Fu); ∆w_vec = fill(∆w, N[nw]); ∆w′⁻¹_vec = fill(1/∆w, N[nw]); apply_m!(Gv, Fu, Val(OP), nw, isfwd, ∆w_vec, ∆w′⁻¹_vec, isbloch, e⁻ⁱᵏᴸ, n_bounds=n_bounds, α=α))  # fill: create vector of ∆w

# Concrete apply_m! for 3D
function apply_m!(Gv::AbsArrNumber{3},  # v-component of output field (v = x, y, z)
                  Fu::AbsArrNumber{3},  # u-component of input field (u = x, y, z)
                  ::Val{OP},  # Val(:(=)) or Val(:(+=)): set (=) or add (+=) operator to use
                  nw::Integer,  # 1|2|3 for averaging along x|y|z
                  isfwd::Bool,  # true|false for forward|backward averaging
                  ∆w::AbsVecNumber,  # line segments to multiply with; vector of length N[nw]
                  ∆w′⁻¹::AbsVecNumber,  # inverse of line segments to divide by; vector of length N[nw]
                  isbloch::Bool,  # boundary condition in w-direction
                  e⁻ⁱᵏᴸ::Number;  # Bloch phase factor
                  n_bounds::Tuple2{AbsVecInteger}=calc_boundary_indices(size(Gv)),  # (nₛ,nₑ): start and end indices of chunks in last dimension to be processed in parallel
                  α::Number=1  # scale factor to multiply to result before adding it to Gv: Gv += α m(Fu)
                  ) where {OP}
    @assert(size(Gv)==size(Fu))
    @assert(size(Fu,nw)==length(∆w))
    @assert(length(∆w)==length(∆w′⁻¹))

    Nx, Ny, Nz = size(Fu)
    kₛ, kₑ = n_bounds
    Nₜ = length(kₛ)

    α2 = 0.5 * α

    # Make sure not to include branches inside for loops.
    @sync if isfwd
        if nw == 1  # w = x
            if isbloch
                # 1. At locations except for the positive end of the x-direction
                for t = 1:Nₜ
                    kₛₜ, kₑₜ = kₛ[t], kₑ[t]
                    let kₛₜ=kₛₜ, kₑₜ=kₑₜ
                        @spawn for k = kₛₜ:kₑₜ
                            for j = 1:Ny, i = 1:Nx-1
                                @inbounds set_or_add!(Gv, i, j, k, (α2 * ∆w′⁻¹[i]) * (∆w[i+1]*Fu[i+1,j,k] + ∆w[i]*Fu[i,j,k]), Val(OP))
                            end
                        end
                    end
                end

                # 2. At the positive end of the x-direction (where the boundary fields are
                # taken from the negative-end boundary)
                β = α2 * ∆w′⁻¹[Nx]
                for k = 1:Nz, j = 1:Ny
                    @inbounds set_or_add!(Gv, Nx, j, k, β * (∆w[1]*e⁻ⁱᵏᴸ*Fu[1,j,k] + ∆w[Nx]*Fu[Nx,j,k]), Val(OP))  # Fu[Nx+1,j,k] = exp(-i kx Lx) * Fu[1,j,k]
                end
            else  # symmetry boundary
                # 1. At the locations except for the positive and negative ends of the
                # x-direction
                for t = 1:Nₜ
                    kₛₜ, kₑₜ = kₛ[t], kₑ[t]
                    let kₛₜ=kₛₜ, kₑₜ=kₑₜ
                        @spawn for k = kₛₜ:kₑₜ
                            for j = 1:Ny, i = 2:Nx-1
                                @inbounds set_or_add!(Gv, i, j, k, (α2 * ∆w′⁻¹[i]) * (∆w[i+1]*Fu[i+1,j,k] + ∆w[i]*Fu[i,j,k]), Val(OP))
                            end
                        end
                    end
                end

                # 2. At the negative end of the x-direction (where the boundary fields are
                # assumed zero)
                β = α2 * ∆w′⁻¹[1]
                for k = 1:Nz, j = 1:Ny
                    @inbounds set_or_add!(Gv, 1, j, k, β * ∆w[2]*Fu[2,j,k], Val(OP))  # Fu[1,j,k] == 0
                end

                # 3. At the positive end of the x-direction (where the boundary fields are
                # assumed zero)
                β = α2 * ∆w′⁻¹[Nx]
                for k = 1:Nz, j = 1:Ny
                    @inbounds set_or_add!(Gv, Nx, j, k, β * ∆w[Nx]*Fu[Nx,j,k], Val(OP))  # Fu[Nx+1,j,k] == 0
                end
            end
        elseif nw == 2  # w = y
            if isbloch
                # 1. At locations except for the positive end of the y-direction
                for t = 1:Nₜ
                    kₛₜ, kₑₜ = kₛ[t], kₑ[t]
                    let kₛₜ=kₛₜ, kₑₜ=kₑₜ
                        @spawn for k = kₛₜ:kₑₜ
                            for j = 1:Ny-1, i = 1:Nx
                                @inbounds set_or_add!(Gv, i, j, k, (α2 * ∆w′⁻¹[j]) * (∆w[j+1]*Fu[i,j+1,k] + ∆w[j]*Fu[i,j,k]), Val(OP))
                            end
                        end
                    end
                end

                # 2. At the positive end of the y-direction (where the boundary fields are
                # taken from the negative-end boundary)
                β = α2 * ∆w′⁻¹[Ny]
                for k = 1:Nz, i = 1:Nx
                    @inbounds set_or_add!(Gv, i, Ny, k, β * (∆w[1]*e⁻ⁱᵏᴸ*Fu[i,1,k] + ∆w[Ny]*Fu[i,Ny,k]), Val(OP))  # Fu[i,Ny+1,k] = exp(-i ky Ly) * Fu[i,1,k]
                end
            else  # symmetry boundary
                # 1. At the locations except for the positive and negative ends of the
                # y-direction
                for t = 1:Nₜ
                    kₛₜ, kₑₜ = kₛ[t], kₑ[t]
                    let kₛₜ=kₛₜ, kₑₜ=kₑₜ
                        @spawn for k = kₛₜ:kₑₜ
                            for j = 2:Ny-1, i = 1:Nx
                                @inbounds set_or_add!(Gv, i, j, k, (α2 * ∆w′⁻¹[j]) * (∆w[j+1]*Fu[i,j+1,k] + ∆w[j]*Fu[i,j,k]), Val(OP))
                            end
                        end
                    end
                end

                # 2. At the negative end of the y-direction (where the boundary fields are
                # assumed zero)
                β = α2 * ∆w′⁻¹[1]
                for k = 1:Nz, i = 1:Nx
                    @inbounds set_or_add!(Gv, i, 1, k, β * ∆w[2]*Fu[i,2,k], Val(OP))  # Fu[i,1,k] == 0
                end

                # 3. At the positive end of the y-direction (where the boundary fields are
                # assumed zero)
                β = α2 * ∆w′⁻¹[Ny]
                for k = 1:Nz, i = 1:Nx
                    @inbounds set_or_add!(Gv, i, Ny, k, β * ∆w[Ny]*Fu[i,Ny,k], Val(OP))  # Fu[i,Ny+1,k] == 0
                end
            end
        else  # nw == 3; w = z
            if isbloch
                # 1. At locations except for the positive end of the z-direction
                kₑ[Nₜ] = Nz-1  # initially kₑ[Nₜ] = Nz
                for t = 1:Nₜ
                    kₛₜ, kₑₜ = kₛ[t], kₑ[t]
                    let kₛₜ=kₛₜ, kₑₜ=kₑₜ
                        @spawn for k = kₛₜ:kₑₜ
                            for j = 1:Ny, i = 1:Nx
                                @inbounds set_or_add!(Gv, i, j, k, (α2 * ∆w′⁻¹[k]) * (∆w[k+1]*Fu[i,j,k+1] + ∆w[k]*Fu[i,j,k]), Val(OP))
                            end
                        end
                    end
                end

                # 2. At the positive end of the z-direction (where the boundary fields are
                # taken from the negative-end boundary)
                β = α2 * ∆w′⁻¹[Nz]
                for j = 1:Ny, i = 1:Nx
                    @inbounds set_or_add!(Gv, i, j, Nz, β * (∆w[1]*e⁻ⁱᵏᴸ*Fu[i,j,1] + ∆w[Nz]*Fu[i,j,Nz]), Val(OP))  # Fu[i,j,Nz+1] = exp(-i kz Lz) * Fu[i,j,1]
                end
            else  # symmetry boundary
                # 1. At the locations except for the positive and negative ends of the
                # z-direction
                kₛ[1] = 2  # initially kₛ[1] = 1
                kₑ[Nₜ] = Nz-1  # initially kₑ[Nₜ] = Nz
                for t = 1:Nₜ
                    kₛₜ, kₑₜ = kₛ[t], kₑ[t]
                    let kₛₜ=kₛₜ, kₑₜ=kₑₜ
                        @spawn for k = kₛₜ:kₑₜ
                            for j = 1:Ny, i = 1:Nx
                                @inbounds set_or_add!(Gv, i, j, k, (α2 * ∆w′⁻¹[k]) * (∆w[k+1]*Fu[i,j,k+1] + ∆w[k]*Fu[i,j,k]), Val(OP))
                            end
                        end
                    end
                end

                # 2. At the negative end of the z-direction (where the boundary fields are
                # assumed zero)
                β = α2 * ∆w′⁻¹[1]
                for j = 1:Ny, i = 1:Nx
                    @inbounds set_or_add!(Gv, i, j, 1, β * ∆w[2]*Fu[i,j,2], Val(OP))  # Fu[i,j,1] == 0
                end

                # 3. At the positive end of the z-direction (where the boundary fields are
                # assumed zero)
                β = α2 * ∆w′⁻¹[Nz]
                for j = 1:Ny, i = 1:Nx
                    @inbounds set_or_add!(Gv, i, j, Nz, β * ∆w[Nz]*Fu[i,j,Nz], Val(OP))  # Fu[i,j,Nz+1] == 0
                end
            end
        end  # if nw == ...
    else  # backward averaging
        if nw == 1  # w = x
            # 1. At the locations except for the negative end of the x-direction; unlike for
            # the forward difference, for the backward difference this part of the code is
            # common for both the Bloch and symmetry boundary conditions.
            for t = 1:Nₜ
                kₛₜ, kₑₜ = kₛ[t], kₑ[t]
                let kₛₜ=kₛₜ, kₑₜ=kₑₜ
                    @spawn for k = kₛₜ:kₑₜ
                        for j = 1:Ny, i = 2:Nx  # not i = 2:Nx-1
                            @inbounds set_or_add!(Gv, i, j, k, (α2 * ∆w′⁻¹[i]) * (∆w[i]*Fu[i,j,k] + ∆w[i-1]*Fu[i-1,j,k]), Val(OP))
                        end
                    end
                end
            end

            # 2. At the negative end of the x-direction (where the boundary fields are taken
            # from the positive-end boundary for the Bloch boundary condition)
            if isbloch
                β = α2 * ∆w′⁻¹[1]
                for k = 1:Nz, j = 1:Ny
                    @inbounds set_or_add!(Gv, 1, j, k, β * (∆w[1]*Fu[1,j,k] + ∆w[Nx]*Fu[Nx,j,k]/e⁻ⁱᵏᴸ), Val(OP))  # Fu[0,j,k] = Fu[Nx,j,k] / exp(-i kx Lx)
                end
            else  # symmetry boundary
                β = α * ∆w′⁻¹[1]
                for k = 1:Nz, j = 1:Ny
                    @inbounds set_or_add!(Gv, 1, j, k, β * ∆w[1]*Fu[1,j,k], Val(OP))  # Fu[0,j,k] = Fu[1,j,k]
                end
            end
        elseif nw == 2  # w = y
            # 1. At the locations except for the negative end of the y-direction; unlike for
            # the forward difference, for the backward difference this part of the code is
            # common for both the Bloch and symmetry boundary conditions.
            for t = 1:Nₜ
                kₛₜ, kₑₜ = kₛ[t], kₑ[t]
                let kₛₜ=kₛₜ, kₑₜ=kₑₜ
                    @spawn for k = kₛₜ:kₑₜ
                        for j = 2:Ny, i = 1:Nx  # not j = 2:Ny-1
                            @inbounds set_or_add!(Gv, i, j, k, (α2 * ∆w′⁻¹[j]) * (∆w[j]*Fu[i,j,k] + ∆w[j-1]*Fu[i,j-1,k]), Val(OP))
                        end
                    end
                end
            end

            # 2. At the negative end of the y-direction (where the boundary fields are taken
            # from the positive-end boundary for the Bloch boundary condition)
            if isbloch
                β = α2 * ∆w′⁻¹[1]
                for k = 1:Nz, i = 1:Nx
                    @inbounds set_or_add!(Gv, i, 1, k, β * (∆w[1]*Fu[i,1,k] + ∆w[Ny]*Fu[i,Ny,k]/e⁻ⁱᵏᴸ), Val(OP))  # Fu[i,0,k] = Fu[0,Ny,k] / exp(-i ky Ly)
                end
            else  # symmetry boundary
                β = α * ∆w′⁻¹[1]
                for k = 1:Nz, i = 1:Nx
                    @inbounds set_or_add!(Gv, i, 1, k, β * ∆w[1]*Fu[i,1,k], Val(OP))  # Fu[i,0,k] = Fu[i,1,k]
                end
            end
        else  # nw == 3; w = z
            # 1. At the locations except for the negative end of the z-direction; unlike for
            # the forward difference, for the backward difference this part of the code is
            # common for both the Bloch and symmetry boundary conditions.
            kₛ[1] = 2  # initially kₛ[1] = 1
            for t = 1:Nₜ
                kₛₜ, kₑₜ = kₛ[t], kₑ[t]
                let kₛₜ=kₛₜ, kₑₜ=kₑₜ
                    @spawn for k = kₛₜ:kₑₜ
                        for j = 1:Ny, i = 1:Nx
                            @inbounds set_or_add!(Gv, i, j, k, (α2 * ∆w′⁻¹[k]) * (∆w[k]*Fu[i,j,k] + ∆w[k-1]*Fu[i,j,k-1]), Val(OP))
                        end
                    end
                end
            end

            # 2. At the negative end of the z-direction (where the boundary fields are taken
            # from the positive-end boundary for the Bloch boundary condition)
            if isbloch
                β = α2 * ∆w′⁻¹[1]
                for j = 1:Ny, i = 1:Nx
                    @inbounds set_or_add!(Gv, i, j, 1, β * (∆w[1]*Fu[i,j,1] + ∆w[Nz]*Fu[i,j,Nz]/e⁻ⁱᵏᴸ), Val(OP))  # Fu[i,j,0] = Fu[i,j,Nz] / exp(-i kz Lz)
                end
            else  # symmetry boundary
                β = α * ∆w′⁻¹[1]
                for j = 1:Ny, i = 1:Nx
                    @inbounds set_or_add!(Gv, i, j, 1, β * ∆w[1]*Fu[i,j,1], Val(OP))  # Fu[i,j,0] = Fu[i,j,Nz]
                end
            end
        end  # if nw == ...
    end  # if isfwd

    # Recover the original values of the potentially changed kₛ[1] and kₑ[Nt].
    kₛ[1] = 1
    kₑ[Nₜ] = Nz

    return nothing
end

# Concrete apply_m! for 2D
function apply_m!(Gv::AbsArrNumber{2},  # v-component of output field (v = x, y)
                  Fu::AbsArrNumber{2},  # u-component of input field (u = x, y)
                  ::Val{OP},  # Val(:(=)) or Val(:(+=)): set (=) or add (+=) operator to use
                  nw::Integer,  # 1|2 for averaging along x|y
                  isfwd::Bool,  # true|false for forward|backward averaging
                  ∆w::AbsVecNumber,  # line segments to multiply with; vector of length N[nw]
                  ∆w′⁻¹::AbsVecNumber,  # inverse of line segments to divide by; vector of length N[nw]
                  isbloch::Bool,  # boundary condition in w-direction
                  e⁻ⁱᵏᴸ::Number;  # Bloch phase factor
                  n_bounds::Tuple2{AbsVecInteger}=calc_boundary_indices(size(Gv)),  # (nₛ,nₑ): start and end indices of chunks in last dimension to be processed in parallel
                  α::Number=1  # scale factor to multiply to result before adding it to Gv: Gv += α m(Fu)
                  ) where {OP}
    @assert(size(Gv)==size(Fu))
    @assert(size(Fu,nw)==length(∆w))
    @assert(length(∆w)==length(∆w′⁻¹))

    Nx, Ny = size(Fu)
    jₛ, jₑ = n_bounds
    Nₜ = length(jₛ)

    α2 = 0.5 * α

    # Make sure not to include branches inside for loops.
    @sync if isfwd
        if nw == 1  # w = x
            if isbloch
                # 1. At locations except for the positive end of the x-direction
                for t = 1:Nₜ
                    jₛₜ, jₑₜ = jₛ[t], jₑ[t]
                    let jₛₜ=jₛₜ, jₑₜ=jₑₜ
                        @spawn for j = jₛₜ:jₑₜ
                            for i = 1:Nx-1
                                @inbounds set_or_add!(Gv, i, j, (α2 * ∆w′⁻¹[i]) * (∆w[i+1]*Fu[i+1,j] + ∆w[i]*Fu[i,j]), Val(OP))
                            end
                        end
                    end
                end

                # 2. At the positive end of the x-direction (where the boundary fields are
                # taken from the negative-end boundary)
                β = α2 * ∆w′⁻¹[Nx]
                for j = 1:Ny
                    @inbounds set_or_add!(Gv, Nx, j, β * (∆w[1]*e⁻ⁱᵏᴸ*Fu[1,j] + ∆w[Nx]*Fu[Nx,j]), Val(OP))  # Fu[Nx+1,j] = exp(-i kx Lx) * Fu[1,j]
                end
            else  # symmetry boundary
                # 1. At the locations except for the positive and negative ends of the
                # x-direction
                for t = 1:Nₜ
                    jₛₜ, jₑₜ = jₛ[t], jₑ[t]
                    let jₛₜ=jₛₜ, jₑₜ=jₑₜ
                        @spawn for j = jₛₜ:jₑₜ
                            for i = 2:Nx-1
                                @inbounds set_or_add!(Gv, i, j, (α2 * ∆w′⁻¹[i]) * (∆w[i+1]*Fu[i+1,j] + ∆w[i]*Fu[i,j]), Val(OP))
                            end
                        end
                    end
                end

                # 2. At the negative end of the x-direction (where the boundary fields are
                # assumed zero)
                β = α2 * ∆w′⁻¹[1]
                for j = 1:Ny
                    @inbounds set_or_add!(Gv, 1, j, β * ∆w[2]*Fu[2,j], Val(OP))  # Fu[1,j] == 0
                end

                # 3. At the positive end of the x-direction (where the boundary fields are
                # assumed zero)
                β = α2 * ∆w′⁻¹[Nx]
                for j = 1:Ny
                    @inbounds set_or_add!(Gv, Nx, j, β * ∆w[Nx]*Fu[Nx,j], Val(OP))  # Fu[Nx+1,j] == 0
                end
            end
        else  # nw == 2; w = y
            if isbloch
                # 1. At locations except for the positive end of the y-direction
                jₑ[Nₜ] = Ny-1  # initially jₑ[Nₜ] = Ny
                for t = 1:Nₜ
                    jₛₜ, jₑₜ = jₛ[t], jₑ[t]
                    let jₛₜ=jₛₜ, jₑₜ=jₑₜ
                        @spawn for j = jₛₜ:jₑₜ
                            for i = 1:Nx
                                @inbounds set_or_add!(Gv, i, j, (α2 * ∆w′⁻¹[j]) * (∆w[j+1]*Fu[i,j+1] + ∆w[j]*Fu[i,j]), Val(OP))
                            end
                        end
                    end
                end

                # 2. At the positive end of the y-direction (where the boundary fields are
                # taken from the negative-end boundary)
                β = α2 * ∆w′⁻¹[Ny]
                for i = 1:Nx
                    @inbounds set_or_add!(Gv, i, Ny, β * (∆w[1]*e⁻ⁱᵏᴸ*Fu[i,1] + ∆w[Ny]*Fu[i,Ny]), Val(OP))  # Fu[i,Ny+1] = exp(-i ky Ly) * Fu[i,1]
                end
            else  # symmetry boundary
                # 1. At the locations except for the positive and negative ends of the
                # y-direction
                jₛ[1] = 2  # initially jₛ[1] = 1
                jₑ[Nₜ] = Ny-1  # initially jₛ[Nₜ] = Ny
                for t = 1:Nₜ
                    jₛₜ, jₑₜ = jₛ[t], jₑ[t]
                    let jₛₜ=jₛₜ, jₑₜ=jₑₜ
                        @spawn for j = jₛₜ:jₑₜ
                            for i = 1:Nx
                                @inbounds set_or_add!(Gv, i, j, (α2 * ∆w′⁻¹[j]) * (∆w[j+1]*Fu[i,j+1] + ∆w[j]*Fu[i,j]), Val(OP))
                            end
                        end
                    end
                end

                # 2. At the negative end of the y-direction (where the boundary fields are
                # assumed zero)
                β = α2 * ∆w′⁻¹[1]
                for i = 1:Nx
                    @inbounds set_or_add!(Gv, i, 1, β * ∆w[2]*Fu[i,2], Val(OP))  # Fu[i,1] == 0
                end

                # 3. At the positive end of the y-direction (where the boundary fields are
                # assumed zero)
                β = α2 * ∆w′⁻¹[Ny]
                for i = 1:Nx
                    @inbounds set_or_add!(Gv, i, Ny, β * ∆w[Ny]*Fu[i,Ny], Val(OP))  # Fu[i,Ny+1] == 0
                end
            end
        end  # if nw == ...
    else  # backward averaging
        if nw == 1  # w = x
            # 1. At the locations except for the negative end of the x-direction; unlike for
            # the forward difference, for the backward difference this part of the code is
            # common for both the Bloch and symmetry boundary conditions.
            for t = 1:Nₜ
                jₛₜ, jₑₜ = jₛ[t], jₑ[t]
                let jₛₜ=jₛₜ, jₑₜ=jₑₜ
                    @spawn for j = jₛₜ:jₑₜ
                        for i = 2:Nx  # not i = 2:Nx-1
                            @inbounds set_or_add!(Gv, i, j, (α2 * ∆w′⁻¹[i]) * (∆w[i]*Fu[i,j] + ∆w[i-1]*Fu[i-1,j]), Val(OP))
                        end
                    end
                end
            end

            # 2. At the negative end of the x-direction (where the boundary fields are taken
            # from the positive-end boundary for the Bloch boundary condition)
            if isbloch
                β = α2 * ∆w′⁻¹[1]
                for j = 1:Ny
                    @inbounds set_or_add!(Gv, 1, j, β * (∆w[1]*Fu[1,j] + ∆w[Nx]*Fu[Nx,j]/e⁻ⁱᵏᴸ), Val(OP))  # Fu[0,j] = Fu[Nx,j] / exp(-i kx Lx)
                end
            else  # symmetry boundary
                β = α * ∆w′⁻¹[1]
                for j = 1:Ny
                    @inbounds set_or_add!(Gv, 1, j, β * ∆w[1]*Fu[1,j], Val(OP))  # Fu[0,j] = Fu[1,j]
                end
            end
        elseif nw == 2  # w = y
            # 1. At the locations except for the negative end of the y-direction; unlike for
            # the forward difference, for the backward difference this part of the code is
            # common for both the Bloch and symmetry boundary conditions.
            jₛ[1] = 2  # initially jₛ[1] = 1
            for t = 1:Nₜ
                jₛₜ, jₑₜ = jₛ[t], jₑ[t]
                let jₛₜ=jₛₜ, jₑₜ=jₑₜ
                    @spawn for j = jₛₜ:jₑₜ
                        for i = 1:Nx
                            @inbounds set_or_add!(Gv, i, j, (α2 * ∆w′⁻¹[j]) * (∆w[j]*Fu[i,j] + ∆w[j-1]*Fu[i,j-1]), Val(OP))
                        end
                    end
                end
            end

            # 2. At the negative end of the y-direction (where the boundary fields are taken
            # from the positive-end boundary for the Bloch boundary condition)
            if isbloch
                β = α2 * ∆w′⁻¹[1]
                for i = 1:Nx
                    @inbounds set_or_add!(Gv, i, 1, β * (∆w[1]*Fu[i,1] + ∆w[Ny]*Fu[i,Ny]/e⁻ⁱᵏᴸ), Val(OP))  # Fu[i,0] = Fu[0,Ny] / exp(-i ky Ly)
                end
            else  # symmetry boundary
                β = α * ∆w′⁻¹[1]
                for i = 1:Nx
                    @inbounds set_or_add!(Gv, i, 1, β * ∆w[1]*Fu[i,1], Val(OP))  # Fu[i,0] = Fu[i,1]
                end
            end
        end  # if nw == ...
    end  # if isfwd

    # Recover the original values of the potentially changed jₛ[1] and jₑ[Nt].
    jₛ[1] = 1
    jₑ[Nₜ] = Ny

    return nothing
end

# Concrete apply_m! for 1D
function apply_m!(Gv::AbsArrNumber{1},  # v-component of output field (v = x, y)
                  Fu::AbsArrNumber{1},  # u-component of input field (u = x, y)
                  ::Val{OP},  # Val(:(=)) or Val(:(+=)): set (=) or add (+=) operator to use
                  nw::Integer,  # 1|2 for averaging along x|y
                  isfwd::Bool,  # true|false for forward|backward averaging
                  ∆w::AbsVecNumber,  # line segments to multiply with; vector of length N[nw]
                  ∆w′⁻¹::AbsVecNumber,  # inverse of line segments to divide by; vector of length N[nw]
                  isbloch::Bool,  # boundary condition in w-direction
                  e⁻ⁱᵏᴸ::Number;  # Bloch phase factor
                  n_bounds::Tuple2{AbsVecInteger}=calc_boundary_indices(size(Gv)),  # (nₛ,nₑ): start and end indices of chunks in last dimension to be processed in parallel
                  α::Number=1  # scale factor to multiply to result before adding it to Gv: Gv += α m(Fu)
                  ) where {OP}
    @assert(size(Gv)==size(Fu))
    @assert(size(Fu,nw)==length(∆w))
    @assert(length(∆w)==length(∆w′⁻¹))

    Nx = length(Fu)  # not size(Fu) unlike code for 2D and 3D
    iₛ, iₑ = n_bounds
    Nₜ = length(iₛ)

    α2 = 0.5 * α

    # Make sure not to include branches inside for loops.
    @sync if isfwd
        if isbloch
            # 1. At locations except for the positive end of the x-direction
            iₑ[Nₜ] = Nx-1  # initially iₑ[Nₜ] = Nx
            for t = 1:Nₜ
                iₛₜ, iₑₜ = iₛ[t], iₑ[t]
                let iₛₜ=iₛₜ, iₑₜ=iₑₜ
                    @spawn for i = iₛₜ:iₑₜ
                        @inbounds set_or_add!(Gv, i, (α2 * ∆w′⁻¹[i]) * (∆w[i+1]*Fu[i+1] + ∆w[i]*Fu[i]), Val(OP))
                    end
                end
            end

            # 2. At the positive end of the x-direction (where the boundary fields are taken
            # from the negative-end boundary)
            β = α2 * ∆w′⁻¹[Nx]
            @inbounds set_or_add!(Gv, Nx, β * (∆w[1]*e⁻ⁱᵏᴸ*Fu[1] + ∆w[Nx]*Fu[Nx]), Val(OP))  # Fu[Nx+1] = exp(-i kx Lx) * Fu[1]
        else  # symmetry boundary
            # 1. At the locations except for the positive and negative ends of the x-direction
            iₛ[1] = 2  # initially iₛ[1] = 1
            iₑ[Nₜ] = Nx-1  # initially iₑ[Nₜ] = Nx
            for t = 1:Nₜ
                iₛₜ, iₑₜ = iₛ[t], iₑ[t]
                let iₛₜ=iₛₜ, iₑₜ=iₑₜ
                    @spawn for i = iₛₜ:iₑₜ
                        @inbounds set_or_add!(Gv, i, (α2 * ∆w′⁻¹[i]) * (∆w[i+1]*Fu[i+1] + ∆w[i]*Fu[i]), Val(OP))
                    end
                end
            end

            # 2. At the negative end of the x-direction (where the boundary fields are assumed
            # zero)
            β = α2 * ∆w′⁻¹[1]
            @inbounds set_or_add!(Gv, 1, β * ∆w[2]*Fu[2], Val(OP))  # Fu[1] == 0

            # 3. At the positive end of the x-direction (where the boundary fields are assumed
            # zero)
            β = α2 * ∆w′⁻¹[Nx]
            @inbounds set_or_add!(Gv, Nx, β * ∆w[Nx]*Fu[Nx], Val(OP))  # Fu[Nx+1] == 0
        end
    else  # backward averaging
        # 1. At the locations except for the negative end of the x-direction; unlike for the
        # forward difference, for the backward difference this part of the code is common
        # for both the Bloch and symmetry boundary conditions.
        iₛ[1] = 2  # initially iₛ[1] = 1
        for t = 1:Nₜ
            iₛₜ, iₑₜ = iₛ[t], iₑ[t]
            let iₛₜ=iₛₜ, iₑₜ=iₑₜ
                @spawn for i = iₛₜ:iₑₜ
                    @inbounds set_or_add!(Gv, i, (α2 * ∆w′⁻¹[i]) * (∆w[i]*Fu[i] + ∆w[i-1]*Fu[i-1]), Val(OP))
                end
            end
        end

        # 2. At the negative end of the x-direction (where the boundary fields are taken
        # from the positive-end boundary for the Bloch boundary condition)
        if isbloch
            β = α2 * ∆w′⁻¹[1]
            @inbounds set_or_add!(Gv, 1, β * (∆w[1]*Fu[1] + ∆w[Nx]*Fu[Nx]/e⁻ⁱᵏᴸ), Val(OP))  # Fu[0] = Fu[Nx] / exp(-i kx Lx)
        else  # symmetry boundary
            β = α * ∆w′⁻¹[1]
            @inbounds set_or_add!(Gv, 1, β * ∆w[1]*Fu[1], Val(OP))  # Fu[0] = Fu[1]
        end
    end  # if isfwd

    # Recover the original values of the potentially changed iₛ[1] and iₑ[Nt].
    iₛ[1] = 1
    iₑ[Nₜ] = Nx

    return nothing
end
