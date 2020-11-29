# Average fields along the field directions (i.e., Fw along the w-direction).  See the
# description of matrix/mean.jl/create_m().  Technically, apply_m!() can be used to average
# fields normal to the direction of averaging, but it is not used that way in apply_mean!().

# Assumes the space dimension and field dimension are the same.  In other words, when the
# space coordinate indices are (i,j,k), then the field has three vector components.
# Therefore, for the input field array F[i,j,k,w], we assume w = 1:3.
export apply_m!, apply_mean!

apply_mean!(G::T,  # output field; G[i,j,k,w] is w-component of G at (i,j,k)
            F::T,  # input field; G[i,j,k,w] is w-component of G at (i,j,k)
            isfwd::AbsVecBool,  # isfwd[w] = true|false for forward|backward averaging
            isbloch::AbsVecBool=fill(true,length(isfwd)),  # boundary conditions in x, y, z
            e⁻ⁱᵏᴸ::AbsVecNumber=ones(length(isfwd));  # Bloch phase factor in x, y, z
            α::Number=1.0  # scale factor to multiply to result before adding it to G: G += α ∇×F
            ) where {T<:AbsArrNumber{4}} =
    (N = size(F)[nXYZ]; ∆l = ones.((N...,)); ∆l⁻¹ = ones.((N...,)); apply_mean!(G, F, isfwd, ∆l, ∆l⁻¹, isbloch, e⁻ⁱᵏᴸ, α=α))

apply_mean!(G::T,  # output field; G[i,j,k,w] is w-component of G at (i,j,k)
            F::T,  # input field; G[i,j,k,w] is w-component of G at (i,j,k)
            isfwd::AbsVecBool,  # isfwd[w] = true|false for forward|backward averaging
            ∆l::Tuple3{AbsVecNumber},  # line segments to multiply with; vectors of length N
            ∆l′⁻¹::Tuple3{AbsVecNumber},  # inverse of line segments to divide by; vectors of length N
            isbloch::AbsVecBool=fill(true,length(isfwd)),  # boundary conditions in x, y, z
            e⁻ⁱᵏᴸ::AbsVecNumber=ones(length(isfwd));  # Bloch phase factor in x, y, z
            α::Number=1.0  # scale factor to multiply to result before adding it to G: G += α mean(F)
            ) where {T<:AbsArrNumber{4}} =
    (K = length(isfwd); apply_mean!(G, F, SVector{K}(isfwd), ∆l, ∆l′⁻¹, SVector{K}(isbloch), SVector{K}(e⁻ⁱᵏᴸ), α=α))

# For the implementation, see the comments in matrix/mean.jl.
function apply_mean!(G::T,  # output field; G[i,j,k,w] is w-component of G at (i,j,k)
                     F::T,  # input field; G[i,j,k,w] is w-component of G at (i,j,k)
                     isfwd::SBool{3},  # isfwd[w] = true|false for forward|backward averaging
                     ∆l::Tuple3{AbsVecNumber},  # line segments to multiply with; vectors of length N
                     ∆l′⁻¹::Tuple3{AbsVecNumber},  # inverse of line segments to divide by; vectors of length N
                     isbloch::SBool{3},  # boundary conditions in x, y, z
                     e⁻ⁱᵏᴸ::SNumber{3};  # Bloch phase factor in x, y, z
                     α::Number=1  # scale factor to multiply to result before adding it to G: G += α mean(F)
                     ) where {T<:AbsArrNumber{4}}
    indblk = 0  # index of matrix block
    for nv = nXYZ  # Cartesian compotent of output field
        Gv = @view G[:,:,:,nv]  # v-component of output field

        nu = nv  # Cartesian component of input field
        Fu = @view F[:,:,:,nu]  # u-component of input field

        apply_m!(Gv, Fu, nv, isfwd[nv], ∆l[nv], ∆l′⁻¹[nv], isbloch[nv], e⁻ⁱᵏᴸ[nv], α=α)
        indblk += 1
    end
end

apply_m!(Gv::AbsArrNumber{K},  # v-component of output field (v = x, y, z)
         Fu::AbsArrNumber{K},  # u-component of input field (u = x, y, z)
         nw::Integer,  # 1|2|3 for averaging along x|y|z; 1|2 for averaging along horizontal|vertical
         isfwd::Bool,  # true|false for forward|backward averaging
         ∆w::Number,  # line segments to multiply with; vector of length N[nw]
         isbloch::Bool=true,  # boundary condition in w-direction
         e⁻ⁱᵏᴸ::Number=1;  # Bloch phase factor
         α::Number=1  # scale factor to multiply to result before adding it to Gv: Gv += α m(Fu)
         ) where {K} =  # space dimension (field dimension Kf does not show up as we deal with single component)
    (N = size(Fu); ∆w_vec = fill(∆w, N[nw]); ∆w′⁻¹_vec = fill(1/∆w, N[nw]); apply_m!(Gv, Fu, nw, isfwd, ∆w_vec, ∆w′⁻¹_vec, isbloch, e⁻ⁱᵏᴸ, α=α))  # fill: create vector of ∆w

apply_m!(Gv::AbsArrNumber{K},  # v-component of output field (v = x, y, z)
         Fu::AbsArrNumber{K},  # u-component of input field (u = x, y, z)
         nw::Integer,  # 1|2|3 for averaging along x|y|z; 1|2 for averaging along horizontal|vertical
         isfwd::Bool,  # true|false for forward|backward averaging
         ∆w::AbsVecNumber=ones(size(Fu)[nw]),  # line segments to multiply with; vector of length N[nw]
         ∆w′⁻¹::AbsVecNumber=ones(size(Fu)[nw]),  # inverse of line segments to divide by; vector of length N[nw]
         isbloch::Bool=true,  # boundary condition in w-direction
         e⁻ⁱᵏᴸ::Number=1;  # Bloch phase factor
         α::Number=1  # scale factor to multiply to result before adding it to Gv: Gv += α m(Fu)
         ) where {K} =  # space dimension (field dimension Kf does not show up as we deal with single component)
    (N = size(Fu); ∆w_vec = fill(∆w, N[nw]); ∆w′⁻¹_vec = fill(1/∆w, N[nw]); apply_m!(Gv, Fu, nw, isfwd, ∆w_vec, ∆w′⁻¹_vec, isbloch, e⁻ⁱᵏᴸ, α=α))  # fill: create vector of ∆w

function apply_m!(Gv::AbsArrNumber{3},  # v-component of output field (v = x, y, z)
                  Fu::AbsArrNumber{3},  # u-component of input field (u = x, y, z)
                  nw::Integer,  # 1|2|3 for averaging along x|y|z; 1|2 for averaging along horizontal|vertical
                  isfwd::Bool,  # true|false for forward|backward averaging
                  ∆w::AbsVecNumber,  # line segments to multiply with; vector of length N[nw]
                  ∆w′⁻¹::AbsVecNumber,  # inverse of line segments to divide by; vector of length N[nw]
                  isbloch::Bool,  # boundary condition in w-direction
                  e⁻ⁱᵏᴸ::Number;  # Bloch phase factor
                  α::Number=1)  # scale factor to multiply to result before adding it to Gv: Gv += α m(Fu)
    @assert(size(Gv)==size(Fu))
    @assert(size(Fu,nw)==length(∆w))
    @assert(length(∆w)==length(∆w′⁻¹))

    Nx, Ny, Nz = size(Fu)
    α2 = 0.5 * α

    # Make sure not to include branches inside for loops.
    if isfwd
        if nw == 1  # w = x
            if isbloch
                # At locations except for the positive end of the x-direction
                for k = 1:Nz, j = 1:Ny, i = 1:Nx-1
                    β = α2 * ∆w′⁻¹[i]
                    @inbounds Gv[i,j,k] += β * (∆w[i+1]*Fu[i+1,j,k] + ∆w[i]*Fu[i,j,k])
                end

                # At the positive end of the x-direction (where the boundary fields are
                # taken from the negative-end boundary)
                β = α2 * ∆w′⁻¹[Nx]
                for k = 1:Nz, j = 1:Ny
                    @inbounds Gv[Nx,j,k] += β * (∆w[1]*e⁻ⁱᵏᴸ*Fu[1,j,k] + ∆w[Nx]*Fu[Nx,j,k])  # Fu[Nx+1,j,k] = exp(-i kx Lx) * Fu[1,j,k]
                end
            else  # symmetry boundary
                # At the locations except for the positive and negative ends of the
                # x-direction
                for k = 1:Nz, j = 1:Ny, i = 2:Nx-1
                    β = α2 * ∆w′⁻¹[i]
                    @inbounds Gv[i,j,k] += β * (∆w[i+1]*Fu[i+1,j,k] + ∆w[i]*Fu[i,j,k])
                end

                # At the negative end of the x-direction (where the boundary fields are
                # assumed zero)
                β = α2 * ∆w′⁻¹[1]
                for k = 1:Nz, j = 1:Ny
                    @inbounds Gv[1,j,k] += β * ∆w[2]*Fu[2,j,k]  # Fu[1,j,k] == 0
                end

                # At the positive end of the x-direction (where the boundary fields are
                # assumed zero)
                β = α2 * ∆w′⁻¹[Nx]
                for k = 1:Nz, j = 1:Ny
                    @inbounds Gv[Nx,j,k] += β * ∆w[Nx]*Fu[Nx,j,k]  # Fu[Nx+1,j,k] == 0
                end
            end
        elseif nw == 2  # w = y
            if isbloch
                # At locations except for the positive end of the y-direction
                for k = 1:Nz, j = 1:Ny-1
                    β = α2 * ∆w′⁻¹[j]
                    for i = 1:Nx
                        @inbounds Gv[i,j,k] += β * (∆w[j+1]*Fu[i,j+1,k] + ∆w[j]*Fu[i,j,k])
                    end
                end

                # At the positive end of the y-direction (where the boundary fields are
                # taken from the negative-end boundary)
                β = α2 * ∆w′⁻¹[Ny]
                for k = 1:Nz, i = 1:Nx
                    @inbounds Gv[i,Ny,k] += β * (∆w[1]*e⁻ⁱᵏᴸ*Fu[i,1,k] + ∆w[Ny]*Fu[i,Ny,k])  # Fu[i,Ny+1,k] = exp(-i ky Ly) * Fu[i,1,k]
                end
            else  # symmetry boundary
                # At the locations except for the positive and negative ends of the
                # y-direction
                for k = 1:Nz, j = 2:Ny-1
                    β = α2 * ∆w′⁻¹[j]
                    for i = 1:Nx
                        @inbounds Gv[i,j,k] += β * (∆w[j+1]*Fu[i,j+1,k] + ∆w[j]*Fu[i,j,k])
                    end
                end

                # At the negative end of the y-direction (where the boundary fields are
                # assumed zero)
                β = α2 * ∆w′⁻¹[1]
                for k = 1:Nz, i = 1:Nx
                    @inbounds Gv[i,1,k] += β * ∆w[2]*Fu[i,2,k]  # Fu[i,1,k] == 0
                end

                # At the positive end of the y-direction (where the boundary fields are
                # assumed zero)
                β = α2 * ∆w′⁻¹[Ny]
                for k = 1:Nz, i = 1:Nx
                    @inbounds Gv[i,Ny,k] += β * ∆w[Ny]*Fu[i,Ny,k]  # Fu[i,Ny+1,k] == 0
                end
            end
        else  # nw == 3; w = z
            if isbloch
                # At locations except for the positive end of the z-direction
                for k = 1:Nz-1
                    β = α2 * ∆w′⁻¹[k]
                    for j = 1:Ny, i = 1:Nx
                        @inbounds Gv[i,j,k] += β * (∆w[k+1]*Fu[i,j,k+1] + ∆w[k]*Fu[i,j,k])
                    end
                end

                # At the positive end of the z-direction (where the boundary fields are
                # taken from the negative-end boundary)
                β = α2 * ∆w′⁻¹[Nz]
                for j = 1:Ny, i = 1:Nx
                    @inbounds Gv[i,j,Nz] += β * (∆w[1]*e⁻ⁱᵏᴸ*Fu[i,j,1] + ∆w[Nz]*Fu[i,j,Nz])  # Fu[i,j,Nz+1] = exp(-i kz Lz) * Fu[i,j,1]
                end
            else  # symmetry boundary
                # At the locations except for the positive and negative ends of the
                # z-direction
                for k = 2:Nz-1
                    β = α2 * ∆w′⁻¹[k]
                    for j = 1:Ny, i = 1:Nx
                        @inbounds Gv[i,j,k] += β * (∆w[k+1]*Fu[i,j,k+1] + ∆w[k]*Fu[i,j,k])
                    end
                end

                # At the negative end of the z-direction (where the boundary fields are
                # assumed zero)
                β = α2 * ∆w′⁻¹[1]
                for j = 1:Ny, i = 1:Nx
                    @inbounds Gv[i,j,1] += β * ∆w[2]*Fu[i,j,2]  # Fu[i,j,1] == 0
                end

                # At the positive end of the z-direction (where the boundary fields are
                # assumed zero)
                β = α2 * ∆w′⁻¹[Nz]
                for j = 1:Ny, i = 1:Nx
                    @inbounds Gv[i,j,Nz] += β * ∆w[Nz]*Fu[i,j,Nz]  # Fu[i,j,Nz+1] == 0
                end
            end
        end  # if nw == ...
    else  # backward averaging
        if nw == 1  # w = x
            # At the locations except for the negative end of the x-direction; unlike for
            # the forward difference, for the backward difference this part of the code is
            # common for both the Bloch and symmetry boundary conditions.
            for k = 1:Nz, j = 1:Ny, i = 2:Nx  # not i = 2:Nx-1
                β = α2 * ∆w′⁻¹[i]
                @inbounds Gv[i,j,k] += β * (∆w[i]*Fu[i,j,k] + ∆w[i-1]*Fu[i-1,j,k])
            end

            if isbloch
                # At the negative end of the x-direction (where the boundary fields are
                # taken from the positive-end boundary)
                β = α2 * ∆w′⁻¹[1]
                for k = 1:Nz, j = 1:Ny
                    @inbounds Gv[1,j,k] += β * (∆w[1]*Fu[1,j,k] + ∆w[Nx]*Fu[Nx,j,k]/e⁻ⁱᵏᴸ)  # Fu[0,j,k] = Fu[Nx,j,k] / exp(-i kx Lx)
                end
            else  # symmetry boundary
                # At the negative end of the x-direction (where the boundary fields are
                # taken from the positive-end boundary)
                β = α * ∆w′⁻¹[1]
                for k = 1:Nz, j = 1:Ny
                    @inbounds Gv[1,j,k] += β * ∆w[1]*Fu[1,j,k]  # Fu[0,j,k] = Fu[1,j,k]
                end
            end
        elseif nw == 2  # w = y
            # At the locations except for the negative end of the y-direction; unlike for
            # the forward difference, for the backward difference this part of the code is
            # common for both the Bloch and symmetry boundary conditions.
            for k = 1:Nz, j = 2:Ny  # not j = 2:Ny-1
                β = α2 * ∆w′⁻¹[j]
                for i = 1:Nx
                    @inbounds Gv[i,j,k] += β * (∆w[j]*Fu[i,j,k] + ∆w[j-1]*Fu[i,j-1,k])
                end
            end

            if isbloch
                # At the negative end of the y-direction (where the boundary fields are
                # taken from the positive-end boundary)
                β = α2 * ∆w′⁻¹[1]
                for k = 1:Nz, i = 1:Nx
                    @inbounds Gv[i,1,k] += β * (∆w[1]*Fu[i,1,k] + ∆w[Ny]*Fu[i,Ny,k]/e⁻ⁱᵏᴸ)  # Fu[i,0,k] = Fu[0,Ny,k] / exp(-i ky Ly)
                end
            else  # symmetry boundary
                # At the negative end of the y-direction (where the boundary fields are
                # taken from the positive-end boundary)
                β = α * ∆w′⁻¹[1]
                for k = 1:Nz, i = 1:Nx
                    @inbounds Gv[i,1,k] += β * ∆w[1]*Fu[i,1,k]  # Fu[i,0,k] = Fu[i,1,k]
                end
            end
        else  # nw == 3; w = z
            # At the locations except for the negative end of the z-direction; unlike for
            # the forward difference, for the backward difference this part of the code is
            # common for both the Bloch and symmetry boundary conditions.
            for k = 2:Nz  # not k = 2:Nz-1
                β = α2 * ∆w′⁻¹[k]
                for j = 1:Ny, i = 1:Nx
                    @inbounds Gv[i,j,k] += β * (∆w[k]*Fu[i,j,k] + ∆w[k-1]*Fu[i,j,k-1])
                end
            end

            if isbloch
                # At the negative end of the z-direction (where the boundary fields are
                # taken from the positive-end boundary)
                β = α2 * ∆w′⁻¹[1]
                for j = 1:Ny, i = 1:Nx
                    @inbounds Gv[i,j,1] += β * (∆w[1]*Fu[i,j,1] + ∆w[Nz]*Fu[i,j,Nz]/e⁻ⁱᵏᴸ)  # Fu[i,j,0] = Fu[i,j,Nz] / exp(-i kz Lz)
                end
            else  # symmetry boundary
                # At the negative end of the z-direction (where the boundary fields are
                # taken from the positive-end boundary)
                β = α * ∆w′⁻¹[1]
                for j = 1:Ny, i = 1:Nx
                    @inbounds Gv[i,j,1] += β * ∆w[1]*Fu[i,j,1]  # Fu[i,j,0] = Fu[i,j,Nz]
                end
            end
        end  # if nw == ...
    end  # if isfwd
end
