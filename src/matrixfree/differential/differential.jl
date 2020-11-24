export apply_∂!

apply_∂!(Gv::T,  # v-component of output field (v = x, y, z)
         Fu::T,  # u-component of input field (u = x, y, z)
         nw::Integer,  # 1|2|3 for x|y|z
         isfwd::Bool,  # true|false for forward|backward difference
         ∆w::Number=1.0,  # spatial discretization; vector of length N[nw]
         isbloch::Bool=true,  # boundary condition in w-direction
         e⁻ⁱᵏᴸ::Number=1.0;  # Bloch phase factor
         α::Number=1  # scale factor to multiply to result before adding it to Gv: Gv += α ∂Fu/∂w
         ) where {T<:AbsArrNumber{3}} =
    (N = size(Fu); apply_∂!(Gv, Fu, nw, isfwd, fill(∆w, N[nw]), isbloch, e⁻ⁱᵏᴸ, α=α))  # fill: create vector of ∆w

# The field arrays Fu (and Gv) represents a 3D array of a specific Cartesian component of the
# field, and indexed as Fu[i,j,k], where (i,j,k) is the grid cell location.
# This function adds the derivatives to the existing values of Gv.  Therefore, if you want
# to get the derivative values themselves, pass Gv initialized with zeros.
function apply_∂!(Gv::T,  # v-component of output field (v = x, y, z)
                  Fu::T,  # u-component of input field (u = x, y, z)
                  nw::Integer,  # 1|2|3 for x|y|z
                  isfwd::Bool,  # true|false for forward|backward difference
                  ∆w::AbsVecNumber,  # spatial discretization; vector of length N[nw]
                  isbloch::Bool=true,  # boundary condition in w-direction
                  e⁻ⁱᵏᴸ::Number=1;  # Bloch phase factor: L = Lw
                  α::Number=1  # scale factor to multiply to result before adding it to Gv: Gv += α ∂Fu/∂w
                  ) where {T<:AbsArrNumber{3}}
    @assert(size(Gv)==size(Fu))
    @assert(size(Fu,nw)==length(∆w))

    Nx, Ny, Nz = size(Fu)

    # Make sure not to include branches inside for loops.
    if isfwd
        if nw == nX
            for k = 1:Nz, j = 1:Ny, i = 2:Nx-1
                @inbounds Gv[i,j,k] += (α / ∆w[i]) * (Fu[i+1,j,k] - Fu[i,j,k])
            end

            if isbloch
                β = α / ∆w[1]
                for k = 1:Nz, j = 1:Ny
                    @inbounds Gv[1,j,k] += β * (Fu[2,j,k] - Fu[1,j,k])  # nothing special
                end

                β = α / ∆w[Nx]
                for k = 1:Nz, j = 1:Ny
                    @inbounds Gv[Nx,j,k] += β * (e⁻ⁱᵏᴸ*Fu[1,j,k] - Fu[Nx,j,k])  # Fu[Nx+1,j,k] = exp(-i kx Lx) * Fu[1,j,k]
                end
            else  # symmetry boundary
                β = α / ∆w[1]
                for k = 1:Nz, j = 1:Ny
                    @inbounds Gv[1,j,k] += β * Fu[2,j,k]  # Fu[1,j,k] == 0
                end

                β = α / ∆w[Nx]
                for k = 1:Nz, j = 1:Ny
                    @inbounds Gv[Nx,j,k] -= β * Fu[Nx,j,k]  # Fu[Nx+1,j,k] == 0
                end
            end
        elseif nw == nY
            for k = 1:Nz, j = 2:Ny-1
                β = α / ∆w[j]
                for i = 1:Nx
                    @inbounds Gv[i,j,k] += β * (Fu[i,j+1,k] - Fu[i,j,k])
                end
            end

            if isbloch
                β = α / ∆w[1]
                for k = 1:Nz, i = 1:Nx
                    @inbounds Gv[i,1,k] += β * (Fu[i,2,k] - Fu[i,1,k])  # nothing special
                end

                β = α / ∆w[Ny]
                for k = 1:Nz, i = 1:Nx
                    @inbounds Gv[i,Ny,k] += β * (e⁻ⁱᵏᴸ*Fu[i,1,k] - Fu[i,Ny,k])  # Fu[i,Ny+1,k] = exp(-i ky Ly) * Fu[i,1,k]
                end
            else  # symmetry boundary
                β = α / ∆w[1]
                for k = 1:Nz, i = 1:Nx
                    @inbounds Gv[i,1,k] += β * Fu[i,2,k]  # Fu[i,1,k] == 0
                end

                β = α / ∆w[Ny]
                for k = 1:Nz, i = 1:Nx
                    @inbounds Gv[i,Ny,k] -= β * Fu[i,Ny,k]  # Fu[i,Ny+1,k] == 0
                end
            end
        else  # nw == nZ
            for k = 2:Nz-1
                β = α / ∆w[k]
                for j = 1:Ny, i = 1:Nx
                    @inbounds Gv[i,j,k] += β * (Fu[i,j,k+1] - Fu[i,j,k])
                end
            end

            if isbloch
                β = α / ∆w[1]
                for j = 1:Ny, i = 1:Nx
                    @inbounds Gv[i,j,1] += β * (Fu[i,j,2] - Fu[i,j,1])  # nothing special
                end

                β = α / ∆w[Nz]
                for j = 1:Ny, i = 1:Nx
                    @inbounds Gv[i,j,Nz] += β * (e⁻ⁱᵏᴸ*Fu[i,j,1] - Fu[i,j,Nz])  # Fu[i,j,Nz+1] = exp(-i kz Lz) * Fu[i,j,1]
                end
            else  # symmetry boundary
                β = α / ∆w[1]
                for j = 1:Ny, i = 1:Nx
                    @inbounds Gv[i,j,1] += β * Fu[i,j,2]  # Fu[i,j,1] == 0
                end

                β = α / ∆w[Nz]
                for j = 1:Ny, i = 1:Nx
                    @inbounds Gv[i,j,Nz] -= β * Fu[i,j,Nz]  # Fu[i,j,Nz+1] == 0
                end
            end
        end  # if nw == ...
    else  # backward difference
        if nw == nX
            for k = 1:Nz, j = 1:Ny, i = 2:Nx  # not i = 2:Nx-1
                @inbounds Gv[i,j,k] += (α / ∆w[i]) * (Fu[i,j,k] - Fu[i-1,j,k])
            end

            if isbloch
                β = α / ∆w[1]
                for k = 1:Nz, j = 1:Ny
                    @inbounds Gv[1,j,k] += β * (Fu[1,j,k] - Fu[Nx,j,k]/e⁻ⁱᵏᴸ)  # Fu[0,j,k] = Fu[Nx,j,k] / exp(-i kx Lx)
                end
            else  # symmetry boundary
                # Do nothing, because the derivative of dual fields at the symmetry boundary
                # is zero.
            end
        elseif nw == nY
            for k = 1:Nz, j = 2:Ny  # not j = 2:Ny-1
                β = α / ∆w[j]
                for i = 1:Nx
                    @inbounds Gv[i,j,k] += β * (Fu[i,j,k] - Fu[i,j-1,k])
                end
            end

            if isbloch
                β = α / ∆w[1]
                for k = 1:Nz, i = 1:Nx
                    @inbounds Gv[i,1,k] += β * (Fu[i,1,k] - Fu[i,Ny,k]/e⁻ⁱᵏᴸ)  # Fu[i,0,k] = Fu[i,Ny,k] / exp(-i ky Ly)
                end
            else  # symmetry boundary
                # Do nothing, because the derivative of dual fields at the symmetry boundary
                # is zero.
            end
        else  # nw == nZ
            for k = 2:Nz  # not k = 2:Nz-1
                β = α / ∆w[k]
                for j = 1:Ny, i = 1:Nx
                    @inbounds Gv[i,j,k] += β * (Fu[i,j,k] - Fu[i,j,k-1])
                end
            end

            if isbloch
                β = α / ∆w[1]
                for j = 1:Ny, i = 1:Nx
                    @inbounds Gv[i,j,1] += β * (Fu[i,j,1] - Fu[i,j,Nz]/e⁻ⁱᵏᴸ)  # Fu[i,j,0] = Fu[i,j,Nz] / exp(-i kz Lz)
                end
            else  # symmetry boundary
                # Do nothing, because the derivative of dual fields at the symmetry boundary
                # is zero.
            end
        end  # if nw == ...
    end  # if isfwd
end

include("curl.jl")
# include("divergence.jl")
# include("gradient.jl")
