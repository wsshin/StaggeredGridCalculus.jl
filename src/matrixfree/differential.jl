export apply_∂!, apply_curl!

# To-dos
# - Test if using a separate index vector (which is an identity map) to iterate over a 3D
# array makes iteration significantly slower.  This is to simulate the case of FEM.


# The field arrays F (and G) represents a 3D array of a specific Cartesian component of the
# field, and indexed as F[i,j,k], where (i,j,k) is the grid cell location.
function apply_∂!(G::T,  # output field
                  F::T,  # input field
                  nw::Integer,  # 1|2|3 for x|y|z
                  isfwd::Bool,  # true|false for forward|backward difference
                  ∆w::AbsVecNumber,  # spatial discretization; vector of length N[nw]
                  isbloch::Bool=true,  # boundary condition in w-direction
                  e⁻ⁱᵏᴸ::Number=1  # Bloch phase factor
                  ) where {T<:AbsArrNumber{3}}
    @assert(size(G)==size(F))
    @assert(size(F,nw)==length(∆w))

    Nx, Ny, Nz = size(F)

    # Make sure not to include branches inside for loops.
    if isfwd
        if nw == nX
            for k = 1:Nz, j = 1:Ny, i = 1:Nx-1
                @inbounds G[i,j,k] = (F[i+1,j,k] - F[i,j,k]) / ∆w[i]
            end

            if isbloch
                for k = 1:Nz, j = 1:Ny
                    @inbounds G[Nx,j,k] = (e⁻ⁱᵏᴸ*F[1,j,k] - F[Nx,j,k]) / ∆w[Nx]
                end
            else  # symmetry boundary
                for k = 1:Nz, j = 1:Ny
                    @inbounds G[Nx,j,k] = -F[Nx,j,k] / ∆w[Nx]  # F[1,j,k] == 0
                end
            end
        elseif nw == nY
            for k = 1:Nz, j = 1:Ny-1, i = 1:Nx
                @inbounds G[i,j,k] = (F[i,j+1,k] - F[i,j,k]) / ∆w[j]
            end

            if isbloch
                for k = 1:Nz, i = 1:Nx
                    @inbounds G[i,Ny,k] = (e⁻ⁱᵏᴸ*F[i,1,k] - F[i,Ny,k]) / ∆w[Ny]
                end
            else  # symmetry boundary
                for k = 1:Nz, i = 1:Nx
                    @inbounds G[i,Ny,k] = -F[i,Ny,k] / ∆w[Ny]  # F[i,1,k] == 0
                end
            end
        else  # nw == nZ
            for k = 1:Nz-1, j = 1:Ny, i = 1:Nx
                @inbounds G[i,j,k] = (F[i,j,k+1] - F[i,j,k]) / ∆w[k]
            end

            if isbloch
                for j = 1:Ny, i = 1:Nx
                    @inbounds G[i,j,Nz] = (e⁻ⁱᵏᴸ*F[i,j,1] - F[i,j,Nz]) / ∆w[Nz]
                end
            else  # symmetry boundary
                for j = 1:Ny, i = 1:Nx
                    @inbounds G[i,j,Nz] = -F[i,j,Nz] / ∆w[Nz]  # F[i,j,1] == 0
                end
            end
        end  # if nw == ...
    else  # backward difference
        if nw == nX
            for k = 1:Nz, j = 1:Ny, i = 2:Nx
                @inbounds G[i,j,k] = (F[i,j,k] - F[i-1,j,k]) / ∆w[i]
            end

            if isbloch
                for k = 1:Nz, j = 1:Ny
                    @inbounds G[1,j,k] = (F[1,j,k] - F[Nx,j,k]/e⁻ⁱᵏᴸ) / ∆w[1]
                end
            else  # symmetry boundary
                for k = 1:Nz, j = 1:Ny
                    @inbounds G[1,j,k] = 2F[1,j,k] / ∆w[1]  # F[0,j,k] == -F[1,j,k]
                end
            end
        elseif nw == nY
            for k = 1:Nz, j = 2:Ny, i = 1:Nx
                @inbounds G[i,j,k] = (F[i,j,k] - F[i,j-1,k]) / ∆w[j]
            end

            if isbloch
                for k = 1:Nz, i = 1:Nx
                    @inbounds G[i,1,k] = (F[i,1,k] - F[i,Ny,k]/e⁻ⁱᵏᴸ) / ∆w[1]
                end
            else  # symmetry boundary
                for k = 1:Nz, i = 1:Nx
                    @inbounds G[i,1,k] = 2F[i,1,k] / ∆w[1]  # F[i,0,k] == -F[i,1,k]
                end
            end
        else  # nw == nZ
            for k = 2:Nz, j = 1:Ny, i = 1:Nx
                @inbounds G[i,j,k] = (F[i,j,k] - F[i,j,k-1]) / ∆w[k]
            end

            if isbloch
                for j = 1:Ny, i = 1:Nx
                    @inbounds G[i,j,1] = (F[i,j,1] - F[i,j,Nz]/e⁻ⁱᵏᴸ) / ∆w[1]
                end
            else  # symmetry boundary
                for j = 1:Ny, i = 1:Nx
                    @inbounds G[i,j,1] = 2F[i,j,1] / ∆w[1]  # F[i,j,0] == -F[i,j,1]
                end
            end
        end  # if nw == ...
    end  # if isfwd
end
