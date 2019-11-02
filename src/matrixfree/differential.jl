export apply_∂!, apply_curl!

# To-dos
# - Test if using a separate index vector (which is an identity map) to iterate over a 3D
# array makes iteration significantly slower.  This is to simulate the case of FEM.


function apply_curl!(G::T,  # output field; G[i,j,k,w] is w-component of G at (i,j,k)
                     F::T,  # input field; G[i,j,k,w] is w-component of G at (i,j,k)
                     isfwd::SVec3Bool,  # isfwd[w] = true|false: create ∂w by forward|backward difference
                     ∆l::Tuple3{AbsVecNumber},  # ∆l[w]: distances between grid planes in x-direction
                     isbloch::SVec3Bool,  # boundary conditions in x, y, z
                     e⁻ⁱᵏᴸ::SVec3Number  # Bloch phase factor in x, y, z
                    ) where {T<:AbsArrNumber{4}}
    for nv = nXYZ  # Cartesian compotent of output field
        Gv = @view G[:,:,:,nv]  # v-component of output field
        parity = 1
        for nw = next2(nv)  # direction of differentiation
            nu = 6 - nv - nw  # Cantesian component of input field; 6 = nX + nY + nZ
            Fu = @view F[:,:,:,nu]  # u-component of input field

            # Need to avoid allocation in parity*∆l[nw]
            apply_∂!(Gv, Fu, nw, isfwd[nw], parity*∆l[nw], isbloch[nw], e⁻ⁱᵏᴸ[nw])  # Gv .+= ∂w Fu
            parity = -1
        end
    end
end

apply_∂!(Gv::T,  # v-component of output field (v = x, y, z)
         Fu::T,  # u-component of input field (u = x, y, z)
         nw::Integer,  # 1|2|3 for x|y|z
         isfwd::Bool,  # true|false for forward|backward difference
         ∆w::Number=1.0,  # spatial discretization; vector of length N[nw]
         isbloch::Bool=true,  # boundary condition in w-direction
         e⁻ⁱᵏᴸ::Number=1.0  # Bloch phase factor
        ) where {T<:AbsArrNumber{3}} =
    (N = size(Fu); apply_∂!(Gv, Fu, nw, isfwd, fill(∆w, N[nw]), isbloch, e⁻ⁱᵏᴸ))  # fill: create vector of ∆w

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
                  e⁻ⁱᵏᴸ::Number=1.0  # Bloch phase factor
                 ) where {T<:AbsArrNumber{3}}
    @assert(size(Gv)==size(Fu))
    @assert(size(Fu,nw)==length(∆w))

    Nx, Ny, Nz = size(Fu)

    # Make sure not to include branches inside for loops.
    if isfwd
        if nw == nX
            for k = 1:Nz, j = 1:Ny, i = 2:Nx-1
                @inbounds Gv[i,j,k] += (Fu[i+1,j,k] - Fu[i,j,k]) / ∆w[i]
            end

            if isbloch
                for k = 1:Nz, j = 1:Ny
                    @inbounds Gv[1,j,k] += (Fu[2,j,k] - Fu[1,j,k]) / ∆w[1]
                end

                for k = 1:Nz, j = 1:Ny
                    @inbounds Gv[Nx,j,k] += (e⁻ⁱᵏᴸ*Fu[1,j,k] - Fu[Nx,j,k]) / ∆w[Nx]
                end
            else  # symmetry boundary
                for k = 1:Nz, j = 1:Ny
                    @inbounds Gv[1,j,k] += Fu[2,j,k] / ∆w[1]  # Fu[1,j,k] == 0
                end

                for k = 1:Nz, j = 1:Ny
                    @inbounds Gv[Nx,j,k] += -Fu[Nx,j,k] / ∆w[Nx]  # Fu[1,j,k] == 0
                end
            end
        elseif nw == nY
            for k = 1:Nz, j = 2:Ny-1, i = 1:Nx
                @inbounds Gv[i,j,k] += (Fu[i,j+1,k] - Fu[i,j,k]) / ∆w[j]
            end

            if isbloch
                for k = 1:Nz, i = 1:Nx
                    @inbounds Gv[i,1,k] += (Fu[i,2,k] - Fu[i,1,k]) / ∆w[1]
                end

                for k = 1:Nz, i = 1:Nx
                    @inbounds Gv[i,Ny,k] += (e⁻ⁱᵏᴸ*Fu[i,1,k] - Fu[i,Ny,k]) / ∆w[Ny]
                end
            else  # symmetry boundary
                for k = 1:Nz, i = 1:Nx
                    @inbounds Gv[i,1,k] += Fu[i,2,k] / ∆w[1]
                end

                for k = 1:Nz, i = 1:Nx
                    @inbounds Gv[i,Ny,k] += -Fu[i,Ny,k] / ∆w[Ny]  # Fu[i,1,k] == 0
                end
            end
        else  # nw == nZ
            for k = 2:Nz-1, j = 1:Ny, i = 1:Nx
                @inbounds Gv[i,j,k] += (Fu[i,j,k+1] - Fu[i,j,k]) / ∆w[k]
            end

            if isbloch
                for j = 1:Ny, i = 1:Nx
                    @inbounds Gv[i,j,1] += (Fu[i,j,2] - Fu[i,j,1]) / ∆w[1]
                end

                for j = 1:Ny, i = 1:Nx
                    @inbounds Gv[i,j,Nz] += (e⁻ⁱᵏᴸ*Fu[i,j,1] - Fu[i,j,Nz]) / ∆w[Nz]
                end
            else  # symmetry boundary
                for j = 1:Ny, i = 1:Nx
                    @inbounds Gv[i,j,1] += Fu[i,j,2] / ∆w[1]
                end

                for j = 1:Ny, i = 1:Nx
                    @inbounds Gv[i,j,Nz] += -Fu[i,j,Nz] / ∆w[Nz]  # Fu[i,j,1] == 0
                end
            end
        end  # if nw == ...
    else  # backward difference
        if nw == nX
            for k = 1:Nz, j = 1:Ny, i = 2:Nx
                @inbounds Gv[i,j,k] += (Fu[i,j,k] - Fu[i-1,j,k]) / ∆w[i]
            end

            if isbloch
                for k = 1:Nz, j = 1:Ny
                    @inbounds Gv[1,j,k] += (Fu[1,j,k] - Fu[Nx,j,k]/e⁻ⁱᵏᴸ) / ∆w[1]
                end
            else  # symmetry boundary
                # Do nothing, because the derivative of dual fields at the symmetry boundary
                # is zero.
            end
        elseif nw == nY
            for k = 1:Nz, j = 2:Ny, i = 1:Nx
                @inbounds Gv[i,j,k] += (Fu[i,j,k] - Fu[i,j-1,k]) / ∆w[j]
            end

            if isbloch
                for k = 1:Nz, i = 1:Nx
                    @inbounds Gv[i,1,k] += (Fu[i,1,k] - Fu[i,Ny,k]/e⁻ⁱᵏᴸ) / ∆w[1]
                end
            else  # symmetry boundary
                # Do nothing, because the derivative of dual fields at the symmetry boundary
                # is zero.
            end
        else  # nw == nZ
            for k = 2:Nz, j = 1:Ny, i = 1:Nx
                @inbounds Gv[i,j,k] += (Fu[i,j,k] - Fu[i,j,k-1]) / ∆w[k]
            end

            if isbloch
                for j = 1:Ny, i = 1:Nx
                    @inbounds Gv[i,j,1] += (Fu[i,j,1] - Fu[i,j,Nz]/e⁻ⁱᵏᴸ) / ∆w[1]
                end
            else  # symmetry boundary
                # Do nothing, because the derivative of dual fields at the symmetry boundary
                # is zero.
            end
        end  # if nw == ...
    end  # if isfwd
end
