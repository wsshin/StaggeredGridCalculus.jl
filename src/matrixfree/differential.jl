export apply_∂!, apply_curl!

# To-dos
# - Test if using a separate index vector (which is an identity map) to iterate over a 3D
# array makes iteration significantly slower.  This is to simulate the case of FEM.
apply_curl!(G::T,  # output field; G[i,j,k,w] is w-component of G at (i,j,k)
            F::T,  # input field; G[i,j,k,w] is w-component of G at (i,j,k)
            isfwd::AbsVecBool,  # isfwd[w] = true|false: create ∂w by forward|backward difference
            isbloch::AbsVecBool=fill(true,length(isfwd)),  # boundary conditions in x, y, z
            e⁻ⁱᵏᴸ::AbsVecNumber=ones(length(isfwd));  # Bloch phase factor in x, y, z
            α::Number=1  # scale factor to multiply to result before adding it to G: G += α ∇×F
            ) where {T<:AbsArrNumber{4}} =
    (∆l = ones.(size(F)[nXYZ]); apply_curl!(G, F, isfwd, ∆l, isbloch, e⁻ⁱᵏᴸ, α=α))

apply_curl!(G::T,  # output field; G[i,j,k,w] is w-component of G at (i,j,k)
            F::T,  # input field; G[i,j,k,w] is w-component of G at (i,j,k)
            isfwd::AbsVecBool,  # isfwd[w] = true|false: create ∂w by forward|backward difference
            ∆l::Tuple3{AbsVecNumber},  # ∆l[w]: distances between grid planes in x-direction
            isbloch::AbsVecBool=fill(true,length(isfwd)),  # boundary conditions in x, y, z
            e⁻ⁱᵏᴸ::AbsVecNumber=ones(length(isfwd));  # Bloch phase factor in x, y, z
            α::Number=1  # scale factor to multiply to result before adding it to G: G += α ∇×F
            ) where {T<:AbsArrNumber{4}} =
    # I should not cast e⁻ⁱᵏᴸ into a complex vector, because then the entire curl matrix
    # becomes a complex matrix.  Sometimes I want to keep it real (e.g., when no PML and
    # Bloch phase factors are used).  In fact, this is the reason why I accept e⁻ⁱᵏᴸ instead
    # of constructing it from k and L as exp.(-im .* k .* L), which is always complex even
    # if k = 0.
    #
    # I should not cast ∆l to a vector of any specific type (e.g., Float, CFloat), either,
    # because sometimes I would want to even create an integral curl operator.
    (K = length(isfwd); apply_curl!(G, F, SVector{K}(isfwd), ∆l, SVector{K}(isbloch), SVector{K}(e⁻ⁱᵏᴸ), α=α))

function apply_curl!(G::T,  # output field; G[i,j,k,w] is w-component of G at (i,j,k)
                     F::T,  # input field; G[i,j,k,w] is w-component of G at (i,j,k)
                     isfwd::SBool{3},  # isfwd[w] = true|false: create ∂w by forward|backward difference
                     ∆l::Tuple3{AbsVecNumber},  # ∆l[w]: distances between grid planes in x-direction
                     isbloch::SBool{3}=SVector(true,true,true),  # boundary conditions in x, y, z
                     e⁻ⁱᵏᴸ::SNumber{3}=SVector(1.0,1.0,1.0);  # Bloch phase factor in x, y, z
                     α::Number=1  # scale factor to multiply to result before adding it to G: G += α ∇×F
                     ) where {T<:AbsArrNumber{4}}
    for nv = nXYZ  # Cartesian compotent of output field
        Gv = @view G[:,:,:,nv]  # v-component of output field
        parity = 1
        for nw = next2(nv)  # direction of differentiation
            nu = 6 - nv - nw  # Cantesian component of input field; 6 = nX + nY + nZ
            Fu = @view F[:,:,:,nu]  # u-component of input field

            # Need to avoid allocation in parity*∆l[nw]
            apply_∂!(Gv, Fu, nw, isfwd[nw], ∆l[nw], isbloch[nw], e⁻ⁱᵏᴸ[nw], α=parity*α)  # Gv += α (±∂Fu/∂w)
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
