export apply_m!, apply_mean!

apply_mean!(G::T,  # output field; G[i,j,k,w] is w-component of G at (i,j,k)
            F::T,  # input field; G[i,j,k,w] is w-component of G at (i,j,k)
            isfwd::AbsVecBool,  # isfwd[w] = true|false for forward|backward averaging
            isbloch::AbsVecBool=fill(true,length(isfwd)),  # boundary conditions in x, y, z
            e⁻ⁱᵏᴸ::AbsVecNumber=ones(length(isfwd));  # Bloch phase factor in x, y, z
            α::Number=1  # scale factor to multiply to result before adding it to G: G += α ∇×F
            ) where {T<:AbsArrNumber{4}} =
    (N = size(F)[nXYZ]; ∆l = ones.((N...,)); apply_mean!(G, F, isfwd, ∆l, ∆l, isbloch, e⁻ⁱᵏᴸ, α=α))

apply_mean!(G::T,  # output field; G[i,j,k,w] is w-component of G at (i,j,k)
            F::T,  # input field; G[i,j,k,w] is w-component of G at (i,j,k)
            isfwd::AbsVecBool,  # isfwd[w] = true|false for forward|backward averaging
            ∆l::Tuple3{AbsVecNumber},  # line segments to multiply with; vectors of length N
            ∆l′::Tuple3{AbsVecNumber},  # line segments to divide by; vectors of length N
            isbloch::AbsVecBool=fill(true,length(isfwd)),  # boundary conditions in x, y, z
            e⁻ⁱᵏᴸ::AbsVecNumber=ones(length(isfwd));  # Bloch phase factor in x, y, z
            α::Number=1  # scale factor to multiply to result before adding it to G: G += α mean(F)
            ) where {T<:AbsArrNumber{4}} =
    (K = length(isfwd); apply_mean!(G, F, SVector{K}(isfwd), ∆l, ∆l′, SVector{K}(isbloch), SVector{K}(e⁻ⁱᵏᴸ), α=α))

# For the implementation, see the comments in matrix/mean.jl.
function apply_mean!(G::T,  # output field; G[i,j,k,w] is w-component of G at (i,j,k)
                     F::T,  # input field; G[i,j,k,w] is w-component of G at (i,j,k)
                     isfwd::SBool{3},  # isfwd[w] = true|false for forward|backward averaging
                     ∆l::Tuple3{AbsVecNumber},  # line segments to multiply with; vectors of length N
                     ∆l′::Tuple3{AbsVecNumber},  # line segments to divide by; vectors of length N
                     isbloch::SBool{3}=SBool{3}(true,true,true),  # boundary conditions in x, y, z
                     e⁻ⁱᵏᴸ::SNumber{3}=SFloat{3}(1,1,1);  # Bloch phase factor in x, y, z
                     α::Number=1  # scale factor to multiply to result before adding it to G: G += α mean(F)
                     ) where {T<:AbsArrNumber{4}}
    indblk = 0  # index of matrix block
    for nv = nXYZ  # Cartesian compotent of output field
        Gv = @view G[:,:,:,nv]  # v-component of output field

        nu = nv  # Cartesian component of input field
        Fu = @view F[:,:,:,nu]  # u-component of input field

        apply_m!(Gv, Fu, nv, isfwd[nv], ∆l[nv], ∆l′[nv], isbloch[nv], e⁻ⁱᵏᴸ[nv], α=α)
        indblk += 1
    end
end

apply_m!(Gv::T,  # v-component of output field (v = x, y, z)
         Fu::T,  # u-component of input field (u = x, y, z)
         nw::Integer,  # 1|2|3 for averaging along x|y|z; 1|2 for averaging along horizontal|vertical
         isfwd::Bool,  # true|false for forward|backward averaging
         ∆w::Number=1.0,  # line segments to multiply with; vector of length N[nw]
         isbloch::Bool=true,  # boundary condition in w-direction
         e⁻ⁱᵏᴸ::Number=1;  # Bloch phase factor
         α::Number=1  # scale factor to multiply to result before adding it to Gv: Gv += α m(Fu)
         ) where {T<:AbsArrNumber{3}} =
    (N = size(Fu); ∆w_vec = fill(∆w, N[nw]); apply_m!(Gv, Fu, nw, isfwd, ∆w_vec, ∆w_vec, isbloch, e⁻ⁱᵏᴸ, α=α))  # fill: create vector of ∆w

function apply_m!(Gv::T,  # v-component of output field (v = x, y, z)
                  Fu::T,  # u-component of input field (u = x, y, z)
                  nw::Integer,  # 1|2|3 for averaging along x|y|z; 1|2 for averaging along horizontal|vertical
                  isfwd::Bool,  # true|false for forward|backward averaging
                  ∆w::AbsVecNumber,  # line segments to multiply with; vector of length N[nw]
                  ∆w′::AbsVecNumber,  # line segments to divide by; vector of length N[nw]
                  isbloch::Bool=true,  # boundary condition in w-direction
                  e⁻ⁱᵏᴸ::Number=1;  # Bloch phase factor
                  α::Number=1  # scale factor to multiply to result before adding it to Gv: Gv += α m(Fu)
                  ) where {T<:AbsArrNumber{3}}
    @assert(size(Gv)==size(Fu))
    @assert(size(Fu,nw)==length(∆w))

    Nx, Ny, Nz = size(Fu)
    α2 = 0.5 * α

    # Make sure not to include branches inside for loops.
    if isfwd
        if nw == nX
            for k = 1:Nz, j = 1:Ny, i = 2:Nx-1
                β = α2 / ∆w′[i]
                @inbounds Gv[i,j,k] += β * (∆w[i+1]*Fu[i+1,j,k] + ∆w[i]*Fu[i,j,k])
            end

            if isbloch
                β = α2 / ∆w′[1]
                for k = 1:Nz, j = 1:Ny
                    @inbounds Gv[1,j,k] += β * (∆w[2]*Fu[2,j,k] + ∆w[1]*Fu[1,j,k])  # nothing special
                end

                β = α2 / ∆w′[Nx]
                for k = 1:Nz, j = 1:Ny
                    @inbounds Gv[Nx,j,k] += β * (∆w[1]*e⁻ⁱᵏᴸ*Fu[1,j,k] + ∆w[Nx]*Fu[Nx,j,k])  # Fu[Nx+1,j,k] = exp(-i kx Lx) * Fu[1,j,k]
                end
            else  # symmetry boundary
                β = α2 / ∆w′[1]
                for k = 1:Nz, j = 1:Ny
                    @inbounds Gv[1,j,k] += β * ∆w[2]*Fu[2,j,k]  # Fu[1,j,k] == 0
                end

                β = α2 / ∆w′[Nx]
                for k = 1:Nz, j = 1:Ny
                    @inbounds Gv[Nx,j,k] += β * ∆w[Nx]*Fu[Nx,j,k]  # Fu[Nx+1,j,k] == 0
                end
            end
        elseif nw == nY
            for k = 1:Nz, j = 2:Ny-1
                β = α2 / ∆w′[j]
                for i = 1:Nx
                    @inbounds Gv[i,j,k] += β * (∆w[j+1]*Fu[i,j+1,k] + ∆w[j]*Fu[i,j,k])
                end
            end

            if isbloch
                β = α2 / ∆w′[1]
                for k = 1:Nz, i = 1:Nx
                    @inbounds Gv[i,1,k] += β * (∆w[2]*Fu[i,2,k] + ∆w[1]*Fu[i,1,k])  # nothing special
                end

                β = α2 / ∆w′[Ny]
                for k = 1:Nz, i = 1:Nx
                    @inbounds Gv[i,Ny,k] += β * (∆w[1]*e⁻ⁱᵏᴸ*Fu[i,1,k] + ∆w[Ny]*Fu[i,Ny,k])  # Fu[i,Ny+1,k] = exp(-i ky Ly) * Fu[i,1,k]
                end
            else  # symmetry boundary
                β = α2 / ∆w′[1]
                for k = 1:Nz, i = 1:Nx
                    @inbounds Gv[i,1,k] += β * ∆w[2]*Fu[i,2,k]  # Fu[i,1,k] == 0
                end

                β = α2 / ∆w′[Ny]
                for k = 1:Nz, i = 1:Nx
                    @inbounds Gv[i,Ny,k] += β * ∆w[Ny]*Fu[i,Ny,k]  # Fu[i,Ny+1,k] == 0
                end
            end
        else  # nw == nZ
            for k = 2:Nz-1
                β = α2 / ∆w′[k]
                for j = 1:Ny, i = 1:Nx
                    @inbounds Gv[i,j,k] += β * (∆w[k+1]*Fu[i,j,k+1] + ∆w[k]*Fu[i,j,k])
                end
            end

            if isbloch
                β = α2 / ∆w′[1]
                for j = 1:Ny, i = 1:Nx
                    @inbounds Gv[i,j,1] += β * (∆w[2]*Fu[i,j,2] + ∆w[1]*Fu[i,j,1])  # nothing special
                end

                β = α2 / ∆w′[Nz]
                for j = 1:Ny, i = 1:Nx
                    @inbounds Gv[i,j,Nz] += β * (∆w[1]*e⁻ⁱᵏᴸ*Fu[i,j,1] + ∆w[Nz]*Fu[i,j,Nz])  # Fu[i,j,Nz+1] = exp(-i kz Lz) * Fu[i,j,1]
                end
            else  # symmetry boundary
                β = α2 / ∆w′[1]
                for j = 1:Ny, i = 1:Nx
                    @inbounds Gv[i,j,1] += β * ∆w[2]*Fu[i,j,2]  # Fu[i,j,1] == 0
                end

                β = α2 / ∆w′[Nz]
                for j = 1:Ny, i = 1:Nx
                    @inbounds Gv[i,j,Nz] += β * ∆w[Nz]*Fu[i,j,Nz]  # Fu[i,j,Nz+1] == 0
                end
            end
        end  # if nw == ...
    else  # backward averaging
        if nw == nX
            for k = 1:Nz, j = 1:Ny, i = 2:Nx  # not i = 2:Nx-1
                β = α2 / ∆w′[i]
                @inbounds Gv[i,j,k] += β * (∆w[i]*Fu[i,j,k] + ∆w[i-1]*Fu[i-1,j,k])
            end

            if isbloch
                β = α2 / ∆w′[1]
                for k = 1:Nz, j = 1:Ny
                    @inbounds Gv[1,j,k] += β * (∆w[1]*Fu[1,j,k] + ∆w[Nx]*Fu[Nx,j,k]/e⁻ⁱᵏᴸ)  # Fu[0,j,k] = Fu[Nx,j,k] / exp(-i kx Lx)
                end
            else  # symmetry boundary
                β = α / ∆w′[1]
                for k = 1:Nz, j = 1:Ny
                    @inbounds Gv[1,j,k] += β * ∆w[1]*Fu[1,j,k]  # Fu[0,j,k] = Fu[1,j,k]
                end
            end
        elseif nw == nY
            for k = 1:Nz, j = 2:Ny  # not j = 2:Ny-1
                β = α2 / ∆w′[j]
                for i = 1:Nx
                    @inbounds Gv[i,j,k] += β * (∆w[j]*Fu[i,j,k] + ∆w[j-1]*Fu[i,j-1,k])
                end
            end

            if isbloch
                β = α2 / ∆w′[1]
                for k = 1:Nz, i = 1:Nx
                    @inbounds Gv[i,1,k] += β * (∆w[1]*Fu[i,1,k] + ∆w[Ny]*Fu[i,Ny,k]/e⁻ⁱᵏᴸ)  # Fu[i,0,k] = Fu[0,Ny,k] / exp(-i ky Ly)
                end
            else  # symmetry boundary
                β = α / ∆w′[1]
                for k = 1:Nz, i = 1:Nx
                    @inbounds Gv[i,1,k] += β * ∆w[1]*Fu[i,1,k]  # Fu[i,0,k] = Fu[i,1,k]
                end
            end
        else  # nw == nZ
            for k = 2:Nz  # not k = 2:Nz-1
                β = α2 / ∆w′[k]
                for j = 1:Ny, i = 1:Nx
                    @inbounds Gv[i,j,k] += β * (∆w[k]*Fu[i,j,k] + ∆w[k-1]*Fu[i,j,k-1])
                end
            end

            if isbloch
                β = α2 / ∆w′[1]
                for j = 1:Ny, i = 1:Nx
                    @inbounds Gv[i,j,1] += β * (∆w[1]*Fu[i,j,1] + ∆w[Nz]*Fu[i,j,Nz]/e⁻ⁱᵏᴸ)  # Fu[i,j,0] = Fu[i,j,Nz] / exp(-i kz Lz)
                end
            else  # symmetry boundary
                β = α / ∆w′[1]
                for j = 1:Ny, i = 1:Nx
                    @inbounds Gv[i,j,1] += β * ∆w[1]*Fu[i,j,1]  # Fu[i,j,0] = Fu[i,j,Nz]
                end
            end
        end  # if nw == ...
    end  # if isfwd
end
