@testset "mean" begin

@testset "create_m, 1D" begin
    @test isa(create_m(nX, true, [10]), Any)
    @test isa(create_m(nX, false, [10]), Any)
end

@testset "create_mean and apply_mean! 3D" begin
    N = SVector(8,9,10)
    M = prod(N)

    Fu = rand(N...)
    Gv = similar(Fu)
    gv = zeros(M)

    Mask = similar(Fu)  # masking array

    K = length(N)
    KM = K * M
    F = rand(Complex{Float64}, N..., K)
    f = F[:]
    G = similar(F)
    g = zeros(Complex{Float64}, KM)

    for isfwd = (true,false), isbloch = (true,false)  # (backward,forward difference), (Bloch,symmetric)
        ∆l = rand.(tuple(N...))
        ∆l′ = rand.(tuple(N...))
        Mdiag = spzeros(KM,KM)

        for nw = 1:K
            sub_on = Vector{Int}(undef, K)
            sub_off = Vector{Int}(undef, K)

            Nw = N[nw]
            ∆w = ∆l[nw]
            ∆w′ = ∆l′[nw]

            Mws = spzeros(M,M)

            for ind_on = 1:M
                sub_on .= CartesianIndices(N.data)[ind_on].I  # subscripts of diagonal entry
                indw_on = sub_on[nw]
                ∆w_on = ∆w[indw_on]
                ∆w′_on = ∆w′[indw_on]
                Mws[ind_on,ind_on] = 0.5 * ∆w_on / ∆w′_on  # diagonal entries

                # Calculate the column index of the off-diagonal entry in the row `ind`.
                sub_off .= sub_on  # subscripts of off-diagonal entry
                if isfwd  # forward difference
                    if sub_off[nw] == Nw
                        sub_off[nw] = 1
                    else
                        sub_off[nw] += 1
                    end
                else  # backward difference
                    if sub_off[nw] == 1
                        if isbloch
                            sub_off[nw] = Nw
                        else  # symmetry boundary
                            # Leave sub_off[nw] at 1
                        end
                    else
                        sub_off[nw] -= 1
                    end
                end

                indw_off = sub_off[nw]
                ∆w_off = ∆w[indw_off]
                ind_off = LinearIndices(N.data)[sub_off...]  # index of off-diagonal entry
                if !isfwd && !isbloch && sub_on[nw]==1  # bacward difference && symmetry boundary at negative end
                    Mws[ind_on, ind_off] = ∆w_off / ∆w′_on  # double diagonal entry
                else
                    Mws[ind_on, ind_off] = 0.5 * ∆w_off / ∆w′_on  # off-diagonal entry
                end
            end

            if isfwd && !isbloch  # forward difference and symmetry boundary
                # Initialize the masking array.
                Mask .= 1
                Mask[Base.setindex(axes(Mask), 1, nw)...] .= 0

                # The input fields at the symmetry boundary should be zero, so apply the
                # mask to the input field.
                Mws = Mws * spdiagm(0=>Mask[:])
            end

            # Test create_m.
            @test create_m(nw, isfwd, N, ∆w, ∆w′, isbloch) == Mws

            # Test apply_m!.
            fu = Fu[:]
            mul!(gv, Mws, fu)
            Gv .= 0
            apply_m!(Gv, Fu, nw, isfwd, ∆w, ∆w′, isbloch)
            @test Gv[:] ≈ gv

            # Construct Mdiag, Msup, Msub.
            Is = 1+(nw-1)*M:nw*M
            Mdiag[Is,Is] .= Mws
        end  # for nw
        isfwdK = fill(isfwd, K)
        isblochK = fill(isbloch, K)
        @test create_mean(isfwdK, N, ∆l, ∆l′, isblochK, reorder=false) == Mdiag

        # Test apply_mean!.
        mul!(g, Mdiag, f); G .= 0; apply_mean!(G, F, isfwdK, ∆l, ∆l′, isblochK); @test G[:] ≈ g
    end  # isfwd = ..., isbloch = ...
end  # @testset "create_mean and apply_mean! 3D"

@testset "create_mean and apply_mean! 2D" begin
    N = SVector(8,9)
    M = prod(N)

    Fu = rand(N...)
    Gv = similar(Fu)
    gv = zeros(M)

    Mask = similar(Fu)  # masking array

    K = length(N)
    KM = K * M
    F = rand(Complex{Float64}, N..., K)
    f = F[:]
    G = similar(F)
    g = zeros(Complex{Float64}, KM)

    for isfwd = (true,false), isbloch = (true,false)  # (backward,forward difference), (Bloch,symmetric)
        ∆l = rand.(tuple(N...))
        ∆l′ = rand.(tuple(N...))
        Mdiag = spzeros(KM,KM)

        for nw = 1:K
            sub_on = Vector{Int}(undef, K)
            sub_off = Vector{Int}(undef, K)

            Nw = N[nw]
            ∆w = ∆l[nw]
            ∆w′ = ∆l′[nw]

            Mws = spzeros(M,M)

            for ind_on = 1:M
                sub_on .= CartesianIndices(N.data)[ind_on].I  # subscripts of diagonal entry
                indw_on = sub_on[nw]
                ∆w_on = ∆w[indw_on]
                ∆w′_on = ∆w′[indw_on]
                Mws[ind_on,ind_on] = 0.5 * ∆w_on / ∆w′_on  # diagonal entries

                # Calculate the column index of the off-diagonal entry in the row `ind`.
                sub_off .= sub_on  # subscripts of off-diagonal entry
                if isfwd  # forward difference
                    if sub_off[nw] == Nw
                        sub_off[nw] = 1
                    else
                        sub_off[nw] += 1
                    end
                else  # backward difference
                    if sub_off[nw] == 1
                        if isbloch
                            sub_off[nw] = Nw
                        else  # symmetry boundary
                            # Leave sub_off[nw] at 1
                        end
                    else
                        sub_off[nw] -= 1
                    end
                end

                indw_off = sub_off[nw]
                ∆w_off = ∆w[indw_off]
                ind_off = LinearIndices(N.data)[sub_off...]  # index of off-diagonal entry
                if !isfwd && !isbloch && sub_on[nw]==1  # bacward difference && symmetry boundary at negative end
                    Mws[ind_on, ind_off] = ∆w_off / ∆w′_on  # double diagonal entry
                else
                    Mws[ind_on, ind_off] = 0.5 * ∆w_off / ∆w′_on  # off-diagonal entry
                end
            end

            if isfwd && !isbloch  # forward difference and symmetry boundary
                # Initialize the masking array.
                Mask .= 1
                Mask[Base.setindex(axes(Mask), 1, nw)...] .= 0

                # The input fields at the symmetry boundary should be zero, so apply the
                # mask to the input field.
                Mws = Mws * spdiagm(0=>Mask[:])
            end

            # Test create_m.
            @test create_m(nw, isfwd, N, ∆w, ∆w′, isbloch) == Mws

            # Test apply_m!.
            fu = Fu[:]
            mul!(gv, Mws, fu)
            Gv .= 0
            # apply_m!(Gv, Fu, nw, isfwd, ∆w, ∆w′, isbloch)
            # @test Gv[:] ≈ gv

            # Construct Mdiag, Msup, Msub.
            Is = 1+(nw-1)*M:nw*M
            Mdiag[Is,Is] .= Mws
        end  # for nw
        isfwdK = fill(isfwd, K)
        isblochK = fill(isbloch, K)
        @test create_mean(isfwdK, N, ∆l, ∆l′, isblochK, reorder=false) == Mdiag

        # Test apply_mean!.
        # mul!(g, Mdiag, f); G .= 0; apply_mean!(G, F, isfwdK, ∆l, ∆l′, isblochK); @test G[:] ≈ g
    end  # isfwd = ..., isbloch = ...
end  # @testset "create_mean and apply_mean! 2D"

# @testset "create_mean" begin
# create_mean(isfwd::AbsVecBool,  # isfwd[w] = true|false for forward|backward averaging
#             N::AbsVecInteger,  # size of grid
#             isbloch::AbsVecBool=fill(true,length(N)),  # boundary conditions in x, y, z
#             e⁻ⁱᵏᴸ::AbsVecNumber=ones(length(N));  # Bloch phase factor in x, y, z
#             kdiag::Integer=0,  # 0|+1|-1 for diagonal|superdiagonal|subdiagonal of material parameter
#             reorder::Bool=true) =  # true for more tightly banded matrix
#
#     N = SVector(8,9,10)
#     # N = SVector(3,3,3)
#     M = prod(N)
#     for nw = nXYZ
#     # for nw = (nX,)
#         Nw = N[nw]
#         sub′ = Vector{Int}(undef, 3)
#
#         for isfwd = (true, false)
#         # for isfwd = (true,)
#             Mws = spzeros(M,M)
#
#             for ind = 1:M
#                 Mws[ind,ind] = 0.5  # diagonal entries
#
#                 # Calculate the column index of the off-diagonal entry in the row `ind`.
#                 sub′ .= CartesianIndices(N.data)[ind].I  # subscripts of off-diagonal entry
#                 if isfwd  # forward difference
#                     if sub′[nw] == Nw
#                         sub′[nw] = 1
#                     else
#                         sub′[nw] += 1
#                     end
#                 else  # isfwd = false (backward difference)
#                     if sub′[nw] == 1
#                         sub′[nw] = Nw
#                     else
#                         sub′[nw] -= 1
#                     end
#                 end
#
#                 ind′ = LinearIndices(N.data)[sub′...]  # index of off-diagonal entry
#                 Mws[ind, ind′] = 0.5  # off-diagonal entry
#             end
#             @test create_m(nw, isfwd, N) == Mws
#         end
#     end
#
# end  # @testset "create_mean"

end  # @testset "mean"
