@testset "mean" begin

@testset "create_mean and apply_mean! 1D" begin
    N = SVector(10,)
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
        ∆l′⁻¹ = rand.(tuple(N...))
        Mdiag = spzeros(KM,KM)

        for nw = 1:K
            sub_on = Vector{Int}(undef, K)
            sub_off = Vector{Int}(undef, K)

            Nw = N[nw]
            ∆w = ∆l[nw]
            ∆w′⁻¹ = ∆l′⁻¹[nw]

            Mws = spzeros(M,M)

            for ind_on = 1:M
                sub_on .= CartesianIndices(N.data)[ind_on].I  # subscripts of diagonal entry
                indw_on = sub_on[nw]
                ∆w_on = ∆w[indw_on]
                ∆w′_on⁻¹ = ∆w′⁻¹[indw_on]
                Mws[ind_on,ind_on] = 0.5 * ∆w_on * ∆w′_on⁻¹  # diagonal entries

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
                    Mws[ind_on, ind_off] = ∆w_off * ∆w′_on⁻¹  # double diagonal entry
                else
                    Mws[ind_on, ind_off] = 0.5 * ∆w_off * ∆w′_on⁻¹  # off-diagonal entry
                end
            end

            if isfwd && !isbloch  # forward difference and symmetry boundary
                # Initialize the masking array.
                Mask .= 1
                Mask[Base.setindex(axes(Mask), 1, nw)...] = 0  # not .= 0 unlike code for 2D and 3D

                # The input fields at the symmetry boundary should be zero, so apply the
                # mask to the input field.
                Mws = Mws * spdiagm(0=>Mask[:])
            end

            # Test create_m.
            @test create_m(nw, isfwd, N, ∆w, ∆w′⁻¹, isbloch) == Mws

            # Test apply_m!.
            fu = Fu[:]
            mul!(gv, Mws, fu)
            apply_m!(Gv, Fu, Val(:(=)), nw, isfwd, ∆w, ∆w′⁻¹, isbloch)
            @test Gv[:] ≈ gv

            # Construct Mdiag, Msup, Msub.
            Is = 1+(nw-1)*M:nw*M
            Mdiag[Is,Is] .= Mws
        end  # for nw
        isfwdK = fill(isfwd, K)
        isblochK = fill(isbloch, K)
        @test create_mean(isfwdK, N, ∆l, ∆l′⁻¹, isblochK, order_cmpfirst=false) == Mdiag

        # Test apply_mean!.
        mul!(g, Mdiag, f)
        apply_mean!(G, F, Val(:(=)), isfwdK, ∆l, ∆l′⁻¹, isblochK)
        @test G[:] ≈ g
    end  # isfwd = ..., isbloch = ...
end  # @testset "create_mean and apply_mean! 1D"

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
        ∆l′⁻¹ = rand.(tuple(N...))
        Mdiag = spzeros(KM,KM)

        for nw = 1:K
            sub_on = Vector{Int}(undef, K)
            sub_off = Vector{Int}(undef, K)

            Nw = N[nw]
            ∆w = ∆l[nw]
            ∆w′⁻¹ = ∆l′⁻¹[nw]

            Mws = spzeros(M,M)

            for ind_on = 1:M
                sub_on .= CartesianIndices(N.data)[ind_on].I  # subscripts of diagonal entry
                indw_on = sub_on[nw]
                ∆w_on = ∆w[indw_on]
                ∆w′_on⁻¹ = ∆w′⁻¹[indw_on]
                Mws[ind_on,ind_on] = 0.5 * ∆w_on * ∆w′_on⁻¹  # diagonal entries

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
                    Mws[ind_on, ind_off] = ∆w_off * ∆w′_on⁻¹  # double diagonal entry
                else
                    Mws[ind_on, ind_off] = 0.5 * ∆w_off * ∆w′_on⁻¹  # off-diagonal entry
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
            @test create_m(nw, isfwd, N, ∆w, ∆w′⁻¹, isbloch) == Mws

            # Test apply_m!.
            fu = Fu[:]
            mul!(gv, Mws, fu)
            apply_m!(Gv, Fu, Val(:(=)), nw, isfwd, ∆w, ∆w′⁻¹, isbloch)
            @test Gv[:] ≈ gv

            # Construct Mdiag, Msup, Msub.
            Is = 1+(nw-1)*M:nw*M
            Mdiag[Is,Is] .= Mws
        end  # for nw
        isfwdK = fill(isfwd, K)
        isblochK = fill(isbloch, K)
        @test create_mean(isfwdK, N, ∆l, ∆l′⁻¹, isblochK, order_cmpfirst=false) == Mdiag

        # Test apply_mean!.
        mul!(g, Mdiag, f)
        apply_mean!(G, F, Val(:(=)), isfwdK, ∆l, ∆l′⁻¹, isblochK)
        @test G[:] ≈ g
    end  # isfwd = ..., isbloch = ...
end  # @testset "create_mean and apply_mean! 2D"

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
        ∆l′⁻¹ = rand.(tuple(N...))
        Mdiag = spzeros(KM,KM)

        for nw = 1:K
            sub_on = Vector{Int}(undef, K)
            sub_off = Vector{Int}(undef, K)

            Nw = N[nw]
            ∆w = ∆l[nw]
            ∆w′⁻¹ = ∆l′⁻¹[nw]

            Mws = spzeros(M,M)

            for ind_on = 1:M
                sub_on .= CartesianIndices(N.data)[ind_on].I  # subscripts of diagonal entry
                indw_on = sub_on[nw]
                ∆w_on = ∆w[indw_on]
                ∆w′_on⁻¹ = ∆w′⁻¹[indw_on]
                Mws[ind_on,ind_on] = 0.5 * ∆w_on * ∆w′_on⁻¹  # diagonal entries

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
                    Mws[ind_on, ind_off] = ∆w_off * ∆w′_on⁻¹  # double diagonal entry
                else
                    Mws[ind_on, ind_off] = 0.5 * ∆w_off * ∆w′_on⁻¹  # off-diagonal entry
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
            @test create_m(nw, isfwd, N, ∆w, ∆w′⁻¹, isbloch) == Mws

            # Test apply_m!.
            fu = Fu[:]
            mul!(gv, Mws, fu)
            apply_m!(Gv, Fu, Val(:(=)), nw, isfwd, ∆w, ∆w′⁻¹, isbloch)
            @test Gv[:] ≈ gv

            # Construct Mdiag, Msup, Msub.
            Is = 1+(nw-1)*M:nw*M
            Mdiag[Is,Is] .= Mws
        end  # for nw
        isfwdK = fill(isfwd, K)
        isblochK = fill(isbloch, K)
        @test create_mean(isfwdK, N, ∆l, ∆l′⁻¹, isblochK, order_cmpfirst=false) == Mdiag

        # Test apply_mean!.
        mul!(g, Mdiag, f)
        apply_mean!(G, F, Val(:(=)), isfwdK, ∆l, ∆l′⁻¹, isblochK)
        @test G[:] ≈ g
    end  # isfwd = ..., isbloch = ...
end  # @testset "create_mean and apply_mean! 3D"

end  # @testset "mean"
