@testset "partial" begin

@testset "create_∂ and apply_∂!, 1D" begin
    N = (10,)
    M = prod(N)

    Fᵤ = rand(N...)
    Gᵥ = similar(Fᵤ)
    gᵥ = zeros(M)

    Mask = similar(Fᵤ)  # masking array
    for nw = 1:1
        sub = Vector{Int}(undef, 1)
        sub′ = Vector{Int}(undef, 1)

        Nw = N[nw]
        ∆w⁻¹ = rand(Nw)

        for ns = (-1,1), isbloch = (true,false)  # (backward,forward difference), (Bloch,symmetric)
            ∂ws = spzeros(M,M)

            for ind = 1:M
                sub .= CartesianIndices(N)[ind].I  # subscripts of diagonal entry
                indw = sub[nw]
                ∆wᵢ⁻¹ = ∆w⁻¹[indw]
                ∂ws[ind,ind] = -ns * ∆wᵢ⁻¹  # diagonal entries

                # Calculate the column index of the off-diagonal entry in the row `ind`.
                sub′ .= sub  # subscripts of off-diagonal entry
                if ns == 1  # forward difference
                    if sub′[nw] == Nw
                        sub′[nw] = 1
                    else
                        sub′[nw] += 1
                    end
                else  # backward difference
                    if sub′[nw] == 1
                        sub′[nw] = Nw
                    else
                        sub′[nw] -= 1
                    end
                end

                ind′ = LinearIndices(N)[sub′...]
                ∂ws[ind, ind′] += ns * ∆wᵢ⁻¹  # off-diagonal entry
            end

            if !isbloch  # symmetry boundary
                # Initialize the masking array.
                Mask .= 1
                Mask[Base.setindex(axes(Mask), 1, nw)...] = 0  # not .= 0 unlike code for 2D and 3D
                if ns == 1  # forward difference
                    # The input fields at the symmetry boundary should be zero, so apply the
                    # mask to the input field.
                    ∂ws = ∂ws * spdiagm(0=>Mask[:])
                else  # backward difference
                    # The output fields at the symmetry boundary should be zero, so apply
                    # mask to the output field.
                    ∂ws = spdiagm(0=>Mask[:]) * ∂ws
                end
            end

            # Test create_∂.
            @test create_∂(nw, ns==1, [N...], ∆w⁻¹, isbloch) == ∂ws

            # Test apply_∂!.
            fᵤ = Fᵤ[:]
            mul!(gᵥ, ∂ws, fᵤ)
            apply_∂!(Gᵥ, Fᵤ, Val(:(=)), nw, ns==1, ∆w⁻¹, isbloch)
            @test Gᵥ[:] ≈ gᵥ

            # print("matrix: "); @btime mul!($gᵥ, $∂ws, $fᵤ)
            # print("matrix-free: "); @btime apply_∂!($Gᵥ, $Fᵤ, $nw, $ns==1, $∆w⁻¹)
            # println()
        end
    end
end  # @testset "create_∂ and apply_∂!, 1D"

@testset "create_∂ and apply_∂!, 2D" begin
    N = (8,9)
    M = prod(N)

    Fᵤ = rand(N...)
    Gᵥ = similar(Fᵤ)
    gᵥ = zeros(M)

    Mask = similar(Fᵤ)  # masking array
    for nw = 1:2
        sub = Vector{Int}(undef, 2)
        sub′ = Vector{Int}(undef, 2)

        Nw = N[nw]
        ∆w⁻¹ = rand(Nw)

        for ns = (-1,1), isbloch = (true,false)  # (backward,forward difference), (Bloch,symmetric)
            ∂ws = spzeros(M,M)

            for ind = 1:M
                sub .= CartesianIndices(N)[ind].I  # subscripts of diagonal entry
                indw = sub[nw]
                ∆wᵢ⁻¹ = ∆w⁻¹[indw]
                ∂ws[ind,ind] = -ns * ∆wᵢ⁻¹  # diagonal entries

                # Calculate the column index of the off-diagonal entry in the row `ind`.
                sub′ .= sub  # subscripts of off-diagonal entry
                if ns == 1  # forward difference
                    if sub′[nw] == Nw
                        sub′[nw] = 1
                    else
                        sub′[nw] += 1
                    end
                else  # backward difference
                    if sub′[nw] == 1
                        sub′[nw] = Nw
                    else
                        sub′[nw] -= 1
                    end
                end

                ind′ = LinearIndices(N)[sub′...]
                ∂ws[ind, ind′] += ns * ∆wᵢ⁻¹  # off-diagonal entry
            end

            if !isbloch  # symmetry boundary
                # Initialize the masking array.
                Mask .= 1
                Mask[Base.setindex(axes(Mask), 1, nw)...] .= 0
                if ns == 1  # forward difference
                    # The input fields at the symmetry boundary should be zero, so apply the
                    # mask to the input field.
                    ∂ws = ∂ws * spdiagm(0=>Mask[:])
                else  # backward difference
                    # The output fields at the symmetry boundary should be zero, so apply
                    # mask to the output field.
                    ∂ws = spdiagm(0=>Mask[:]) * ∂ws
                end
            end

            # Test create_∂.
            @test create_∂(nw, ns==1, [N...], ∆w⁻¹, isbloch) == ∂ws

            # Test apply_∂!.
            fᵤ = Fᵤ[:]
            mul!(gᵥ, ∂ws, fᵤ)
            apply_∂!(Gᵥ, Fᵤ, Val(:(=)), nw, ns==1, ∆w⁻¹, isbloch)
            @test Gᵥ[:] ≈ gᵥ

            # print("matrix: "); @btime mul!($gᵥ, $∂ws, $fᵤ)
            # print("matrix-free: "); @btime apply_∂!($Gᵥ, $Fᵤ, $nw, $ns==1, $∆w⁻¹)
            # println()
        end
    end
end  # @testset "create_∂ and apply_∂!, 2D"

@testset "create_∂ and apply_∂!, 3D" begin
    N = (8,9,10)
    M = prod(N)

    Fᵤ = rand(N...)
    Gᵥ = similar(Fᵤ)
    gᵥ = zeros(M)

    Mask = similar(Fᵤ)  # masking array
    for nw = 1:3
        sub = Vector{Int}(undef, 3)
        sub′ = Vector{Int}(undef, 3)

        Nw = N[nw]
        ∆w⁻¹ = rand(Nw)

        for ns = (-1,1), isbloch = (true,false)  # (backward,forward difference), (Bloch,symmetric)
            ∂ws = spzeros(M,M)

            for ind = 1:M
                sub .= CartesianIndices(N)[ind].I  # subscripts of diagonal entry
                indw = sub[nw]
                ∆wᵢ⁻¹ = ∆w⁻¹[indw]
                ∂ws[ind,ind] = -ns * ∆wᵢ⁻¹  # diagonal entries

                # Calculate the column index of the off-diagonal entry in the row `ind`.
                sub′ .= sub  # subscripts of off-diagonal entry
                if ns == 1  # forward difference
                    if sub′[nw] == Nw
                        sub′[nw] = 1
                    else
                        sub′[nw] += 1
                    end
                else  # backward difference
                    if sub′[nw] == 1
                        sub′[nw] = Nw
                    else
                        sub′[nw] -= 1
                    end
                end

                ind′ = LinearIndices(N)[sub′...]
                ∂ws[ind, ind′] += ns * ∆wᵢ⁻¹  # off-diagonal entry
            end

            if !isbloch  # symmetry boundary
                # Initialize the masking array.
                Mask .= 1
                Mask[Base.setindex(axes(Mask), 1, nw)...] .= 0
                if ns == 1  # forward difference
                    # The input fields at the symmetry boundary should be zero, so apply the
                    # mask to the input field.
                    ∂ws = ∂ws * spdiagm(0=>Mask[:])
                else  # backward difference
                    # The output fields at the symmetry boundary should be zero, so apply
                    # mask to the output field.
                    ∂ws = spdiagm(0=>Mask[:]) * ∂ws
                end
            end

            # Test create_∂.
            @test create_∂(nw, ns==1, [N...], ∆w⁻¹, isbloch) == ∂ws

            # Test apply_∂!.
            fᵤ = Fᵤ[:]
            mul!(gᵥ, ∂ws, fᵤ)
            apply_∂!(Gᵥ, Fᵤ, Val(:(=)), nw, ns==1, ∆w⁻¹, isbloch)
            @test Gᵥ[:] ≈ gᵥ

            # print("matrix: "); @btime mul!($gᵥ, $∂ws, $fᵤ)
            # print("matrix-free: "); @btime apply_∂!($Gᵥ, $Fᵤ, $nw, $ns==1, $∆w⁻¹)
            # println()
        end
    end
end  # @testset "create_∂ and apply_∂!, 3D"

end  # @testset "differential"
