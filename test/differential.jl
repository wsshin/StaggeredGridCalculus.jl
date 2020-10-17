@testset "differential" begin

@testset "create_∂, 1D" begin
    @test isa(create_∂(nX, true, [10]), Any)
    @test isa(create_∂(nX, false, [10]), Any)
end

@testset "create_∂ and apply_∂!, 3D" begin
    N = (8,9,10)
    M = prod(N)

    Fu = rand(N...)
    Gv = similar(Fu)
    gv = zeros(M)

    Mask = similar(Fu)  # masking array
    for nw = nXYZ
        sub = Vector{Int}(undef, 3)
        sub′ = Vector{Int}(undef, 3)

        Nw = N[nw]
        ∆w = rand(Nw)

        for ns = (-1,1), isbloch = (true,false)  # (backward,forward difference), (Bloch,symmetric)
            ∂ws = spzeros(M,M)

            for ind = 1:M
                sub .= CartesianIndices(N)[ind].I  # subscripts of diagonal entry
                indw = sub[nw]
                ∆wᵢ = ∆w[indw]
                ∂ws[ind,ind] = -ns / ∆wᵢ  # diagonal entries

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
                ∂ws[ind, ind′] += ns / ∆wᵢ  # off-diagonal entry
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
            @test create_∂(nw, ns==1, [N...], ∆w, isbloch) == ∂ws

            # Test apply_∂!.
            fu = Fu[:]
            mul!(gv, ∂ws, fu)
            Gv .= 0
            apply_∂!(Gv, Fu, nw, ns==1, ∆w, isbloch)
            @test Gv[:] ≈ gv

            # print("matrix: "); @btime mul!($gv, $∂ws, $fu)
            # print("matrix-free: "); @btime apply_∂!($Gv, $Fu, $nw, $ns==1, $∆w)
            # println()
        end
    end
end  # @testset "create_∂"

end  # @testset "differential"
