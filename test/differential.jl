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

N = (3,4,5)
M = prod(N)
r = reshape(collect(1:3M), M, 3)'[:]  # index mapping from block matrix to narrowly banded matrix
Z = spzeros(M,M)

F = rand(Complex{Float64}, N..., 3)
G = similar(F)
g = zeros(Complex{Float64}, 3M)

@testset "create_curl and apply_curl! for primal field U" begin
    # Construct Cu for a uniform grid and BLOCH boundaries.
    isfwd = [true, true, true]  # U is differentiated forward
    Cu = create_curl(isfwd, [N...], reorder=false)

    # Test the overall coefficients.
    @test all(any(Cu.≠0, dims=1))  # no zero columns
    @test all(any(Cu.≠0, dims=2))  # no zero rows
    @test all(sum(Cu, dims=2) .== 0)  # all row sums are zero, because Cu * ones(M) = 0

    ∂x = (nw = 1; create_∂(nw, isfwd[nw], [N...]))
    ∂y = (nw = 2; create_∂(nw, isfwd[nw], [N...]))
    ∂z = (nw = 3; create_∂(nw, isfwd[nw], [N...]))
    @test Cu == [Z -∂z ∂y;
                 ∂z Z -∂x;
                 -∂y ∂x Z]

    # Construct Cu for a nonuniform grid and general boundaries.
    ∆ldual = rand.(N)
    isbloch = [true, false, false]
    e⁻ⁱᵏᴸ = rand(ComplexF64, 3)

    Cu = create_curl(isfwd, [N...], ∆ldual, isbloch, e⁻ⁱᵏᴸ, reorder=false)

    # Test Cu.
    ∂x = (nw = 1; create_∂(nw, isfwd[nw], [N...], ∆ldual[nw], isbloch[nw], e⁻ⁱᵏᴸ[nw]))
    ∂y = (nw = 2; create_∂(nw, isfwd[nw], [N...], ∆ldual[nw], isbloch[nw], e⁻ⁱᵏᴸ[nw]))
    ∂z = (nw = 3; create_∂(nw, isfwd[nw], [N...], ∆ldual[nw], isbloch[nw], e⁻ⁱᵏᴸ[nw]))
    @test Cu == [Z -∂z ∂y;
                 ∂z Z -∂x;
                 -∂y ∂x Z]

    # Test reordering.
    Cu_reorder = create_curl(isfwd, [N...], ∆ldual, isbloch, e⁻ⁱᵏᴸ, reorder=true)
    @test Cu_reorder == Cu[r,r]

    # Test apply_curl!.
    f = F[:]
    mul!(g, Cu, f)
    G .= 0
    apply_curl!(G, F, isfwd, ∆ldual, isbloch, e⁻ⁱᵏᴸ)
    @test G[:] ≈ g

    # print("matrix: "); @btime mul!($g, $Cu, $f)
    # print("matrix-free: "); @btime apply_curl!($G, $F, $isfwd, $∆ldual, $isbloch, $e⁻ⁱᵏᴸ)
end  # @testset "create_curl for U"

@testset "create_curl for dual field V" begin
    # Construct Cv for a uniform grid and BLOCH boundaries.
    isfwd = [false, false, false]  # V is differentiated backward
    Cv = create_curl(isfwd, [N...], reorder=false)

    # Test the overall coefficients.
    @test all(any(Cv.≠0, dims=1))  # no zero columns
    @test all(any(Cv.≠0, dims=2))  # no zero rows
    @test all(sum(Cv, dims=2) .== 0)  # all row sums are zero, because Cv * ones(sum(Min)) = 0

    ∂x = (nw = 1; create_∂(nw, isfwd[nw], [N...]))
    ∂y = (nw = 2; create_∂(nw, isfwd[nw], [N...]))
    ∂z = (nw = 3; create_∂(nw, isfwd[nw], [N...]))
    @test Cv == [Z -∂z ∂y;
                 ∂z Z -∂x;
                 -∂y ∂x Z]

    # Construct Cv for a nonuniform grid and general boundaries.
    ∆lprim = rand.(N)
    isbloch = [true, false, false]
    e⁻ⁱᵏᴸ = rand(ComplexF64, 3)

    Cv = create_curl(isfwd, [N...], ∆lprim, isbloch, e⁻ⁱᵏᴸ, reorder=false)

    # Test Cv.
    ∂x = (nw = 1; create_∂(nw, isfwd[nw], [N...], ∆lprim[nw], isbloch[nw], e⁻ⁱᵏᴸ[nw]))
    ∂y = (nw = 2; create_∂(nw, isfwd[nw], [N...], ∆lprim[nw], isbloch[nw], e⁻ⁱᵏᴸ[nw]))
    ∂z = (nw = 3; create_∂(nw, isfwd[nw], [N...], ∆lprim[nw], isbloch[nw], e⁻ⁱᵏᴸ[nw]))
    @test Cv == [Z -∂z ∂y;
                 ∂z Z -∂x;
                 -∂y ∂x Z]

    # Test reordering
    Cv_reorder = create_curl(isfwd, [N...], ∆lprim, isbloch, e⁻ⁱᵏᴸ, reorder=true)
    @test Cv_reorder == Cv[r,r]

    # Test apply_curl!.
    f = F[:]
    mul!(g, Cv, f)
    G .= 0
    apply_curl!(G, F, isfwd, ∆lprim, isbloch, e⁻ⁱᵏᴸ)
    @test G[:] ≈ g
end  # @testset "create_curl for V"

@testset "curl of curl" begin
    # Construct Cu and Cv for a uniform grid and BLOCH boundaries.
    ∆ldual = ones.(N)
    ∆lprim = ones.(N)
    isbloch = [true, false, false]
    e⁻ⁱᵏᴸ = ones(3)

    Cu = create_curl([true,true,true], [N...], ∆ldual, isbloch, e⁻ⁱᵏᴸ, reorder=false)
    Cv = create_curl([false,false,false], [N...], ∆lprim, isbloch, e⁻ⁱᵏᴸ, reorder=false)

    # Test symmetry of each block.
    for i = nXYZ
        for j = next2(i)
            -Cv[(i-1)*M+1:i*M,(j-1)*M+1:j*M]' == Cu[(i-1)*M+1:i*M,(j-1)*M+1:j*M]
        end
    end

    # Construct Cv * Cu for all BLOCH.
    isbloch = fill(true, 3)
    Cu = create_curl([true,true,true], [N...], ∆ldual, isbloch, e⁻ⁱᵏᴸ, reorder=false)
    Cv = create_curl([false,false,false], [N...], ∆lprim, isbloch, e⁻ⁱᵏᴸ, reorder=false)
    A = Cv * Cu

    # Test curl of curl.
    @test all(diag(A) .== 4)  # all diagonal entries are 4
    @test all(sum(A.≠0, dims=2) .== 13)  # 13 nonzero entries per row
    @test A == A'  # Hermitian

    B = A - 4I
    @test all(abs.(B[B.≠0]).==1)  # all nonzero off-diagonal entries are ±1
end  # @testset "curl of curl"

@testset "curl of curl, mixed forward and backward" begin
    # Construct Cu and Cv for a uniform grid and BLOCH boundaries.
    ∆ldual = ones.(N)
    ∆lprim = ones.(N)
    isbloch = [true, false, false]
    e⁻ⁱᵏᴸ = ones(3)

    isfwd = [true,false,true]
    Cu = create_curl(isfwd, [N...], ∆ldual, isbloch, e⁻ⁱᵏᴸ, reorder=false)
    Cv = create_curl(.!isfwd, [N...], ∆lprim, isbloch, e⁻ⁱᵏᴸ, reorder=false)

    # Test symmetry of each block.
    for i = nXYZ
        for j = next2(i)
            -Cv[(i-1)*M+1:i*M,(j-1)*M+1:j*M]' == Cu[(i-1)*M+1:i*M,(j-1)*M+1:j*M]
        end
    end

    # Construct Cv * Cu for all BLOCH.
    isbloch = fill(true, 3)
    Cu = create_curl(isfwd, [N...], ∆ldual, isbloch, e⁻ⁱᵏᴸ, reorder=false)
    Cv = create_curl(.!isfwd, [N...], ∆lprim, isbloch, e⁻ⁱᵏᴸ, reorder=false)
    A = Cv * Cu

    # Test curl of curl.
    @test all(diag(A) .== 4)  # all diagonal entries are 4
    @test all(sum(A.≠0, dims=2) .== 13)  # 13 nonzero entries per row
    @test A == A'  # Hermitian

    B = A - 4I
    @test all(abs.(B[B.≠0]).==1)  # all nonzero off-diagonal entries are ±1
end  # @testset "curl of curl"

end  # @testset "differential"
