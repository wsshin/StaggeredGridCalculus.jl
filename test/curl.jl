@testset "curl" begin

N = (3,4,5)
M = prod(N)
r = reshape(collect(1:3M), M, 3)'[:]  # index mapping from block matrix to narrowly banded matrix
Z = spzeros(M,M)

F = rand(Complex{Float64}, N..., 3)
G = similar(F)
g = zeros(Complex{Float64}, 3M)

@testset "create_curl and apply_curl! for primal field U" begin
    # Construct Cu for a uniform grid and Bloch boundaries.
    isfwd = [true, true, true]  # U is differentiated forward
    Cu = create_curl(isfwd, [N...], reorder=false)

    # Test the overall coefficients.
    @test size(Cu) == (3M,3M)
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
end  # @testset "create_curl and apply_curl! for primal field U"

@testset "create_curl and apply_curl! for dual field V" begin
    # Construct Cv for a uniform grid and Bloch boundaries.
    isfwd = [false, false, false]  # V is differentiated backward
    Cv = create_curl(isfwd, [N...], reorder=false)

    # Test the overall coefficients.
    @test size(Cv) == (3M,3M)
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
end  # @testset "create_curl and apply_curl! for dual field V"

@testset "curl of curl" begin
    # Construct Cu and Cv for a uniform grid and Bloch boundaries.
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

    # Construct Cv * Cu.
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
    # Construct Cu and Cv for a uniform grid and Bloch boundaries.
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

    # Construct Cv * Cu.
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
end  # @testset "curl of curl, mixed forward and backward"

end  # @testset "curl"
