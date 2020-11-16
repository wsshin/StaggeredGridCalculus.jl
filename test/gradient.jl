@testset "gradient" begin

N = (3,4,5)
M = prod(N)
r = reshape(collect(1:3M), M, 3)'[:]  # index mapping from block matrix to narrowly banded matrix
Z = spzeros(M,M)

F = rand(Complex{Float64}, N..., 3)
G = similar(F)
g = zeros(Complex{Float64}, 3M)

@testset "create_grad and apply_grad! to generate primal field U" begin
    # Construct uG for a uniform grid and Bloch boundaries.
    isfwd = [true, true, true]  # to generate U, scalar is differentiated forward
    uG = create_grad(isfwd, [N...], reorder=false)

    # Test the overall coefficients.
    @test size(uG) == (3M,M)
    @test all(any(uG.≠0, dims=1))  # no zero columns
    @test all(any(uG.≠0, dims=2))  # no zero rows
    @test all(sum(uG, dims=1) .== 0)  # all column sums are zero, because each input field to uG is used twice in each Cartesian direction, once multiplied with +1 and once with -1
    @test all(sum(uG, dims=2) .== 0)  # all row sums are zero, because uG * ones(M) = 0
    @test all(sum(abs.(uG), dims=1) .== 6)  # each column of uG has six nonzero entries, which are ±1's
    @test all(sum(abs.(uG), dims=2) .== 2)  # each row of uG has two nonzero entries, which are ±1's

    ∂x = (nw = 1; create_∂(nw, isfwd[nw], [N...]))
    ∂y = (nw = 2; create_∂(nw, isfwd[nw], [N...]))
    ∂z = (nw = 3; create_∂(nw, isfwd[nw], [N...]))
    @test uG == [∂x; ∂y; ∂z]

    # Construct uG for a nonuniform grid and general boundaries.
    ∆lprim = rand.(N)
    isbloch = [true, false, false]
    e⁻ⁱᵏᴸ = rand(ComplexF64, 3)
    parity = [1, -1, -1]

    uG = create_grad(isfwd, [N...], ∆lprim, isbloch, e⁻ⁱᵏᴸ, parity=parity, reorder=false)

    # Test Cu.
    ∂x = (nw = 1; create_∂(nw, isfwd[nw], [N...], ∆lprim[nw], isbloch[nw], e⁻ⁱᵏᴸ[nw]))
    ∂y = (nw = 2; create_∂(nw, isfwd[nw], [N...], ∆lprim[nw], isbloch[nw], e⁻ⁱᵏᴸ[nw]))
    ∂z = (nw = 3; create_∂(nw, isfwd[nw], [N...], ∆lprim[nw], isbloch[nw], e⁻ⁱᵏᴸ[nw]))
    @test uG == [parity[1].*∂x; parity[2].*∂y; parity[3].*∂z]

    # Test reordering.
    uG_reorder = create_grad(isfwd, [N...], ∆lprim, isbloch, e⁻ⁱᵏᴸ, parity=parity, reorder=true)
    @test uG_reorder == uG[r,:]

    # Test permutation.
    uG_permute = create_grad(isfwd, [N...], ∆lprim, isbloch, e⁻ⁱᵏᴸ, parity=parity, vpermute=[2,1,3], reorder=false)
    @test uG_permute == [parity[1].*∂y; parity[2].*∂x; parity[3].*∂z]

    uG_permute_reorder = create_grad(isfwd, [N...], ∆lprim, isbloch, e⁻ⁱᵏᴸ, parity=parity, vpermute=[2,1,3], reorder=true)
    @test uG_permute_reorder == uG_permute[r,:]

    # Test apply_grad!.
    # # to be filled
end  # @testset "create_grad and apply_grad! for primal field U"

# @testset "create_grad and apply_grad! for dual field V" begin
#     # To be filled
# end  # @testset "create_grad and apply_grad! for dual field V"

@testset "curl of gradient" begin
    # Construct Cu and uG for a uniform grid and periodic boundaries.
    isfwd = [true, true, true]  # curl(U) and gradient to generate U are differentiated forward
    isbloch = [true, false, false]

    Cu = create_curl(isfwd, [N...], reorder=false)
    uG = create_grad(isfwd, [N...], reorder=false)

    # Construct Dv * Cu.
    A = Cu * uG

    # Test Divergence of curl.
    @test size(A) == (3M,M)
    @test all(A.==0)
end  # @testset "curl of gradent"

end  # @testset "gradient"
