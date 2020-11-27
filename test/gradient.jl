@testset "gradient" begin

N = [4,5]
M = prod(N)
r = reshape(collect(1:2M), M, 2)'[:]  # index mapping from block matrix to narrowly banded matrix
Z = spzeros(M,M)

F = rand(Complex{Float64}, N..., 2)
G = similar(F)
g = zeros(Complex{Float64}, 2M)

@testset "create_grad and apply_grad! to generate primal field U" begin
    # Construct uG for a uniform grid and Bloch boundaries.
    isfwd = [true, true]  # to generate U, scalar is differentiated forward
    uG = create_grad(isfwd, N, order_cmpfirst=false)

    # Test the overall coefficients.
    @test size(uG) == (2M,M)
    @test all(any(uG.≠0, dims=1))  # no zero columns
    @test all(any(uG.≠0, dims=2))  # no zero rows
    @test all(sum(uG, dims=1) .== 0)  # all column sums are zero, because each input field to uG is used twice in each Cartesian direction, once multiplied with +1 and once with -1
    @test all(sum(uG, dims=2) .== 0)  # all row sums are zero, because uG * ones(M) = 0
    @test all(sum(abs.(uG), dims=1) .== 4)  # each column of uG has six nonzero entries, which are ±1's
    @test all(sum(abs.(uG), dims=2) .== 2)  # each row of uG has two nonzero entries, which are ±1's

    ∂x = (nw = 1; create_∂(nw, isfwd[nw], N))
    ∂y = (nw = 2; create_∂(nw, isfwd[nw], N))
    @test uG == [∂x; ∂y]

    # Construct uG for a nonuniform grid and general boundaries.
    ∆lprim⁻¹ = rand.(tuple(N...))
    isbloch = [true, false]
    e⁻ⁱᵏᴸ = rand(ComplexF64, 2)
    scale∂ = [1, -1]

    uG = create_grad(isfwd, N, ∆lprim⁻¹, isbloch, e⁻ⁱᵏᴸ, scale∂=scale∂, order_cmpfirst=false)

    # Test Cu.
    ∂x = (nw = 1; create_∂(nw, isfwd[nw], N, ∆lprim⁻¹[nw], isbloch[nw], e⁻ⁱᵏᴸ[nw]))
    ∂y = (nw = 2; create_∂(nw, isfwd[nw], N, ∆lprim⁻¹[nw], isbloch[nw], e⁻ⁱᵏᴸ[nw]))
    @test uG == [scale∂[1].*∂x; scale∂[2].*∂y]

    # Test Cartesian-component-first ordering.
    uG_compfirst = create_grad(isfwd, N, ∆lprim⁻¹, isbloch, e⁻ⁱᵏᴸ, scale∂=scale∂, order_cmpfirst=true)
    @test uG_compfirst == uG[r,:]

    # Test permutation.
    permute∂ = [2, 1]  # with scale∂ = [1,-1], create divergence operator [-∂y; ∂x]
    uG_permute = create_grad(isfwd, N, ∆lprim⁻¹, isbloch, e⁻ⁱᵏᴸ, permute∂=permute∂, scale∂=scale∂, order_cmpfirst=false)
    @test uG_permute == [scale∂[2].*∂y; scale∂[1].*∂x]

    uG_permute_compfirst = create_grad(isfwd, N, ∆lprim⁻¹, isbloch, e⁻ⁱᵏᴸ, permute∂=permute∂, scale∂=scale∂, order_cmpfirst=true)
    @test uG_permute_compfirst == uG_permute[r,:]

    # Test apply_grad!.
    # # to be filled
end  # @testset "create_grad and apply_grad! for primal field U"

# @testset "create_grad and apply_grad! for dual field V" begin
#     # To be filled
# end  # @testset "create_grad and apply_grad! for dual field V"

@testset "curl of gradient" begin
    # Construct Cu and uG for a uniform grid and periodic boundaries.
    N = [3,4,5]
    M = prod(N)
    isfwd = [true, true, true]  # curl(U) and gradient to generate U are differentiated forward

    Cu = create_curl(isfwd, N, order_cmpfirst=false)
    uG = create_grad(isfwd, N, order_cmpfirst=false)

    # Construct Dv * Cu.
    A = Cu * uG

    # Test Divergence of curl.
    @test size(A) == (3M,M)
    @test all(A.==0)
end  # @testset "curl of gradent"

end  # @testset "gradient"
