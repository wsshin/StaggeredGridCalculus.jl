@testset "gradient" begin

N = [4,5]
M = prod(N)
r = reshape(collect(1:2M), M, 2)'[:]  # index mapping from block matrix to narrowly banded matrix
Z = spzeros(M,M)

F = rand(Complex{Float64}, N..., 2)
G = similar(F)
g = zeros(Complex{Float64}, 2M)

@testset "create_grad and apply_grad! to generate primal field U" begin
    for ci = CartesianIndices((false:true,false:true))
        # Construct G for a uniform grid and Bloch boundaries.
        isfwd = Vector{Bool}([ci.I...])
        G = create_grad(isfwd, N, order_cmpfirst=false)

        # Test the overall coefficients.
        @test size(G) == (2M,M)
        @test all(any(G.≠0, dims=1))  # no zero columns
        @test all(any(G.≠0, dims=2))  # no zero rows
        @test all(sum(G, dims=1) .== 0)  # all column sums are zero, because each input field to G is used twice in each Cartesian direction, once multiplied with +1 and once with -1
        @test all(sum(G, dims=2) .== 0)  # all row sums are zero, because G * ones(M) = 0
        @test all(sum(abs.(G), dims=1) .== 4)  # each column of G has six nonzero entries, which are ±1's
        @test all(sum(abs.(G), dims=2) .== 2)  # each row of G has two nonzero entries, which are ±1's

        ∂x = (nw = 1; create_∂(nw, isfwd[nw], N))
        ∂y = (nw = 2; create_∂(nw, isfwd[nw], N))
        @test G == [∂x; ∂y]

        # Construct G for a nonuniform grid and general boundaries.
        ∆l⁻¹ = rand.(tuple(N...))  # isfwd = true (false) uses ∆l⁻¹ at dual (primal) locations
        isbloch = [true, false]
        e⁻ⁱᵏᴸ = rand(ComplexF64, 2)
        scale∂ = [1, -1]

        G = create_grad(isfwd, N, ∆l⁻¹, isbloch, e⁻ⁱᵏᴸ, scale∂=scale∂, order_cmpfirst=false)

        # Test G.
        ∂x = (nw = 1; create_∂(nw, isfwd[nw], N, ∆l⁻¹[nw], isbloch[nw], e⁻ⁱᵏᴸ[nw]))
        ∂y = (nw = 2; create_∂(nw, isfwd[nw], N, ∆l⁻¹[nw], isbloch[nw], e⁻ⁱᵏᴸ[nw]))
        @test G == [scale∂[1].*∂x; scale∂[2].*∂y]

        # Test Cartesian-component-first ordering.
        G_cmpfirst = create_grad(isfwd, N, ∆l⁻¹, isbloch, e⁻ⁱᵏᴸ, scale∂=scale∂, order_cmpfirst=true)
        @test G_cmpfirst == G[r,:]

        # Test permutation.
        permute∂ = [2, 1]  # with scale∂ = [1,-1], create divergence operator [-∂y; ∂x]
        G_permute = create_grad(isfwd, N, ∆l⁻¹, isbloch, e⁻ⁱᵏᴸ, permute∂=permute∂, scale∂=scale∂, order_cmpfirst=false)
        @test G_permute == [scale∂[2].*∂y; scale∂[1].*∂x]

        G_permute_cmpfirst = create_grad(isfwd, N, ∆l⁻¹, isbloch, e⁻ⁱᵏᴸ, permute∂=permute∂, scale∂=scale∂, order_cmpfirst=true)
        @test G_permute_cmpfirst == G_permute[r,:]

        # Test apply_grad!.
        # # to be filled
    end
end  # @testset "create_grad and apply_grad! for primal field U"

# @testset "create_grad and apply_grad! for dual field V" begin
#     # To be filled
# end  # @testset "create_grad and apply_grad! for dual field V"

@testset "curl of gradient" begin
    # Construct C and G for a uniform grid and periodic boundaries.
    N = [3,4,5]
    M = prod(N)
    isfwd = [true, true, true]  # curl(U) and gradient to generate U are differentiated forward

    C = create_curl(isfwd, N, order_cmpfirst=false)
    G = create_grad(isfwd, N, order_cmpfirst=false)

    # Construct C * G
    A = C * G

    # Test Divergence of curl.
    @test size(A) == (3M,M)
    @test all(A.==0)
end  # @testset "curl of gradent"

end  # @testset "gradient"
