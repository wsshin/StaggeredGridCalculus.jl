@testset "divergence" begin

N = [4,5]
M = prod(N)
r = reshape(collect(1:2M), M, 2)'[:]  # index mapping from block matrix to narrowly banded matrix
Z = spzeros(M,M)

F = rand(Complex{Float64}, N..., 2)
g = zeros(Complex{Float64}, N...)
gvec = zeros(Complex{Float64}, M)

@testset "create_divg and apply_divg!" begin
    for ci = CartesianIndices((false:true,false:true))
        # Construct D for a uniform grid and Bloch boundaries.
        isfwd = Vector{Bool}([ci.I...])
        D = create_divg(isfwd, N, order_cmpfirst=false)

        # Test the overall coefficients.
        @test size(D) == (M,2M)
        @test all(any(D.≠0, dims=1))  # no zero columns
        @test all(any(D.≠0, dims=2))  # no zero rows
        @test all(sum(D, dims=1) .== 0)  # all column sums are zero, because each input field to D is used twice, once multiplied with +1 and once with -1
        @test all(sum(D, dims=2) .== 0)  # all row sums are zero, because D * ones(M) = 0
        @test all(sum(abs.(D), dims=1) .== 2)  # each column of D has two nonzero entries, which are ±1's
        @test all(sum(abs.(D), dims=2) .== 4)  # each row of D has six nonzero entries, which are ±1's

        ∂x = (nw = 1; create_∂(nw, isfwd[nw], N))
        ∂y = (nw = 2; create_∂(nw, isfwd[nw], N))
        @test D == [∂x ∂y]

        # Construct D for a nonuniform grid and general boundaries.
        ∆l⁻¹ = rand.(tuple(N...))  # isfwd = true (false) uses ∆l⁻¹ at dual (primal) locations
        isbloch = [true, false]
        e⁻ⁱᵏᴸ = rand(ComplexF64, 2)
        scale∂ = [1, -1]  # +∂x, -∂y

        D = create_divg(isfwd, N, ∆l⁻¹, isbloch, e⁻ⁱᵏᴸ, scale∂=scale∂, order_cmpfirst=false)

        # Test D.
        ∂x = (nw = 1; create_∂(nw, isfwd[nw], N, ∆l⁻¹[nw], isbloch[nw], e⁻ⁱᵏᴸ[nw]))
        ∂y = (nw = 2; create_∂(nw, isfwd[nw], N, ∆l⁻¹[nw], isbloch[nw], e⁻ⁱᵏᴸ[nw]))
        @test D == [scale∂[1].*∂x scale∂[2].*∂y]

        # Test Cartesian-component-first ordering.
        D_cmpfirst = create_divg(isfwd, N, ∆l⁻¹, isbloch, e⁻ⁱᵏᴸ, scale∂=scale∂, order_cmpfirst=true)
        @test D_cmpfirst == D[:,r]

        # Test permutation.
        permute∂ = [2, 1]  # with scale∂ = [1,-1], create divergence operator [-∂y, ∂x]
        D_permute = create_divg(isfwd, N, ∆l⁻¹, isbloch, e⁻ⁱᵏᴸ, permute∂=permute∂, scale∂=scale∂, order_cmpfirst=false)
        @test D_permute == [scale∂[2].*∂y scale∂[1].*∂x]

        D_permute_cmpfirst = create_divg(isfwd, N, ∆l⁻¹, isbloch, e⁻ⁱᵏᴸ, permute∂=permute∂, scale∂=scale∂, order_cmpfirst=true)
        @test D_permute_cmpfirst == D_permute[:,r]

        # Test apply_divg!.
        f = F[:]
        mul!(gvec, D, f)
        g .= 0
        apply_divg!(g, F, isfwd, ∆l⁻¹, isbloch, e⁻ⁱᵏᴸ, scale∂=scale∂)
        @test g[:] ≈ gvec

        mul!(gvec, D_cmpfirst, f[r])
        g .= 0
        apply_divg!(g, F, isfwd, ∆l⁻¹, isbloch, e⁻ⁱᵏᴸ, scale∂=scale∂)
        @test g[:] ≈ gvec

        mul!(gvec, D_permute_cmpfirst, f[r])
        g .= 0
        apply_divg!(g, F, isfwd, ∆l⁻¹, isbloch, e⁻ⁱᵏᴸ, permute∂=permute∂, scale∂=scale∂)
        @test g[:] ≈ gvec
    end
end  # @testset "create_divg and apply_divg!"

@testset "divergence of curl" begin
    # Construct C and D for a uniform grid and periodic boundaries.
    N = [3,4,5]
    M = prod(N)
    isfwd = [true, true, true]  # curl(U) and divg(V) are differentiated forward

    C = create_curl(isfwd, N, order_cmpfirst=false)
    D = create_divg(isfwd, N, order_cmpfirst=false)

    # Construct D * C.
    A = D * C

    # Test Divergence of curl.
    @test size(A) == (M,3M)
    @test all(A.==0)
end  # @testset "divergence of curl"

end  # @testset "divergence"
