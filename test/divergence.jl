@testset "divergence" begin

N = [4,5]
M = prod(N)
r = reshape(collect(1:2M), M, 2)'[:]  # index mapping from block matrix to narrowly banded matrix
Z = spzeros(M,M)

F = rand(ComplexF64, N..., 2)  # input array of vector field
g = zeros(ComplexF64, N...)  # output array of scalar
gvec = zeros(ComplexF64, M)  # column vector representation of g

@testset "create_divg and apply_divg!" begin
    for ci = CartesianIndices((false:true,false:true))
        # Construct ∇̇ for a uniform grid and Bloch boundaries.
        isfwd = Vector{Bool}([ci.I...])
        ∇̇ = create_divg(isfwd, N, order_cmpfirst=false)

        # Test the overall coefficients.
        @test size(∇̇) == (M,2M)
        @test all(any(∇̇.≠0, dims=1))  # no zero columns
        @test all(any(∇̇.≠0, dims=2))  # no zero rows
        @test all(sum(∇̇, dims=1) .== 0)  # all column sums are zero, because each input field to ∇̇ is used twice, once multiplied with +1 and once with -1
        @test all(sum(∇̇, dims=2) .== 0)  # all row sums are zero, because ∇̇ * ones(M) = 0
        @test all(sum(abs.(∇̇), dims=1) .== 2)  # each column of ∇̇ has two nonzero entries, which are ±1's
        @test all(sum(abs.(∇̇), dims=2) .== 4)  # each row of ∇̇ has six nonzero entries, which are ±1's

        ∂x = (nw = 1; create_∂(nw, isfwd[nw], N))
        ∂y = (nw = 2; create_∂(nw, isfwd[nw], N))
        @test ∇̇ == [∂x ∂y]

        # Construct ∇̇ for a nonuniform grid and general boundaries.
        ∆l⁻¹ = rand.(tuple(N...))  # isfwd = true (false) uses ∆l⁻¹ at dual (primal) locations
        isbloch = [true, false]
        e⁻ⁱᵏᴸ = rand(ComplexF64, 2)

        ∇̇ = create_divg(isfwd, ∆l⁻¹, isbloch, e⁻ⁱᵏᴸ, order_cmpfirst=false)

        # Test ∇̇.
        ∂x = (nw = 1; create_∂(nw, isfwd[nw], N, ∆l⁻¹[nw], isbloch[nw], e⁻ⁱᵏᴸ[nw]))
        ∂y = (nw = 2; create_∂(nw, isfwd[nw], N, ∆l⁻¹[nw], isbloch[nw], e⁻ⁱᵏᴸ[nw]))
        @test ∇̇ == [∂x ∂y]

        # Test Cartesian-component-first ordering.
        ∇̇_cmpfirst = create_divg(isfwd, ∆l⁻¹, isbloch, e⁻ⁱᵏᴸ, order_cmpfirst=true)
        @test ∇̇_cmpfirst == ∇̇[:,r]

        # Test apply_divg!.
        f = F[:]
        mul!(gvec, ∇̇, f)
        apply_divg!(g, F, isfwd, ∆l⁻¹, isbloch, e⁻ⁱᵏᴸ)  # no keyword argument order_cmpfirst in apply_XXX!()
        @test g[:] ≈ gvec

        mul!(gvec, ∇̇_cmpfirst, f[r])
        apply_divg!(g, F, isfwd, ∆l⁻¹, isbloch, e⁻ⁱᵏᴸ)  # no keyword argument order_cmpfirst in apply_XXX!()
        @test g[:] ≈ gvec
    end
end  # @testset "create_divg and apply_divg!"

@testset "divergence of curl" begin
    # Construct ∇̽and ∇̇ for a uniform grid and periodic boundaries.
    N = [3,4,5]
    M = prod(N)
    isfwd = [true, true, true]  # curl(U) and divg(V) are differentiated forward

    ∇̽ = create_curl(isfwd, N, order_cmpfirst=false)
    ∇̇ = create_divg(isfwd, N, order_cmpfirst=false)

    # Construct ∇̇ * ∇̽.
    A = ∇̇ * ∇̽

    # Test Divergence of curl.
    @test size(A) == (M,3M)
    @test all(A.==0)
end  # @testset "divergence of curl"

end  # @testset "divergence"
