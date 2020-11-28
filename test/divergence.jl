@testset "divergence" begin

N = [4,5]
M = prod(N)
r = reshape(collect(1:2M), M, 2)'[:]  # index mapping from block matrix to narrowly banded matrix
Z = spzeros(M,M)

F = rand(Complex{Float64}, N..., 2)
g = zeros(Complex{Float64}, N...)
gvec = zeros(Complex{Float64}, M)

@testset "create_divg and apply_divg! for primal field U" begin
    # Construct Du for a uniform grid and Bloch boundaries.
    isfwd = [false, false]  # U is differentiated backward
    Du = create_divg(isfwd, N, order_cmpfirst=false)

    # Test the overall coefficients.
    @test size(Du) == (M,2M)
    @test all(any(Du.≠0, dims=1))  # no zero columns
    @test all(any(Du.≠0, dims=2))  # no zero rows
    @test all(sum(Du, dims=1) .== 0)  # all column sums are zero, because each input field to Du is used twice, once multiplied with +1 and once with -1
    @test all(sum(Du, dims=2) .== 0)  # all row sums are zero, because Du * ones(M) = 0
    @test all(sum(abs.(Du), dims=1) .== 2)  # each column of Du has two nonzero entries, which are ±1's
    @test all(sum(abs.(Du), dims=2) .== 4)  # each row of Du has six nonzero entries, which are ±1's

    ∂x = (nw = 1; create_∂(nw, isfwd[nw], N))
    ∂y = (nw = 2; create_∂(nw, isfwd[nw], N))
    @test Du == [∂x ∂y]

    # Construct Du for a nonuniform grid and general boundaries.
    ∆lprim⁻¹ = rand.(tuple(N...))
    isbloch = [true, false]
    e⁻ⁱᵏᴸ = rand(ComplexF64, 2)
    scale∂ = [1, -1]  # +∂x, -∂y

    Du = create_divg(isfwd, N, ∆lprim⁻¹, isbloch, e⁻ⁱᵏᴸ, scale∂=scale∂, order_cmpfirst=false)

    # Test Cu.
    ∂x = (nw = 1; create_∂(nw, isfwd[nw], N, ∆lprim⁻¹[nw], isbloch[nw], e⁻ⁱᵏᴸ[nw]))
    ∂y = (nw = 2; create_∂(nw, isfwd[nw], N, ∆lprim⁻¹[nw], isbloch[nw], e⁻ⁱᵏᴸ[nw]))
    @test Du == [scale∂[1].*∂x scale∂[2].*∂y]

    # Test Cartesian-component-first ordering.
    Du_cmpfirst = create_divg(isfwd, N, ∆lprim⁻¹, isbloch, e⁻ⁱᵏᴸ, scale∂=scale∂, order_cmpfirst=true)
    @test Du_cmpfirst == Du[:,r]

    # Test permutation.
    permute∂ = [2, 1]  # with scale∂ = [1,-1], create divergence operator [-∂y, ∂x]
    Du_permute = create_divg(isfwd, N, ∆lprim⁻¹, isbloch, e⁻ⁱᵏᴸ, permute∂=permute∂, scale∂=scale∂, order_cmpfirst=false)
    @test Du_permute == [scale∂[2].*∂y scale∂[1].*∂x]

    Du_permute_cmpfirst = create_divg(isfwd, N, ∆lprim⁻¹, isbloch, e⁻ⁱᵏᴸ, permute∂=permute∂, scale∂=scale∂, order_cmpfirst=true)
    @test Du_permute_cmpfirst == Du_permute[:,r]

    # Test apply_divg!.
    f = F[:]
    mul!(gvec, Du, f)
    g .= 0
    apply_divg!(g, F, isfwd, ∆lprim⁻¹, isbloch, e⁻ⁱᵏᴸ, scale∂=scale∂)
    @test g[:] ≈ gvec

    mul!(gvec, Du_cmpfirst, f[r])
    g .= 0
    apply_divg!(g, F, isfwd, ∆lprim⁻¹, isbloch, e⁻ⁱᵏᴸ, scale∂=scale∂)
    @test g[:] ≈ gvec

    mul!(gvec, Du_permute_cmpfirst, f[r])
    g .= 0
    apply_divg!(g, F, isfwd, ∆lprim⁻¹, isbloch, e⁻ⁱᵏᴸ, permute∂=permute∂, scale∂=scale∂)
    @test g[:] ≈ gvec
end  # @testset "create_divg and apply_divg! for primal field U"

# @testset "create_divg and apply_divg! for dual field V" begin
#     # To be filled
# end  # @testset "create_divg and apply_divg! for dual field V"

@testset "divergence of curl" begin
    # Construct Cu and Dv for a uniform grid and periodic boundaries.
    N = [3,4,5]
    M = prod(N)
    isfwd = [true, true, true]  # curl(U) and divg(V) are differentiated forward

    Cu = create_curl(isfwd, N, order_cmpfirst=false)
    Dv = create_divg(isfwd, N, order_cmpfirst=false)

    # Construct Dv * Cu.
    A = Dv * Cu

    # Test Divergence of curl.
    @test size(A) == (M,3M)
    @test all(A.==0)
end  # @testset "divergence of curl"

end  # @testset "divergence"
