@testset "divergence" begin

N = (3,4,5)
M = prod(N)
r = reshape(collect(1:3M), M, 3)'[:]  # index mapping from block matrix to narrowly banded matrix
Z = spzeros(M,M)

F = rand(Complex{Float64}, N..., 3)
G = similar(F)
g = zeros(Complex{Float64}, 3M)

@testset "create_divg and apply_divg! for primal field U" begin
    # Construct Du for a uniform grid and Bloch boundaries.
    isfwd = [false, false, false]  # U is differentiated backward
    Du = create_divg(isfwd, [N...], order_compfirst=false)

    # Test the overall coefficients.
    @test size(Du) == (M,3M)
    @test all(any(Du.≠0, dims=1))  # no zero columns
    @test all(any(Du.≠0, dims=2))  # no zero rows
    @test all(sum(Du, dims=1) .== 0)  # all column sums are zero, because each input field to Du is used twice, once multiplied with +1 and once with -1
    @test all(sum(Du, dims=2) .== 0)  # all row sums are zero, because Du * ones(M) = 0
    @test all(sum(abs.(Du), dims=1) .== 2)  # each column of Du has two nonzero entries, which are ±1's
    @test all(sum(abs.(Du), dims=2) .== 6)  # each row of Du has six nonzero entries, which are ±1's

    ∂x = (nw = 1; create_∂(nw, isfwd[nw], [N...]))
    ∂y = (nw = 2; create_∂(nw, isfwd[nw], [N...]))
    ∂z = (nw = 3; create_∂(nw, isfwd[nw], [N...]))
    @test Du == [∂x ∂y ∂z]

    # Construct Du for a nonuniform grid and general boundaries.
    ∆lprim = rand.(N)
    isbloch = [true, false, false]
    e⁻ⁱᵏᴸ = rand(ComplexF64, 3)
    scale∂ = [1, -1, -1]

    Du = create_divg(isfwd, [N...], ∆lprim, isbloch, e⁻ⁱᵏᴸ, scale∂=scale∂, order_compfirst=false)

    # Test Cu.
    ∂x = (nw = 1; create_∂(nw, isfwd[nw], [N...], ∆lprim[nw], isbloch[nw], e⁻ⁱᵏᴸ[nw]))
    ∂y = (nw = 2; create_∂(nw, isfwd[nw], [N...], ∆lprim[nw], isbloch[nw], e⁻ⁱᵏᴸ[nw]))
    ∂z = (nw = 3; create_∂(nw, isfwd[nw], [N...], ∆lprim[nw], isbloch[nw], e⁻ⁱᵏᴸ[nw]))
    @test Du == [scale∂[1].*∂x scale∂[2].*∂y scale∂[3].*∂z]

    # Test Cartesian-component-first ordering.
    Du_compfirst = create_divg(isfwd, [N...], ∆lprim, isbloch, e⁻ⁱᵏᴸ, scale∂=scale∂, order_compfirst=true)
    @test Du_compfirst == Du[:,r]

    # Test permutation.
    Du_permute = create_divg(isfwd, [N...], ∆lprim, isbloch, e⁻ⁱᵏᴸ, permute∂=[2,1,3], scale∂=scale∂, order_compfirst=false)
    @test Du_permute == [scale∂[1].*∂y scale∂[2].*∂x scale∂[3].*∂z]

    Du_permute_compfirst = create_divg(isfwd, [N...], ∆lprim, isbloch, e⁻ⁱᵏᴸ, permute∂=[2,1,3], scale∂=scale∂, order_compfirst=true)
    @test Du_permute_compfirst == Du_permute[:,r]

    # Test apply_divg!.
    # # to be filled
end  # @testset "create_divg and apply_divg! for primal field U"

# @testset "create_divg and apply_divg! for dual field V" begin
#     # To be filled
# end  # @testset "create_divg and apply_divg! for dual field V"

@testset "divergence of curl" begin
    # Construct Cu and Dv for a uniform grid and periodic boundaries.
    isfwd = [true, true, true]  # curl(U) and divg(V) are differentiated forward
    isbloch = [true, false, false]

    Cu = create_curl(isfwd, [N...], order_compfirst=false)
    Dv = create_divg(isfwd, [N...], order_compfirst=false)

    # Construct Dv * Cu.
    A = Dv * Cu

    # Test Divergence of curl.
    @test size(A) == (M,3M)
    @test all(A.==0)
end  # @testset "divergence of curl"

end  # @testset "divergence"
