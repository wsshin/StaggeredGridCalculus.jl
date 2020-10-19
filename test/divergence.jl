@testset "divergence" begin

N = (3,4,5)
M = prod(N)
r = reshape(collect(1:3M), M, 3)'[:]  # index mapping from block matrix to narrowly banded matrix
Z = spzeros(M,M)

F = rand(Complex{Float64}, N..., 3)
G = similar(F)
g = zeros(Complex{Float64}, 3M)

@testset "create_divg and apply_divg! for primal field U" begin
    # Construct Du for a uniform grid and BLOCH boundaries.
    isfwd = [false, false, false]  # U is differentiated backward
    Du = create_divg(isfwd, [N...], reorder=false)

    # Test the overall coefficients.
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
    parity = [1, -1, -1]

    Du = create_divg(isfwd, [N...], ∆lprim, isbloch, e⁻ⁱᵏᴸ, parity=parity, reorder=false)

    # Test Cu.
    ∂x = (nw = 1; create_∂(nw, isfwd[nw], [N...], ∆lprim[nw], isbloch[nw], e⁻ⁱᵏᴸ[nw]))
    ∂y = (nw = 2; create_∂(nw, isfwd[nw], [N...], ∆lprim[nw], isbloch[nw], e⁻ⁱᵏᴸ[nw]))
    ∂z = (nw = 3; create_∂(nw, isfwd[nw], [N...], ∆lprim[nw], isbloch[nw], e⁻ⁱᵏᴸ[nw]))
    @test Du == [parity[1].*∂x parity[2].*∂y parity[3].*∂z]

    # Test reordering.
    Du_reorder = create_divg(isfwd, [N...], ∆lprim, isbloch, e⁻ⁱᵏᴸ, parity=parity, reorder=true)
    @test Du_reorder == Du[:,r]

    # # Test apply_curl!.
    # f = F[:]
    # mul!(g, Cu, f)
    # G .= 0
    # apply_curl!(G, F, isfwd, ∆lprim, isbloch, e⁻ⁱᵏᴸ)
    # @test G[:] ≈ g
    #
    # # print("matrix: "); @btime mul!($g, $Cu, $f)
    # # print("matrix-free: "); @btime apply_curl!($G, $F, $isfwd, $∆lprim, $isbloch, $e⁻ⁱᵏᴸ)
end  # @testset "create_divg and apply_divg! for primal field U"

# @testset "create_divg and apply_divg! for dual field V" begin
#     # To be filled
# end  # @testset "create_divergence and apply_divergence! for dual field V"

@testset "divergence of curl" begin
    # Construct Cu and Dv for a uniform grid and periodic boundaries.
    isfwd = [true, true, true]  # curl(U) and divg(V) are differentiated forward
    isbloch = [true, false, false]

    Cu = create_curl(isfwd, [N...], reorder=false)
    Dv = create_divg(isfwd, [N...], reorder=false)

    # Construct Dv * Cu.
    A = Dv * Cu

    # Test Divergence of curl.
    @test size(A) == (M,3M)
    @test all(A.==0)
end  # @testset "curl of curl"

end  # @testset "divergence"
