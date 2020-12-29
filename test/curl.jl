@testset "curl" begin

N = (3,4,5)
M = prod(N)
r = reshape(collect(1:3M), M, 3)'[:]  # index mapping from block matrix to narrowly banded matrix
Z = spzeros(M,M)

F = rand(Complex{Float64}, N..., 3)
G = similar(F)
g = zeros(Complex{Float64}, 3M)

@testset "create_curl and apply_curl!" begin
    for ci = CartesianIndices((false:true,false:true,false:true))
        # Construct Curl for a uniform grid and Bloch boundaries.
        isfwd = Vector{Bool}([ci.I...])
        Curl = create_curl(isfwd, [N...], order_cmpfirst=false)

        # Test the overall coefficients.
        @test size(Curl) == (3M,3M)
        @test all(any(Curl.≠0, dims=1))  # no zero columns
        @test all(any(Curl.≠0, dims=2))  # no zero rows
        @test all(sum(Curl, dims=2) .== 0)  # all row sums are zero, because Curl * ones(M) = 0

        ∂x = (nw = 1; create_∂(nw, isfwd[nw], [N...]))
        ∂y = (nw = 2; create_∂(nw, isfwd[nw], [N...]))
        ∂z = (nw = 3; create_∂(nw, isfwd[nw], [N...]))
        @test Curl == [Z -∂z ∂y;
                       ∂z Z -∂x;
                       -∂y ∂x Z]

        # Construct Curl for a nonuniform grid and general boundaries.
        ∆l⁻¹ = rand.(N)  # isfwd = true (false) uses ∆l⁻¹ at dual (primal) locations
        isbloch = [true, false, false]
        e⁻ⁱᵏᴸ = rand(ComplexF64, 3)

        Curl = create_curl(isfwd, [N...], ∆l⁻¹, isbloch, e⁻ⁱᵏᴸ, order_cmpfirst=false)

        # Test Curl.
        ∂x = (nw = 1; create_∂(nw, isfwd[nw], [N...], ∆l⁻¹[nw], isbloch[nw], e⁻ⁱᵏᴸ[nw]))
        ∂y = (nw = 2; create_∂(nw, isfwd[nw], [N...], ∆l⁻¹[nw], isbloch[nw], e⁻ⁱᵏᴸ[nw]))
        ∂z = (nw = 3; create_∂(nw, isfwd[nw], [N...], ∆l⁻¹[nw], isbloch[nw], e⁻ⁱᵏᴸ[nw]))
        @test Curl == [Z -∂z ∂y;
                     ∂z Z -∂x;
                     -∂y ∂x Z]

        # Test Cartesian-component-major ordering.
        Curl_cmpfirst = create_curl(isfwd, [N...], ∆l⁻¹, isbloch, e⁻ⁱᵏᴸ, order_cmpfirst=true)
        @test Curl_cmpfirst == Curl[r,r]

        # Test apply_curl!.
        f = F[:]
        mul!(g, Curl, f)
        apply_curl!(G, F, Val(:(=)), isfwd, ∆l⁻¹, isbloch, e⁻ⁱᵏᴸ)
        @test G[:] ≈ g

        # print("matrix: "); @btime mul!($g, $Curl, $f)
        # print("matrix-free: "); @btime apply_curl!($G, $F, Val(:(=)), $isfwd, $∆l⁻¹, $isbloch, $e⁻ⁱᵏᴸ)
    end
end  # @testset "create_curl and apply_curl!"

@testset "curl of curl" begin
    # Construct Cu and Cv for a uniform grid and Bloch boundaries.
    ∆ldual⁻¹ = ones.(N)  # inverse of ∆l evaluated at dual locations
    ∆lprim⁻¹ = ones.(N)  # inverse of ∆l evaluated at primal locations
    isbloch = [true, false, false]
    e⁻ⁱᵏᴸ = ones(3)

    Cu = create_curl([true,true,true], [N...], ∆ldual⁻¹, isbloch, e⁻ⁱᵏᴸ, order_cmpfirst=false)
    Cv = create_curl([false,false,false], [N...], ∆lprim⁻¹, isbloch, e⁻ⁱᵏᴸ, order_cmpfirst=false)

    # Test symmetry of each block.
    for i = nXYZ
        for j = next2(i)
            -Cv[(i-1)*M+1:i*M,(j-1)*M+1:j*M]' == Cu[(i-1)*M+1:i*M,(j-1)*M+1:j*M]
        end
    end

    # Construct Cv * Cu.
    isbloch = fill(true, 3)
    Cu = create_curl([true,true,true], [N...], ∆ldual⁻¹, isbloch, e⁻ⁱᵏᴸ, order_cmpfirst=false)
    Cv = create_curl([false,false,false], [N...], ∆lprim⁻¹, isbloch, e⁻ⁱᵏᴸ, order_cmpfirst=false)
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
    ∆ldual⁻¹ = ones.(N)
    ∆lprim⁻¹ = ones.(N)
    isbloch = [true, false, false]
    e⁻ⁱᵏᴸ = ones(3)

    isfwd = [true,false,true]
    Cu = create_curl(isfwd, [N...], ∆ldual⁻¹, isbloch, e⁻ⁱᵏᴸ, order_cmpfirst=false)
    Cv = create_curl(.!isfwd, [N...], ∆lprim⁻¹, isbloch, e⁻ⁱᵏᴸ, order_cmpfirst=false)

    # Test symmetry of each block.
    for i = nXYZ
        for j = next2(i)
            -Cv[(i-1)*M+1:i*M,(j-1)*M+1:j*M]' == Cu[(i-1)*M+1:i*M,(j-1)*M+1:j*M]
        end
    end

    # Construct Cv * Cu.
    isbloch = fill(true, 3)
    Cu = create_curl(isfwd, [N...], ∆ldual⁻¹, isbloch, e⁻ⁱᵏᴸ, order_cmpfirst=false)
    Cv = create_curl(.!isfwd, [N...], ∆lprim⁻¹, isbloch, e⁻ⁱᵏᴸ, order_cmpfirst=false)
    A = Cv * Cu

    # Test curl of curl.
    @test all(diag(A) .== 4)  # all diagonal entries are 4
    @test all(sum(A.≠0, dims=2) .== 13)  # 13 nonzero entries per row
    @test A == A'  # Hermitian

    B = A - 4I
    @test all(abs.(B[B.≠0]).==1)  # all nonzero off-diagonal entries are ±1
end  # @testset "curl of curl, mixed forward and backward"

end  # @testset "curl"
