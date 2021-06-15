@testset "gradient" begin

N = [4,5]
M = prod(N)
r = reshape(collect(1:2M), M, 2)'[:]  # index mapping from block matrix to narrowly banded matrix
Z = spzeros(M,M)

G = rand(ComplexF64, N..., 2)  # output array of vector field; make sure not to be confused with gradient operator
f = zeros(ComplexF64, N...)  # input array of scalar
g = zeros(ComplexF64, 2M)  # column vector representation of G

@testset "create_grad and apply_grad!" begin
    for ci = CartesianIndices((false:true,false:true))
        # Construct Grad for a uniform grid and Bloch boundaries.
        isfwd = Vector{Bool}([ci.I...])
        Grad = create_grad(isfwd, N, order_cmpfirst=false)

        # Test the overall coefficients.
        @test size(Grad) == (2M,M)
        @test all(any(Grad.≠0, dims=1))  # no zero columns
        @test all(any(Grad.≠0, dims=2))  # no zero rows
        @test all(sum(Grad, dims=1) .== 0)  # all column sums are zero, because each input field to Grad is used twice in each Cartesian direction, once multiplied with +1 and once with -1
        @test all(sum(Grad, dims=2) .== 0)  # all row sums are zero, because Grad * ones(M) = 0
        @test all(sum(abs.(Grad), dims=1) .== 4)  # each column of Grad has six nonzero entries, which are ±1's
        @test all(sum(abs.(Grad), dims=2) .== 2)  # each row of Grad has two nonzero entries, which are ±1's

        ∂x = (nw = 1; create_∂(nw, isfwd[nw], N))
        ∂y = (nw = 2; create_∂(nw, isfwd[nw], N))
        @test Grad == [∂x; ∂y]

        # Construct Grad for a nonuniform grid and general boundaries.
        ∆l⁻¹ = rand.(tuple(N...))  # isfwd = true (false) uses ∆l⁻¹ at dual (primal) locations
        isbloch = [true, false]
        e⁻ⁱᵏᴸ = rand(ComplexF64, 2)

        Grad = create_grad(isfwd, ∆l⁻¹, isbloch, e⁻ⁱᵏᴸ, order_cmpfirst=false)

        # Test Grad.
        ∂x = (nw = 1; create_∂(nw, isfwd[nw], N, ∆l⁻¹[nw], isbloch[nw], e⁻ⁱᵏᴸ[nw]))
        ∂y = (nw = 2; create_∂(nw, isfwd[nw], N, ∆l⁻¹[nw], isbloch[nw], e⁻ⁱᵏᴸ[nw]))
        @test Grad == [∂x; ∂y]

        # Test Cartesian-component-first ordering.
        Grad_cmpfirst = create_grad(isfwd, ∆l⁻¹, isbloch, e⁻ⁱᵏᴸ, order_cmpfirst=true)
        @test Grad_cmpfirst == Grad[r,:]

        # Test apply_grad!.
        fvec = f[:]
        mul!(g, Grad, fvec)
        apply_grad!(G, f, Val(:(=)), isfwd, ∆l⁻¹, isbloch, e⁻ⁱᵏᴸ)  # no keyword argument order_cmpfirst in apply_XXX!()
        @test G[:] ≈ g

        mul!(g, Grad_cmpfirst, fvec)
        apply_grad!(G, f, Val(:(=)), isfwd, ∆l⁻¹, isbloch, e⁻ⁱᵏᴸ)  # no keyword argument order_cmpfirst in apply_XXX!()
        @test G[:][r] ≈ g
    end
end  # @testset "create_grad and apply_grad!"

@testset "curl of gradient" begin
    # Construct Curl and Grad for a uniform grid and periodic boundaries.
    N = [3,4,5]
    M = prod(N)
    isfwd = [true, true, true]  # curl(U) and gradient to generate U are differentiated forward

    Curl = create_curl(isfwd, N, order_cmpfirst=false)
    Grad = create_grad(isfwd, N, order_cmpfirst=false)

    # Construct Curl * Grad
    A = Curl * Grad

    # Test Divergence of curl.
    @test size(A) == (3M,M)
    @test all(A.==0)
end  # @testset "curl of gradent"

end  # @testset "gradient"
