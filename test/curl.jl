@testset "curl" begin

N = (3,4,5)
M = prod(N)
r = reshape(collect(1:3M), M, 3)'[:]  # index mapping from block matrix to narrowly banded matrix
Z = spzeros(M,M)

F = rand(ComplexF64, N..., 3)
G = similar(F)
g = zeros(ComplexF64, 3M)

@testset "create_curl and apply_curl!" begin
    for ci = CartesianIndices((false:true,false:true,false:true))
        # Construct Curl for a uniform grid and Bloch boundaries.
        isfwd = Vector{Bool}([ci.I...])
        ∂x = (nw = 1; create_∂(nw, isfwd[nw], [N...]))
        ∂y = (nw = 2; create_∂(nw, isfwd[nw], [N...]))
        ∂z = (nw = 3; create_∂(nw, isfwd[nw], [N...]))
        Curl_blk = [Z -∂z ∂y;
                    ∂z Z -∂x;
                    -∂y ∂x Z]

        Curl = create_curl(isfwd, [N...], order_cmpfirst=false)

        # Test the overall coefficients.
        @test size(Curl) == (3M,3M)
        @test all(any(Curl.≠0, dims=1))  # no zero columns
        @test all(any(Curl.≠0, dims=2))  # no zero rows
        @test all(sum(Curl, dims=2) .== 0)  # all row sums are zero, because Curl * ones(M) = 0

        @test Curl == Curl_blk

        # Construct Curl for a nonuniform grid and general boundaries.
        ∆l⁻¹ = rand.(N)  # isfwd = true (false) uses ∆l⁻¹ at dual (primal) locations
        isbloch = [true, false, false]
        e⁻ⁱᵏᴸ = rand(ComplexF64, 3)
        ∂x = (nw = 1; create_∂(nw, isfwd[nw], [N...], ∆l⁻¹[nw], isbloch[nw], e⁻ⁱᵏᴸ[nw]))
        ∂y = (nw = 2; create_∂(nw, isfwd[nw], [N...], ∆l⁻¹[nw], isbloch[nw], e⁻ⁱᵏᴸ[nw]))
        ∂z = (nw = 3; create_∂(nw, isfwd[nw], [N...], ∆l⁻¹[nw], isbloch[nw], e⁻ⁱᵏᴸ[nw]))
        Curl_blk = [Z -∂z ∂y;
                    ∂z Z -∂x;
                    -∂y ∂x Z]

        Curl = create_curl(isfwd, ∆l⁻¹, isbloch, e⁻ⁱᵏᴸ, order_cmpfirst=false)

        # Test Curl.
        @test Curl == Curl_blk

        # Test Cartesian-component-major ordering.
        Curl_cmpfirst = create_curl(isfwd, ∆l⁻¹, isbloch, e⁻ⁱᵏᴸ, order_cmpfirst=true)
        @test Curl_cmpfirst == Curl[r,r]

        # Test apply_curl!.
        f = F[:]
        mul!(g, Curl, f)
        apply_curl!(G, F, Val(:(=)), isfwd, ∆l⁻¹, isbloch, e⁻ⁱᵏᴸ)
        @test G[:] ≈ g

        # print("matrix: "); @btime mul!($g, $Curl, $f)
        # print("matrix-free: "); @btime apply_curl!($G, $F, Val(:(=)), $isfwd, $∆l⁻¹, $isbloch, $e⁻ⁱᵏᴸ)

        # Test 1×1 curls for 1D, and 1×2, 2×1 curls for 2D.
        nw_blk = [0 -3 2;
                  3 0 -1;
                  -2 1 0]

        cmp_shps = ([1], [2], [3], [1,2], [2,3], [3,1])
        for cmp_shp = cmp_shps
            if length(cmp_shp) == 1
                cmp_outs = setdiff(([1],[2],[3]), tuple(cmp_shp))
            else  # length(cmp_shp) == 2
                cmp_outs = (cmp_shp, setdiff([1,2,3], cmp_shp))
            end

            for cmp_out = cmp_outs
                if length(cmp_shp) == 1
                    cmp_in = [6 - only(cmp_shp) - only(cmp_out)]
                else  # length(cmp_shp) == 2
                    cmp_in = setdiff([1,2,3], cmp_out)
                end

                Ncmp = N[cmp_shp]
                Mcmp = prod(Ncmp)
                Zcmp = spzeros(Mcmp,Mcmp)

                isfwdcmp = isfwd[cmp_shp]
                ∆l⁻¹cmp = ∆l⁻¹[cmp_shp]
                isblochcmp = isbloch[cmp_shp]
                e⁻ⁱᵏᴸcmp = e⁻ⁱᵏᴸ[cmp_shp]

                Curl = create_curl(isfwdcmp, ∆l⁻¹cmp, isblochcmp, e⁻ⁱᵏᴸcmp;
                                   cmp_shp, cmp_out, cmp_in, order_cmpfirst=false)

                ∂_blks = [[(nw = nw_blk[nv,nu];
                            if iszero(nw)
                                ∂_blk = Z
                            else
                                sn = sign(nw)
                                nw = abs(nw)
                                ind_nw = findfirst(cmp_shp.==nw)
                                ∂_blk = sn * create_∂(ind_nw, isfwd[nw], [Ncmp...], ∆l⁻¹[nw], isbloch[nw], e⁻ⁱᵏᴸ[nw])
                            end;
                            ∂_blk) for nu = cmp_in] for nv = cmp_out]

                Curl_blk = cat(((row_blks->cat(row_blks..., dims=2)).(∂_blks))..., dims=1)

                @test Curl == Curl_blk

                Fcmp = rand(ComplexF64, Ncmp..., length(cmp_in))
                Gcmp = similar(Fcmp, Ncmp..., length(cmp_out))
                gcmp = zeros(ComplexF64, length(Gcmp))

                fcmp = Fcmp[:]
                mul!(gcmp, Curl, fcmp)
                apply_curl!(Gcmp, Fcmp, Val(:(=)), isfwdcmp, ∆l⁻¹cmp, isblochcmp, e⁻ⁱᵏᴸcmp;
                            cmp_shp, cmp_out, cmp_in)
                @test Gcmp[:] ≈ gcmp
            end
        end
    end
end  # @testset "create_curl and apply_curl!"

@testset "curl of curl" begin
    # Construct Cu and Cv for a uniform grid and Bloch boundaries.
    ∆ldual⁻¹ = ones.(N)  # inverse of ∆l evaluated at dual locations
    ∆lprim⁻¹ = ones.(N)  # inverse of ∆l evaluated at primal locations
    isbloch = [true, false, false]
    e⁻ⁱᵏᴸ = ones(3)

    Cu = create_curl([true,true,true], ∆ldual⁻¹, isbloch, e⁻ⁱᵏᴸ, order_cmpfirst=false)
    Cv = create_curl([false,false,false], ∆lprim⁻¹, isbloch, e⁻ⁱᵏᴸ, order_cmpfirst=false)

    # Test symmetry of each block.
    for i = 1:3
        for j = mod1.(i .+ [1,2], 3)
            -Cv[(i-1)*M+1:i*M,(j-1)*M+1:j*M]' == Cu[(i-1)*M+1:i*M,(j-1)*M+1:j*M]
        end
    end

    # Construct Cv * Cu.
    isbloch = fill(true, 3)
    Cu = create_curl([true,true,true], ∆ldual⁻¹, isbloch, e⁻ⁱᵏᴸ, order_cmpfirst=false)
    Cv = create_curl([false,false,false], ∆lprim⁻¹, isbloch, e⁻ⁱᵏᴸ, order_cmpfirst=false)
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
    Cu = create_curl(isfwd, ∆ldual⁻¹, isbloch, e⁻ⁱᵏᴸ, order_cmpfirst=false)
    Cv = create_curl(.!isfwd, ∆lprim⁻¹, isbloch, e⁻ⁱᵏᴸ, order_cmpfirst=false)

    # Test symmetry of each block.
    for i = 1:3
        for j = mod1.(i .+ [1,2], 3)
            -Cv[(i-1)*M+1:i*M,(j-1)*M+1:j*M]' == Cu[(i-1)*M+1:i*M,(j-1)*M+1:j*M]
        end
    end

    # Construct Cv * Cu.
    isbloch = fill(true, 3)
    Cu = create_curl(isfwd, ∆ldual⁻¹, isbloch, e⁻ⁱᵏᴸ, order_cmpfirst=false)
    Cv = create_curl(.!isfwd, ∆lprim⁻¹, isbloch, e⁻ⁱᵏᴸ, order_cmpfirst=false)
    A = Cv * Cu

    # Test curl of curl.
    @test all(diag(A) .== 4)  # all diagonal entries are 4
    @test all(sum(A.≠0, dims=2) .== 13)  # 13 nonzero entries per row
    @test A == A'  # Hermitian

    B = A - 4I
    @test all(abs.(B[B.≠0]).==1)  # all nonzero off-diagonal entries are ±1
end  # @testset "curl of curl, mixed forward and backward"

end  # @testset "curl"
