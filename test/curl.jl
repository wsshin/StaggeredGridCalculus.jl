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
        # Construct ∇̽ for a uniform grid and Bloch boundaries.
        isfwd = Vector{Bool}([ci.I...])
        ∂x = (nw = 1; create_∂(nw, isfwd[nw], [N...]))
        ∂y = (nw = 2; create_∂(nw, isfwd[nw], [N...]))
        ∂z = (nw = 3; create_∂(nw, isfwd[nw], [N...]))
        ∇̽_blk = [Z -∂z ∂y;
                    ∂z Z -∂x;
                    -∂y ∂x Z]

        ∇̽ = create_curl(isfwd, [N...], cmpfirst=false)

        # Test the overall coefficients.
        @test size(∇̽) == (3M,3M)
        @test all(any(∇̽.≠0, dims=1))  # no zero columns
        @test all(any(∇̽.≠0, dims=2))  # no zero rows
        @test all(sum(∇̽, dims=2) .== 0)  # all row sums are zero, because ∇̽ * ones(M) = 0

        @test ∇̽ == ∇̽_blk

        # Construct ∇̽ for a nonuniform grid and general boundaries.
        ∆l⁻¹ = rand.(N)  # isfwd = true (false) uses ∆l⁻¹ at dual (primal) locations
        isbloch = [true, false, false]
        e⁻ⁱᵏᴸ = rand(ComplexF64, 3)
        ∂x = (nw = 1; create_∂(nw, isfwd[nw], [N...], ∆l⁻¹[nw], isbloch[nw], e⁻ⁱᵏᴸ[nw]))
        ∂y = (nw = 2; create_∂(nw, isfwd[nw], [N...], ∆l⁻¹[nw], isbloch[nw], e⁻ⁱᵏᴸ[nw]))
        ∂z = (nw = 3; create_∂(nw, isfwd[nw], [N...], ∆l⁻¹[nw], isbloch[nw], e⁻ⁱᵏᴸ[nw]))
        ∇̽_blk = [Z -∂z ∂y;
                    ∂z Z -∂x;
                    -∂y ∂x Z]

        ∇̽ = create_curl(isfwd, ∆l⁻¹, isbloch, e⁻ⁱᵏᴸ, cmpfirst=false)

        # Test ∇̽.
        @test ∇̽ == ∇̽_blk

        # Test Cartesian-component-major ordering.
        ∇̽_cmpfirst = create_curl(isfwd, ∆l⁻¹, isbloch, e⁻ⁱᵏᴸ, cmpfirst=true)
        @test ∇̽_cmpfirst == ∇̽[r,r]

        # Test apply_curl!.
        f = F[:]
        mul!(g, ∇̽, f)
        # apply_curl!(G, F, isfwd, ∆l⁻¹, isbloch, e⁻ⁱᵏᴸ)
        # @test G[:] ≈ g

        # print("matrix: "); @btime mul!($g, $∇̽, $f)
        # print("matrix-free: "); @btime apply_curl!($G, $F, $isfwd, $∆l⁻¹, $isbloch, $e⁻ⁱᵏᴸ)

        # Test 1×1 curls for 1D, and 1×2, 2×1 curls for 2D.
        nw_blk = [0 -3 2;
                  3 0 -1;
                  -2 1 0]

        dir_diffs = ([1], [1,2])
        for dir_diff = dir_diffs
            if length(dir_diff) == 1
                cmp_outs = setdiff(([1],[2],[3]), tuple(dir_diff))  # off-diagonal blocks
            else  # length(dir_diff) == 2
                cmp_outs = (dir_diff, setdiff([1,2,3], dir_diff))
            end

            for cmp_out = cmp_outs
                if length(dir_diff) == 1
                    cmp_in = [6 - only(dir_diff) - only(cmp_out)]
                else  # length(dir_diff) == 2
                    cmp_in = setdiff([1,2,3], cmp_out)
                end

                Ncmp = N[dir_diff]
                Mcmp = prod(Ncmp)
                Zcmp = spzeros(Mcmp,Mcmp)

                isfwdcmp = isfwd[dir_diff]
                ∆l⁻¹cmp = ∆l⁻¹[dir_diff]
                isblochcmp = isbloch[dir_diff]
                e⁻ⁱᵏᴸcmp = e⁻ⁱᵏᴸ[dir_diff]

                @show dir_diff, cmp_out, cmp_in

                ∇̽ = create_curl(isfwdcmp, ∆l⁻¹cmp, isblochcmp, e⁻ⁱᵏᴸcmp;
                                cmp_out, cmp_in, cmpfirst=false)

                ∂_blks = [[(nw = nw_blk[nv,nu];
                            if iszero(nw)
                                ∂_blk = Z
                            else
                                sn = sign(nw)
                                nw = abs(nw)
                                ind_nw = findfirst(dir_diff.==nw)
                                ∂_blk = sn * create_∂(ind_nw, isfwd[nw], [Ncmp...], ∆l⁻¹[nw], isbloch[nw], e⁻ⁱᵏᴸ[nw])
                            end;
                            ∂_blk) for nu = cmp_in] for nv = cmp_out]

                ∇̽_blk = cat(((row_blks->cat(row_blks..., dims=2)).(∂_blks))..., dims=1)

                @test ∇̽ == ∇̽_blk

                Fcmp = rand(ComplexF64, Ncmp..., length(cmp_in))
                Gcmp = similar(Fcmp, Ncmp..., length(cmp_out))
                gcmp = zeros(ComplexF64, length(Gcmp))

                fcmp = Fcmp[:]
                mul!(gcmp, ∇̽, fcmp)
                # apply_curl!(Gcmp, Fcmp, isfwdcmp, ∆l⁻¹cmp, isblochcmp, e⁻ⁱᵏᴸcmp;
                #             dir_diff, cmp_out, cmp_in)
                # @test Gcmp[:] ≈ gcmp
            end
        end
    end
end  # @testset "create_curl and apply_curl!"

@testset "curl of curl" begin
    # Construct ∇̽ᵤ and ∇̽ᵥ for a uniform grid and Bloch boundaries.
    ∆ldual⁻¹ = ones.(N)  # inverse of ∆l evaluated at dual locations
    ∆lprim⁻¹ = ones.(N)  # inverse of ∆l evaluated at primal locations
    isbloch = [true, false, false]
    e⁻ⁱᵏᴸ = ones(3)

    ∇̽ᵤ = create_curl([true,true,true], ∆ldual⁻¹, isbloch, e⁻ⁱᵏᴸ, cmpfirst=false)
    ∇̽ᵥ = create_curl([false,false,false], ∆lprim⁻¹, isbloch, e⁻ⁱᵏᴸ, cmpfirst=false)

    # Test symmetry of each block.
    for i = 1:3
        for j = mod1.(i .+ [1,2], 3)
            -∇̽ᵥ[(i-1)*M+1:i*M,(j-1)*M+1:j*M]' == ∇̽ᵤ[(i-1)*M+1:i*M,(j-1)*M+1:j*M]
        end
    end

    # Construct ∇̽ᵥ * ∇̽ᵤ.
    isbloch = fill(true, 3)
    ∇̽ᵤ = create_curl([true,true,true], ∆ldual⁻¹, isbloch, e⁻ⁱᵏᴸ, cmpfirst=false)
    ∇̽ᵥ = create_curl([false,false,false], ∆lprim⁻¹, isbloch, e⁻ⁱᵏᴸ, cmpfirst=false)
    A = ∇̽ᵥ * ∇̽ᵤ

    # Test curl of curl.
    @test all(diag(A) .== 4)  # all diagonal entries are 4
    @test all(sum(A.≠0, dims=2) .== 13)  # 13 nonzero entries per row
    @test A == A'  # Hermitian

    B = A - 4I
    @test all(abs.(B[B.≠0]).==1)  # all nonzero off-diagonal entries are ±1
end  # @testset "curl of curl"

@testset "curl of curl, mixed forward and backward" begin
    # Construct ∇̽ᵤ and ∇̽ᵥ for a uniform grid and Bloch boundaries.
    ∆ldual⁻¹ = ones.(N)
    ∆lprim⁻¹ = ones.(N)
    isbloch = [true, false, false]
    e⁻ⁱᵏᴸ = ones(3)

    isfwd = [true,false,true]
    ∇̽ᵤ = create_curl(isfwd, ∆ldual⁻¹, isbloch, e⁻ⁱᵏᴸ, cmpfirst=false)
    ∇̽ᵥ = create_curl(.!isfwd, ∆lprim⁻¹, isbloch, e⁻ⁱᵏᴸ, cmpfirst=false)

    # Test symmetry of each block.
    for i = 1:3
        for j = mod1.(i .+ [1,2], 3)
            -∇̽ᵥ[(i-1)*M+1:i*M,(j-1)*M+1:j*M]' == ∇̽ᵤ[(i-1)*M+1:i*M,(j-1)*M+1:j*M]
        end
    end

    # Construct ∇̽ᵥ * ∇̽ᵤ.
    isbloch = fill(true, 3)
    ∇̽ᵤ = create_curl(isfwd, ∆ldual⁻¹, isbloch, e⁻ⁱᵏᴸ, cmpfirst=false)
    ∇̽ᵥ = create_curl(.!isfwd, ∆lprim⁻¹, isbloch, e⁻ⁱᵏᴸ, cmpfirst=false)
    A = ∇̽ᵥ * ∇̽ᵤ

    # Test curl of curl.
    @test all(diag(A) .== 4)  # all diagonal entries are 4
    @test all(sum(A.≠0, dims=2) .== 13)  # 13 nonzero entries per row
    @test A == A'  # Hermitian

    B = A - 4I
    @test all(abs.(B[B.≠0]).==1)  # all nonzero off-diagonal entries are ±1
end  # @testset "curl of curl, mixed forward and backward"

end  # @testset "curl"
