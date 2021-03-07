@testset "util" begin

# @testset "dot" begin
#     t1 = (1,2,3)
#     t2 = (4,5,6)
#     @test t1⋅t2 == sum(t1 .* t2)
# end

# @testset "isapprox" begin
#     aa = [[1,2], [3,4]]
#     tt = ((1,2), (3,4))
#     at = [(1,2), (3,4)]
#     ta = ([1,2], [3,4])
#
#     @test aa ≈ aa
#     @test tt ≈ tt
#     @test_broken at ≈ at
#     @test ta ≈ ta
#     @test aa ≉ tt
#     @test tt ≉ at
#     @test at ≉ ta
#     @test_throws MethodError aa ≉ at
#     @test tt ≉ ta
#     @test aa ≉ ta
# end

@testset "t_ind" begin
    t3 = (1,2,3)
    t32 = ((1,2), (3,4), (5,6))
    t23 = ((1,2,3), (4,5,6))
    t2sa = (SVec(1,2,3), SVec(4,5,6))
    t2a = ([1,2,3], [4,5,6])
    t3a = ([1,2,3,4], [5,6,7,8], [9,10,11,12])

    # @test t_ind(t3, (2,3,1)) == (2,3,1)
    # @test t_ind(t32, 1, 2, 1) == (1,4,5)
    @test @inferred(t_ind(t23, PRIM, DUAL, PRIM)) == (1,5,3)
    @test @inferred(t_ind(t23, CartesianIndex(1,2,1))) == (1,5,3)
    @test @inferred(t_ind(t23, (PRIM,DUAL,PRIM))) == (1,5,3)
    @test @inferred(t_ind(t23, SVec(PRIM,DUAL,PRIM))) == (1,5,3)

    @test @inferred(t_ind(t2a, 2, 3)) == [2,6]
    @test @inferred(t_ind(t2sa, NEG, POS, NEG)) == [1,5,3]
    @test @inferred(t_ind(t2sa, (NEG,POS,NEG))) == [1,5,3]
    @test @inferred(t_ind(t2sa, SVec(NEG,POS,NEG))) == [1,5,3]

    @test @inferred(t_ind(t3a, 2, 3, 4)) == [2,7,12]
    @test @inferred(t_ind(t3a, CartesianIndex(2,3,4))) == [2,7,12]
    @test @inferred(t_ind(t3a, SVec(2,3,4))) == [2,7,12]
end

@testset "invert_∆l" begin
    N = (8,9,10)
    ∆lprim = rand.(N)
    ∆ldual = rand.(N)
    ∆l = (∆lprim, ∆ldual)

    ∆l⁻¹ = invert_∆l(∆l)
    @test ∆l⁻¹ == ((1 ./ ∆lprim[1], 1 ./ ∆lprim[2], 1 ./ ∆lprim[3]), (1 ./ ∆ldual[1], 1 ./ ∆ldual[2], 1 ./ ∆ldual[3]))

    ∆xprim = rand(10)
    ∆xdual = rand(10)
    @test invert_∆l((∆xprim, ∆xdual)) == (1 ./ ∆xprim, 1 ./ ∆xdual)
end

@testset "newtsol" begin
    f(x) = x^2-1
    f′(x) = 2x
    @test ((xsol,isconverged) = StaggeredGridCalculus.newtsol(2., f, f′); isconverged && xsol ≈ 1)
    @test ((xsol,isconverged) = StaggeredGridCalculus.newtsol(2., f); isconverged && xsol ≈ 1)
end

end  # @testset "base"
