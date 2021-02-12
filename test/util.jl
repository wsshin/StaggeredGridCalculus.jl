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
