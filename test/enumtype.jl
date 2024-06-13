@testset "enumtype" begin

@testset "instances" begin
    @test instances(AbstractGridType) == (PrimalGrid(), DualGrid())
    @test instances(AbstractBoundType) == (LowerBound(), UpperBound())
end

@testset "conj" begin
    @test conj(PrimalGrid()) == DualGrid()
    @test conj(DualGrid()) == PrimalGrid()

    @test conj(LowerBound()) == UpperBound()
    @test conj(UpperBound()) == LowerBound()
end

end  # @testset "enumtype"
