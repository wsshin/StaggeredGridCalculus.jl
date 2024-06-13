@testset "enumtype" begin

@testset "instances" begin
    @test instances(AbstractGridType) == (PrimalGrid(), DualGrid())
    @test instances(AbstractBoundType) == (LowerBound(), UpperBound())
end

@testset "numel" begin
    @test numel(AbstractGridType) == 2
    @test numel(AbstractBoundType) == 2
end

@testset "next and alter" begin
    @test conj(PrimalGrid()) == DualGrid()
    @test conj(DualGrid()) == PrimalGrid()

    @test conj(LowerBound()) == UpperBound()
    @test conj(UpperBound()) == LowerBound()
end

end  # @testset "enumtype"
