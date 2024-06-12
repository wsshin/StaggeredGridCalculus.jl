@testset "enumtype" begin

@testset "instances" begin
    @test instances(GridType) == (PRIM, DUAL)
    @test instances(AbstractBoundType) == (LowerBound(), UpperBound())
end

@testset "numel" begin
    @test numel(GridType) == 2
    @test numel(AbstractBoundType) == 2
end

@testset "integers" begin
    @test Int.(SVector(instances(GridType))) == nPD == [nPR, nDL]
end

@testset "next and alter" begin
    @test alter(PRIM) == DUAL
    @test alter(DUAL) == PRIM

    @test conj(LowerBound()) == UpperBound()
    @test conj(UpperBound()) == LowerBound()
end

end  # @testset "enumtype"
