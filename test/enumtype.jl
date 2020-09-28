@testset "enumtype" begin

@testset "instances" begin
    @test instances(GridType) == (PRIM, DUAL)
    @test instances(Sign) == (NEG, POS)
end

@testset "numel" begin
    @test numel(GridType) == 2
    @test numel(Sign) == 2
end

@testset "integers" begin
    @test Int.(SVector(instances(GridType))) == nPD == [nPR, nDL]
    @test Int.(SVector(instances(Sign))) == nNP == [nN, nP]
end

@testset "next and alter" begin
    @test alter(PRIM) == DUAL
    @test alter(DUAL) == PRIM

    @test alter(NEG) == POS
    @test alter(POS) == NEG
end

end  # @testset "enumtype"
