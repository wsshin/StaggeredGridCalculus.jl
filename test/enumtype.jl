@testset "enumtype" begin

@testset "instances" begin
    @test instances(Axis) == (X̂, Ŷ, Ẑ)
    @test instances(GridType) == (PRIM, DUAL)
    @test instances(Sign) == (NEG, POS)
end

@testset "numel" begin
    @test numel(Axis) == 3
    @test numel(GridType) == 2
    @test numel(Sign) == 2
end

@testset "integers" begin
    @test Int.(SVector(instances(Axis))) == nXYZ == [nX, nY, nZ]
    @test Int.(SVector(instances(GridType))) == nPD == [nPR, nDL]
    @test Int.(SVector(instances(Sign))) == nNP == [nN, nP]
end

@testset "next and alter" begin
    @test next3(X̂) == [Ŷ, Ẑ, X̂]
    @test next3(Ŷ) == [Ẑ, X̂, Ŷ]
    @test next3(Ẑ) == [X̂, Ŷ, Ẑ]

    @test next2(X̂) == [Ŷ, Ẑ]
    @test next2(Ŷ) == [Ẑ, X̂]
    @test next2(Ẑ) == [X̂, Ŷ]

    @test next1(X̂) == Ŷ
    @test next1(Ŷ) == Ẑ
    @test next1(Ẑ) == X̂

    @test prev3(X̂) == [Ẑ, Ŷ, X̂]
    @test prev3(Ŷ) == [X̂, Ẑ, Ŷ]
    @test prev3(Ẑ) == [Ŷ, X̂, Ẑ]

    @test prev2(X̂) == [Ẑ, Ŷ]
    @test prev2(Ŷ) == [X̂, Ẑ]
    @test prev2(Ẑ) == [Ŷ, X̂]

    @test prev1(X̂) == Ẑ
    @test prev1(Ŷ) == X̂
    @test prev1(Ẑ) == Ŷ

    @test alter(PRIM) == DUAL
    @test alter(DUAL) == PRIM

    @test alter(NEG) == POS
    @test alter(POS) == NEG
end

end  # @testset "enumtype"
