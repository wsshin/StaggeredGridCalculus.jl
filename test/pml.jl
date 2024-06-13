@testset "pml" begin

@testset "PMLParam" begin
    pml = PMLParam()
    @test pml.m===4.0 && pml.R===exp(-16) && pml.κmax===1.0 && pml.amax===0.0 && pml.ma===4.0
end  # @testset "PMLParam"

@testset "1D" begin
    Npmln = 5
    Npmlp = 7
    Npml = ([Npmln], [Npmlp])
    N = 10
    M = N + Npmln + Npmlp
    ∆ldual = rand(M)
    L = sum(∆ldual)
    l₀ = L / 2
    lprim = cumsum([-l₀; ∆ldual])
    ldual = StaggeredGridCalculus.movingavg(lprim)

    lpml = ([lprim[1+Npmln]], [lprim[end-Npmlp]])
    Lpml = ([lpml[LowerBound()][1]-lprim[1]], [lprim[end]-lpml[UpperBound()][1]])

    bounds = ([lprim[1]], [lprim[end]])
    lprim = lprim[1:end-1]

    @test StaggeredGridCalculus.pml_loc(tuple(lprim), bounds, Npml) == (lpml, Lpml)

    ω = rand()
    pml = PMLParam()
    σmax = (-(pml.m+1) * log(pml.R) / 2 ./ Lpml[LowerBound()], -(pml.m+1) * log(pml.R) / 2 ./ Lpml[UpperBound()])  # see calc_stretch_factor
    f(l) = l≤lpml[LowerBound()][1] ? (1 + σmax[LowerBound()][1] * ((lpml[LowerBound()][1]-l)/Lpml[LowerBound()][1])^pml.m / (im*ω)) : (l≥lpml[UpperBound()][1] ? (1 + σmax[UpperBound()][1] * ((l-lpml[UpperBound()][1])/Lpml[UpperBound()][1])^pml.m / (im*ω)) : 1)
    sprim = f.(lprim)
    sdual = f.(ldual)

    l = (tuple(lprim), tuple(ldual))
    sfactor = StaggeredGridCalculus.create_sfactor(ω, l, lpml, Lpml)
    @test sfactor == (tuple(sprim), tuple(sdual))

    ∆lprim = diff(ldual)
    pushfirst!(∆lprim, ldual[1]+L-ldual[end])  # periodic boundary

    ∆l = (tuple(∆lprim), tuple(∆ldual))
    s∆lprim = sprim .* ∆lprim
    s∆ldual = sdual .* ∆ldual

    s∆l = create_stretched_∆l(ω, ∆l, l, bounds, Npml)
    @test s∆l == (tuple(s∆lprim), tuple(s∆ldual))

    isbloch = [true]
    g1 = Grid(tuple([lprim..., bounds[UpperBound()][1]]), isbloch)
    @test s∆l ≈ create_stretched_∆l(ω, g1, Npml)
end  # @testset "1D"

@testset "1D, Npml = 0" begin
    Npml = ([0],[0])
    lprim = [0.0, 1.0]  # with ghost point
    ldual = [0.5]  # without ghost point

    lpml = ([0.0], [1.0])
    Lpml = ([0.0], [0.0])

    bounds = ([lprim[1]], [lprim[end]])
    lprim = lprim[1:end-1]

    @test StaggeredGridCalculus.pml_loc(tuple(lprim), bounds, Npml) == (lpml, Lpml)

    ω = rand()
    l = (tuple(lprim), tuple(ldual))
    sfactor = StaggeredGridCalculus.create_sfactor(ω, l, lpml, Lpml)
    @test sfactor == (tuple([1.0]), tuple([1.0]))

    ∆lprim  = ∆ldual = [1.0]
    ∆l = (tuple(∆lprim), tuple(∆ldual))

    s∆l = create_stretched_∆l(ω, ∆l, l, bounds, Npml)
    @test s∆l == (tuple([1.0]), tuple([1.0]))

    isbloch = [true]
    g1 = Grid(tuple([0.0,1.0]), isbloch)
    @test s∆l ≈ create_stretched_∆l(ω, g1, Npml)
end  # @testset "1D, Npml = 0"

@testset "3D" begin
    Npml = ([10,5,1], [9,6,4])
    N = [10, 12, 1]
    M = sum(Npml) + N
    ∆ldual = ntuple(d->rand(M[d]), length(N))

    L = SVector(sum.(∆ldual))  # SFloat{3}
    l₀ = L ./ 2  # SFloat{3}
    lprim = map((v,s)->v.-s, map(x->[0; cumsum(x)], ∆ldual), (l₀...,))  # tuple of vectors
    ldual = StaggeredGridCalculus.movingavg.(lprim)

    lpml = (SVector{3}((w->lprim[w][1+Npml[LowerBound()][w]]).(1:3)), SVector{3}((w->lprim[w][end-Npml[UpperBound()][w]]).(1:3)))
    Lpml = (lpml[LowerBound()].-SVector{3}((w->lprim[w][1]).(1:3)), SVector{3}((w->lprim[w][end]).(1:3)).-lpml[UpperBound()])

    bounds = ([v[1] for v = lprim], [v[end] for v = lprim])
    lprim = map(v->v[1:end-1], lprim)

    @test StaggeredGridCalculus.pml_loc(lprim, bounds, Npml) == (lpml, Lpml)

    ω = rand()
    pml = PMLParam()
    σmax = (-(pml.m+1) * log(pml.R) / 2 ./ Lpml[LowerBound()], -(pml.m+1) * log(pml.R) / 2 ./ Lpml[UpperBound()])  # see calc_stretch_factor
    f(l, nw) = l≤lpml[LowerBound()][nw] ? (1 + σmax[LowerBound()][nw] * ((lpml[LowerBound()][nw]-l)/Lpml[LowerBound()][nw])^pml.m / (im*ω)) : (l≥lpml[UpperBound()][nw] ? (1 + σmax[UpperBound()][nw] * ((l-lpml[UpperBound()][nw])/Lpml[UpperBound()][nw])^pml.m / (im*ω)) : 1)
    sprim = (f.(lprim[1],1), f.(lprim[2],2), f.(lprim[3],3))
    sdual = (f.(ldual[1],1), f.(ldual[2],2), f.(ldual[3],3))

    l = (lprim, ldual)
    sfactor = StaggeredGridCalculus.create_sfactor(ω, l, lpml, Lpml)
    @test sfactor == (sprim, sdual)

    ∆lprim = diff.(ldual)
    for k = 1:3
        pushfirst!(∆lprim[k], ldual[k][1]+L[k]-ldual[k][end])  # periodic boundary
    end
    ∆l = (∆lprim, ∆ldual)
    s∆lprim = map((x,y)->x.*y, sfactor[PrimalGrid()], ∆l[PrimalGrid()])
    s∆ldual = map((x,y)->x.*y, sfactor[DualGrid()], ∆l[DualGrid()])

    s∆l = create_stretched_∆l(ω, ∆l, l, bounds, Npml)
    @test s∆l == (s∆lprim, s∆ldual)

    isbloch = [true,true,true]
    g3 = Grid(ntuple(k->[lprim[k]..., bounds[UpperBound()][k]], 3), isbloch)
    @test s∆l ≈ create_stretched_∆l(ω, g3, Npml)
end  # @testset "3D"

@testset "3D, Npml = 0" begin
    Npml = ([0,0,0], [0,0,0])
    lprim = ([0.0,1.0], [0.0,1.0], [0.0,1.0])  # with ghost points
    ldual = ([0.5], [0.5], [0.5])  # without ghost points

    lpml = ([0.0,0.0,0.0], [1.0,1.0,1.0])
    Lpml = ([0.0,0.0,0.0], [0.0,0.0,0.0])

    bounds = ([v[1] for v = lprim], [v[end] for v = lprim])
    lprim = map(v->v[1:end-1], lprim)

    @test StaggeredGridCalculus.pml_loc(lprim, bounds, Npml) == (lpml, Lpml)

    ω = rand()
    l = (lprim, ldual)
    sfactor = StaggeredGridCalculus.create_sfactor(ω, l, lpml, Lpml)
    @test sfactor == (([1.0],[1.0],[1.0]), ([1.0],[1.0],[1.0]))

    ∆lprim  = ∆ldual = ([1.0], [1.0], [1.0])
    ∆l = (∆lprim, ∆ldual)

    s∆l = create_stretched_∆l(ω, ∆l, l, bounds, Npml)
    @test s∆l == (([1.0],[1.0],[1.0]), ([1.0],[1.0],[1.0]))

    isbloch = [true,true,true]
    g3 = Grid(([0.0,1.0], [0.0,1.0], [0.0,1.0]), isbloch)
    @test s∆l ≈ create_stretched_∆l(ω, g3, Npml)
end  # @testset "3D, Npml = 0"

end  # @testset "pml"
