@testset "pml" begin

@testset "PMLParam" begin
    pml = PMLParam()
    @test pml.m===4.0 && pml.R===exp(-16) && pml.κmax===1.0 && pml.amax===0.0 && pml.ma===4.0
end  # @testset "PMLParam"

@testset "1D" begin
    Npmln = 5
    Npmlp = 7
    Npml = (Npmln, Npmlp)
    N = 10
    M = N + Npmln + Npmlp
    ∆ldual = rand(M)
    L = sum(∆ldual)
    l₀ = L / 2
    lprim = cumsum([-l₀; ∆ldual])
    ldual = StaggeredGridCalculus.movingavg(lprim)

    lpml = (lprim[1+Npmln], lprim[end-Npmlp])
    Lpml = (lpml[nN]-lprim[1], lprim[end]-lpml[nP])

    @test get_pml_loc(lprim, Npml) == (lpml, Lpml)

    ω = rand()
    pml = PMLParam()
    σmax = -(pml.m+1) * log(pml.R) / 2 ./ Lpml  # see calc_stretch_factor
    f(l) = l≤lpml[nN] ? (1 + σmax[nN] * ((lpml[nN]-l)/Lpml[nN])^pml.m / (im*ω)) : (l≥lpml[nP] ? (1 + σmax[nP] * ((l-lpml[nP])/Lpml[nP])^pml.m / (im*ω)) : 1)
    sprim = f.(lprim[1:end-1])
    sdual = f.(ldual)

    @test gen_stretch_factor(ω, (lprim[1:end-1], ldual), lpml, Lpml) == (sprim, sdual)
end  # @testset "get_pml_loc"

@testset "1D, Npml = 0" begin
    Npml = (0,0)
    lprim = [0.0, 1.0]  # with ghost point
    ldual = [0.5]  # without ghost point

    lpml = (0.0, 1.0)
    Lpml = (0.0, 0.0)

    @test get_pml_loc(lprim, Npml) == (lpml, Lpml)

    ω = rand()
    @test gen_stretch_factor(ω, (lprim[1:end-1], ldual), lpml, Lpml) == ([1.0], [1.0])
end  # @testset "get_pml_loc"

@testset "3D" begin
    Npml = ([10,5,1], [9,6,4])
    N = [10, 12, 1]
    M = sum(Npml) + N
    ∆ldual = ntuple(d->rand(M[d]), length(N))

    L = SVec(sum.(∆ldual))  # SFloat{3}
    l₀ = L ./ 2  # SFloat{3}
    lprim = map((v,s)->v.-s, map(x->[0; cumsum(x)], ∆ldual), (l₀...,))  # tuple of vectors
    ldual = StaggeredGridCalculus.movingavg.(lprim)

    lpml = (SVec{3}((w->lprim[w][1+Npml[nN][w]]).(1:3)), SVec{3}((w->lprim[w][end-Npml[nP][w]]).(1:3)))
    Lpml = (lpml[nN].-SVec{3}((w->lprim[w][1]).(1:3)), SVec{3}((w->lprim[w][end]).(1:3)).-lpml[nP])

    @test get_pml_loc(lprim, Npml) == (lpml, Lpml)

    ω = rand()
    pml = PMLParam()
    σmax = (-(pml.m+1) * log(pml.R) / 2 ./ Lpml[nN], -(pml.m+1) * log(pml.R) / 2 ./ Lpml[nP])  # see calc_stretch_factor
    f(l, nw) = l≤lpml[nN][nw] ? (1 + σmax[nN][nw] * ((lpml[nN][nw]-l)/Lpml[nN][nw])^pml.m / (im*ω)) : (l≥lpml[nP][nw] ? (1 + σmax[nP][nw] * ((l-lpml[nP][nw])/Lpml[nP][nw])^pml.m / (im*ω)) : 1)
    sprim = (f.(lprim[1][1:end-1],1), f.(lprim[2][1:end-1],2), f.(lprim[3][1:end-1],3))
    sdual = (f.(ldual[1],1), f.(ldual[2],2), f.(ldual[3],3))

    @test gen_stretch_factor(ω, ((l->l[1:end-1]).(lprim), ldual), lpml, Lpml) == (sprim, sdual)
end

@testset "3D, Npml = 0" begin
    Npml = ([0,0,0], [0,0,0])
    lprim = ([0.0,1.0], [0.0,1.0], [0.0,1.0])  # with ghost points
    ldual = ([0.5], [0.5], [0.5])  # without ghost points

    lpml = ([0.0,0.0,0.0], [1.0,1.0,1.0])
    Lpml = ([0.0,0.0,0.0], [0.0,0.0,0.0])

    @test get_pml_loc(lprim, Npml) == (lpml, Lpml)

    ω = rand()
    @test gen_stretch_factor(ω, ((l->l[1:end-1]).(lprim), ldual), lpml, Lpml) == (([1.0],[1.0],[1.0]), ([1.0],[1.0],[1.0]))
end  # @testset "get_pml_loc"

end  # @testset "pml"
