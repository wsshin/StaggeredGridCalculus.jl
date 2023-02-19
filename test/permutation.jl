@testset "permutation" begin

N = [3,4]
M = prod(N)
Z = zeros(M,M)

@testset "create_πcmp" begin
    cmpperm = [3,1,2]
    scale = [-2,3,-4]
    cmpfirst = false
    Kf = length(cmpperm)
    KfM = Kf * M

    # cmpfirst = false
    P = create_πcmp(N, cmpperm; scale, cmpfirst)
    P2 = zeros(KfM,KfM)
    Id = Matrix{eltype(scale)}(I, M, M)

    for nv = 1:Kf  # output component (row index of 3×3 block)
        nw = cmpperm[nv]  # input component (colunm index of 3×3 block)
        iblk = nv
        jblk = nw

        I = (iblk-1)*M+1:iblk*M
        J = (jblk-1)*M+1:jblk*M
        P2[I,J] = scale[nw] * Id
    end
    @test P2 == P

    P⁻¹ = create_inv_πcmp(N, cmpperm; scale, cmpfirst)
    @test P * P⁻¹ == I

    cmpfirst = true
    P_cmpfirst = create_πcmp(N, cmpperm; scale, cmpfirst)
    p = spzeros(eltype(scale), Kf, Kf)
    for nv = 1:Kf
        nw = cmpperm[nv]
        p[nv,nw] = scale[nw]
    end
    P2_cmpfirst = blockdiag(ntuple(k->p, M)...)
    @test P2_cmpfirst == P_cmpfirst

    P⁻¹_cmpfirst = create_inv_πcmp(N, cmpperm; scale)
    @test P_cmpfirst * P⁻¹_cmpfirst == I

end  # @testset "create_πcmp"

end  # @testset "permutation"
