@testset "permutation" begin

N = [3,4]
M = prod(N)
Z = zeros(M,M)

@testset "create_πcmp" begin
    permute = [3,1,2]
    scale = [-1,1,-1]
    Kf = length(permute)
    KfM = Kf * M

    # order_cmpfirst = false
    P = create_πcmp(N, permute, scale=scale, order_cmpfirst=false)
    P2 = zeros(KfM,KfM)
    Id = Matrix{eltype(scale)}(I, M, M)

    for nw = 1:Kf
        iblk = permute[nw]
        jblk = nw
        I = (iblk-1)*M+1:iblk*M
        J = (jblk-1)*M+1:jblk*M
        P2[I,J] = scale[nw] * Id
    end
    @test P2 == P

    # order_cmpfirst = true
    P_cmpfirst = create_πcmp(N, permute, scale=scale)
    p = spzeros(eltype(scale), Kf, Kf)
    for nw = 1:Kf
        p[permute[nw], nw] = scale[nw]
    end
    P2_cmpfirst = blockdiag(ntuple(k->p, M)...)
    @test P2_cmpfirst == P_cmpfirst
end  # @testset "create_πcmp"

end  # @testset "permutation"
