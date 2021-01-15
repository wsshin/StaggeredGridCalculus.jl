export set_or_add!, calc_boundary_indices

# Usage:
# - set_or_add(t, i, j, k, s, Val(:(+=)))
# - set_or_add(t, i, j, k, s, Val(:(=)))
@generated function set_or_add!(t::AbsArrNumber, ind::Tuple{Vararg{Integer}}, s::Number, ::Val{OP}) where {OP}
    return Expr(OP, :(t[ind...]), :s)
end

function calc_boundary_indices(N::Tuple{Vararg{Int}})  # range of index: 1 through N
    Nₜ = nthreads()

    Nₑ = N[end]
    L = prod(N) ÷ Nₑ

    ∆n₀ = Nₑ ÷ Nₜ  # default value of entries of ∆n
    min_L∆n₀ = 50 * 50 * 3  # want ∆n₀ to satisfy L * ∆n₀ ≥ 50 * 50 * 3
    if L*∆n₀ < min_L∆n₀
        Nₜ = Int(cld(Nₑ, min_L∆n₀/L))  # cld to make Nₜ ≥ 1; min_L∆n₀/L is target number of entries in each chuck in last dimension
        ∆n₀ = Nₑ ÷ Nₜ
    end

    ∆n = SVec(ntuple(i->(i≤Nₑ%Nₜ ? ∆n₀+1 : ∆n₀), Nₜ))  # first N%Nₜ entries are ∆n₀+1; remaining entries are ∆n₀
    @assert sum(∆n)==Nₑ

    # nₛ = @MVector ones(Int, Nₜ)
    # for j = 2:Nₜ, i = 1:j-1
    #     nₛ[j] += ∆n[i]  # nₛ[1] = 1, nₛ[2] = 1 + ∆n[1], nₛ[3] = 1 + ∆n[1] + ∆n[2], ...
    # end
    nₛ = SVec(ntuple(i->1+sum(@view(∆n[1:i-1])), Nₜ))  # nₛ[1] = 1, nₛ[2] = 1 + ∆n[1], nₛ[3] = 1 + ∆n[1] + ∆n[2], ...

    # nₑ = @MVector zeros(Int, Nₜ)
    # for j = 1:Nₜ, i = 1:j
    #     nₑ[j] += ∆n[i]  # nₑ[1] = ∆n[1], nₑ[2] = ∆n[1] + ∆n[2], nₑ[3] = ∆n[1] + ∆n[2] + ∆n[3], ...
    # end
    nₑ = SVec(ntuple(i->sum(@view(∆n[1:i])), Nₜ))  # nₑ[1] = ∆n[1], nₑ[2] = ∆n[1] + ∆n[2], nₑ[3] = ∆n[1] + ∆n[2] + ∆n[3], ...

    return (nₛ, nₑ)
end

include("differential/differential.jl")
include("mean.jl")
