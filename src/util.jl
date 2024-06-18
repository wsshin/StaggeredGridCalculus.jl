export getindex_col, invert_∆l

# Below, t_ind returns
# - a tuple if the first argument is a tuple of tuple, and
# - an SVector if the first argument is a tuple of vector.

# Below, earlier methods delegate actions to later methods.  Maybe they need to be
# implemented separately for speed?

# t_ind(t::NTuple{3,T}, ind::NTuple{3,Int}) where {T} = (t[ind[1]], t[ind[2]], t[ind[3]])
# t_ind(t::Tuple32{T}, i::Int, j::Int, k::Int) where {T} = (t[1][i], t[2][j], t[3][k])  # i, j, k = 1 or 2

# From a tuple of two tuples 1 and 2, each with K entries, construct a tuple (with K entries)
# whose kth entry is the kth entry of either tuple 1 or tuple 2.
# E.g., getindex_col(((0.1,0.2,0.3), (1.0,2.0,3.0)), 1, 2, 1) = (0.1, 2.0, 0.3)
@inline getindex_col(t::NTuple{2,NTuple{K}}, ind₁₂::Vararg{<:AbstractBinaryType,K}) where {K} = getindex_col(t, ind₁₂)  # ind = i₁₂, j₁₂, k₁₂, ..., each of whihc is 1 or 2
@inline getindex_col(t::NTuple{2,NTuple{K}}, ind₁₂::CartesianIndex{K}) where {K} = getindex_col(t, ind₁₂.I)  # ind₁₂.I: NTuple{K,Int}
@inline getindex_col(t::NTuple{2,NTuple{K}}, ind₁₂::NTuple{K,Any}) where {K} = getindex_col(t, SVector(ind₁₂))
@inline getindex_col(t::NTuple{2,NTuple{K}}, ind₁₂::SVector{K}) where {K} =  # ind₁₂[k] = 1 or 2
    ntuple(k->@inbounds(t[ind₁₂[k]])[k], K)  # NTuple{K}; without @inbounds, this is allocating for SVector{K} of abstract type such as AbstractBoundType

# From a tuple of two SVectors 1 and 2, each with K entries, construct an SVector (with K
# entries) whose kth entry is the kth entry of either SVector 1 or SVector 2.
# E.g., getindex_col((SVector(0.1,0.2,0.3), SVector(1.0,2.0,3.0)), 1, 2, 1) = SVector(0.1, 2.0, 0.3)
@inline getindex_col(t::NTuple{2,SVector}, args...) = SVector(getindex_col((t[1].data, t[2].data), args...))

# From a tuple of K vectors, construct a vector with K entries whose kth entry is
# taken from the kth vector.  Which entry to take from the kth vector is specified
# by indices.
# E.g., t_ind(([0.1,0.2,0.3,0.4], [1.0,2.0,3.0,4.0], [10.0,20.0,30.0,40.0]), 3, 1, 4) = SVector(0.3, 1.0, 40.0)
# When K = 2, this could be ambiguous with respect to the above.  However, this mainly
# designed for t that is a tuple of non-SVector, for which the ambiguity doesn't exist.
#
# Still, this is confusing.  I'm just going to use getindex.(t, ind).

# @inline t_ind(t::NTuple{K,AbstractVector}, ind::Vararg{Any,K}) where {K} = t_ind(t, ind)
# @inline t_ind(t::NTuple{K,AbstractVector}, ind::CartesianIndex{K}) where {K} = t_ind(t, ind.I)
# @inline t_ind(t::NTuple{K,AbstractVector}, ind::NTuple{K,Int}) where {K} = t_ind(t, SVector(ind))
# @inline t_ind(t::NTuple{K,AbstractVector}, ind::SVector{K,Int}) where {K} =
#     SVector(ntuple(k->t[k][ind[k]], K))  # SVector{K}


function invert_∆l(∆l::NTuple{2,Tuple{Vararg{AbstractVector{<:Number}}}})  # ∆l[PR][k]: grid point spacings at primal grid point locations on k-axis
    ∆lprim, ∆ldual = ∆l
    ∆lprim⁻¹ = map(v->(1 ./ v), ∆lprim)
    ∆ldual⁻¹ = map(v->(1 ./ v), ∆ldual)

    return (∆lprim⁻¹, ∆ldual⁻¹)
end

function movingavg(l::AbstractVector{T}) where {T<:Number}
    # Return (l[1:end-1] + l[2:end]) / 2

    n = length(l)
    if n ≤ 1  # n = 0 or 1
        lmov = float(T)[]
    else  # n ≥ 2
        lmov = Vector{float(T)}(undef, n-1)
        for i = 1:n-1
            lmov[i] = (l[i] + l[i+1]) / 2
        end
    end

    return lmov
end

function movingmin(l::AbstractVector{T}) where {T<:Number}
    # Return min.([l; Inf], [Inf; l])

    n = length(l)
    if n == 0
        lmov = [Inf]
    else  # n ≥ 1
        lmov = Vector{T}(undef, n+1)
        lmov[1] = l[1]
        lmov[n+1] = l[n]
        for i = 2:n
            lmov[i] = min(l[i-1], l[i])
        end
    end

    return lmov
end

# isapprox_kernel(x, y, rtol, atol) = norm2diff(x,y) ≤ atol + rtol * max(norm2(x), norm2(y))
# isapprox(x, y; rtol::Real=τᵣ₀, atol::Real=τₐ₀) = isapprox_kernel(x, y, rtol, atol)

# # Compare x and y with respect to some large number L.  Useful when x or y is zero.
# # It is OK to use ≈ or ≉ for positive numbers (like ∆).
isapprox_wrt(x, y, L::Number, rtol::Real=τᵣ₀) = norm(x-y) ≤ rtol * abs(L)
# isapprox_wrt(x, y, L::Real, rtol::Real=τᵣ₀) = norm2diff(x,y) ≤ rtol * L

# # When this is enabled to support isapprox for arrays of tuples, gen_sublprim1d fails
# # at the line where curr[1:2] ≈ prev[end-1:end].  Don't understand why, because
# # the line uses isapprox for arrays of numbers.
# function isapprox(x::AbsArr{NTuple{N,T}}, y::AbsArr{NTuple{N,T}}; rtol::Real=τᵣ₀, atol::Real=τₐ₀) where {N,T<:Number}
#     return isapprox_kernel(x, y, rtol, atol)
# end

function consolidate_similar!(l::AbstractVector{<:Number}, L::Number)
    # Consolidate approximately equal points.  Assume l is sorted.

    # Original code:
    # ind_unique = (≉).(@view(lprim0[1:end-1]), @view(lprim0[2:end]))  # length(ind_unique) = length(lprim0) - 1
    # push!(ind_unique, true)
    # lprim0 = lprim0[ind_unique]  # [lprim0[1:end-1][original ind_unique]; lprim0[end]]

    if !isempty(l)
        ind_sim = findall(isapprox_wrt.(@view(l[1:end-1]), @view(l[2:end]), L))

        # Below, l[end] must be always included as a unique entry.  There are two cases: where
        # l[end-1] is included as a unique entry or not.  If l[end-1] is included, it must be
        # very different from l[end], so we must include l[end].  If l[end-1] is not included,
        # it must be approximately l[end], but because l[end-1] is not included, we must include
        # l[end].  In other words, l[end] represents the last set of approximately equal entries
        # in any cases.
        deleteat!(l, ind_sim)
        @assert !isapprox_wrt(l[end], l[end-1], L)  # last entry is always unique
    end
end

function findsim(l1::AbstractVector{<:Number}, l2::AbstractVector{<:Number}, L::Number)
    ind = Int[]
    for i = 1:length(l1)
        v1 = l1[i]
        for v2 = l2
            isapprox_wrt(v1, v2, L) && push!(ind, i)
        end
    end

    return ind
end
