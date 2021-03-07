export t_ind, invert_∆l

# Below, t_ind returns
# - a tuple if the first argument is a tuple of tuple, and
# - an SVec if the first argument is a tuple of vector.

# Below, earlier methods delegate actions to later methods.  Maybe they need to be
# implemented separately for speed?

# t_ind(t::Tuple3{T}, ind::Tuple3{Int}) where {T} = (t[ind[1]], t[ind[2]], t[ind[3]])
# t_ind(t::Tuple32{T}, i::Int, j::Int, k::Int) where {T} = (t[1][i], t[2][j], t[3][k])  # i, j, k = 1 or 2

# From a tuple of two tuples 1 and 2, each with K entries, construct a tuple (with K entries)
# whose kth entry is the kth entry of either tuple 1 or tuple 2.
# E.g., t_ind(((0.1,0.2,0.3), (1.0,2.0,3.0)), 1, 2, 1) = (0.1, 2.0, 0.3)
@inline t_ind(t::Tuple23, i₁₂::T, j₁₂::T, k₁₂::T) where {T<:Union{GridType,Sign,Integer}} = t_ind(t, (i₁₂,j₁₂,k₁₂))  # i₁₂, j₁₂, k₁₂ = 1 or 2
@inline t_ind(t::Tuple2{NTuple{K}}, ind₁₂::CartesianIndex{K}) where {K} = t_ind(t, ind₁₂.I)  # ind₁₂.I: NTuple{K,Int}
@inline t_ind(t::Tuple2{NTuple{K}}, ind₁₂::SVec{K,T}) where {K,T<:Union{GridType,Sign,Integer}} = t_ind(t, ind₁₂.data)
@inline t_ind(t::Tuple2{NTuple{K}}, ind₁₂::NTuple{K,T}) where {K,T<:Union{GridType,Sign,Integer}} =  # ind₁₂[k] = 1 or 2
    map((t₁,t₂,i) -> Int(i)==1 ? t₁ : t₂, t[1], t[2], ind₁₂)  # NTuple{K}

# From a tuple of two SVecs 1 and 2, each with K entries, construct an SVec (with K
# entries) whose kth entry is the kth entry of either SVec 1 or SVec 2.
# E.g., t_ind((SVec(0.1,0.2,0.3), SVec(1.0,2.0,3.0)), 1, 2, 1) = SVec(0.1, 2.0, 0.3)
@inline t_ind(t::Tuple2{SVec{3}}, i₁₂::T, j₁₂::T, k₁₂::T) where {T<:Union{GridType,Sign}} = t_ind(t, SVec(i₁₂,j₁₂,k₁₂))
# @inline t_ind(t::Tuple2{SVec{K}}, ind₁₂::CartesianIndex{K}) where {K} = t_ind(t, ind₁₂.I)
@inline t_ind(t::Tuple2{SVec{K}}, ind₁₂::NTuple{K,T}) where {K,T<:Union{GridType,Sign}} = t_ind(t, SVec(ind₁₂))
@inline t_ind(t::Tuple2{SVec{K}}, ind₁₂::SVec{K,T}) where {K,T<:Union{GridType,Sign}} =  # ind₁₂[k] = 1 or 2
    map((t₁,t₂,i) -> Int(i)==1 ? t₁ : t₂, t[1], t[2], ind₁₂)  # SVec{K}

# From a tuple of K vectors, construct a vector with K entries whose kth entry is
# taken from the kth vector.  Which entry to take from the kth vector is specified
# by indices.
# E.g., t_ind(([0.1,0.2,0.3,0.4], [1.0,2.0,3.0,4.0], [10.0,20.0,30.0,40.0]), 3, 1, 4) = SVec(0.3, 1.0, 40.0)
@inline t_ind(t::Tuple3{AbsVec}, i::Int, j::Int, k::Int) = t_ind(t, SVec(i,j,k))
@inline t_ind(t::Tuple2{AbsVec}, i::Int, j::Int) = t_ind(t, SVec(i,j))
@inline t_ind(t::NTuple{K,AbsVec}, ind::CartesianIndex{K}) where {K} = t_ind(t, ind.I)
@inline t_ind(t::NTuple{K,AbsVec}, ind::NTuple{K,Int}) where {K} = t_ind(t, SVec(ind))
@inline t_ind(t::NTuple{K,AbsVec}, ind::SVec{K,Int}) where {K} = map((tₖ,iₖ) -> tₖ[iₖ], SVec(t), ind)  # SVec{K}


# Convenience function for 1D
function invert_∆l(∆l::Tuple2{AbsVecNumber})
    ∆l⁻¹ = invert_∆l((tuple(∆l[nPR]), tuple(∆l[nDL])))

    return (∆l⁻¹[nPR][1], ∆l⁻¹[nDL][1])
end

function invert_∆l(∆l::Tuple2{Tuple{Vararg{AbsVecNumber}}})  # ∆l[PR][k]: grid point spacings at primal grid point locations on k-axis
    ∆lprim, ∆ldual = ∆l
    ∆lprim⁻¹ = map(v->(1 ./ v), ∆lprim)
    ∆ldual⁻¹ = map(v->(1 ./ v), ∆ldual)

    return (∆lprim⁻¹, ∆ldual⁻¹)
end

function movingavg(l::AbsVec{T}) where {T<:Number}
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

function movingmin(l::AbsVec{T}) where {T<:Number}
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


# Numerical differentiation by forward difference.
f′fwd(f::Function, xₙ::Number, fₙ::Number=f(xₙ); h::Real=abs(xₙ)*Base.rtoldefault(Float)) = (f(xₙ+h) - fₙ) / h

function newtsol(x₀::Number, f::Function, f′::Function=(x,fₙ)->f′fwd(f,x,fₙ); rtol::Real=Base.rtoldefault(Float), atol::Real=eps(Float))
    # Solve f(x) = 0 using the Newton-Armijo method.
    # x₀: initial guess
    # f: function of x
    # f′: derivative of f at x.  If f′ can be evaluated using the value of f(x),
    #   write f′ such that it takes f(x) as the 2nd argument

    isconverged = true  # true if solution converged; false otherwise
    maxit = 100    # maximum iterations
    maxitls = 20   # maximum iterations inside the line search
    α = 1e-4
    maxs = 1 / Base.rtoldefault(Float)  # maximum step size
    perturbls = eps(Float)^0.75  # ≈ 1e-12, between sqrt(eps) and eps; some perturbation allowed in line search

    # Initialize.
    n = 0
    xₙ = float(x₀)
    T = typeof(xₙ)
    fₙ::T = f(xₙ)
    τf = rtol*abs(fₙ) + atol
    rx₀ = abs(x₀)
    has2ndarg = hasmethod(f′, Tuple2{Number})

    # Perform the Newton method.
    # @info "fₙ = $fₙ"
    while abs(fₙ) > τf
        # @info "n = $n"

        # abs(xₙ/x₀) ≤ 1e3 || (isconverged = false; break)
        # abs(xₙ/x₀) ≤ 1e3 || throw(ErrorException("Solution xₙ = $xₙ has diverged from x₀ = $x₀."))

        λ = 1.
        nls = 0  # line search iteration counter

        f′ₙ::T = has2ndarg ? f′(xₙ,fₙ) : f′(xₙ)

        # Avoid too large Newton steps.
        s = -fₙ/f′ₙ
        # @info "fₙ = $fₙ, f′ₙ = $f′ₙ, xₙ = $xₙ, s = $s"
        abs(s) ≤ maxs || (isconverged = false; break)
        # abs(s) ≤ maxs || throw(ErrorException("Newton step s = $s is larger than maximum step size $maxs."))
        xₙ₊₁ = xₙ + λ*s
        fₙ₊₁ = f(xₙ₊₁)

        # Perform the line search to determine λ.  The stopping criterion does not
        # have perturbls ≈ 1e-12 on the RHS, but I guess this kind of perturbation
        # allows update in xₙ even in the situation where line search is supposed
        # to fail.
        while abs(fₙ₊₁) ≥ (1 - α*λ) * abs(fₙ) + perturbls
            # @info "nls = $nls, fₙ₊₁ = $(fₙ₊₁)"
            λ /= 2
            xₙ₊₁ = xₙ + λ*s
            fₙ₊₁::T = f(xₙ₊₁)
            nls += 1

            # Too many iteration steps in line search
            nls ≤ maxitls || (isconverged = false; break)
            # nls ≤ maxitls || throw(ErrorException("Line search fails in $nls iteration steps."))
        end

        # Step accepted; continue the Newton method.
        xₙ = xₙ₊₁
        n += 1

        # Too many iteration steps in Newton's method.
        n ≤ maxit || (isconverged = false; break)
        # n ≤ maxit || throw(ErrorException("Newton method fails to converge in $n iteration steps."))

        fₙ = f(xₙ)
        # @info "fₙ = $fₙ"
    end
    # @info "Newton done"

    return xₙ, isconverged
end


# isapprox_kernel(x, y, rtol, atol) = norm2diff(x,y) ≤ atol + rtol * max(norm2(x), norm2(y))
# isapprox(x, y; rtol::Real=Base.rtoldefault(Float), atol::Real=eps(Float)) = isapprox_kernel(x, y, rtol, atol)

# # Compare x and y with respect to some large number L.  Useful when x or y is zero.
# # It is OK to use ≈ or ≉ for positive numbers (like ∆).
isapprox_wrt(x, y, L::Number, rtol::Real=Base.rtoldefault(Float)) = norm(x-y) ≤ rtol * abs(L)
# isapprox_wrt(x, y, L::Real, rtol::Real=Base.rtoldefault(Float)) = norm2diff(x,y) ≤ rtol * L

# # When this is enabled to support isapprox for arrays of tuples, gen_sublprim1d fails
# # at the line where curr[1:2] ≈ prev[end-1:end].  Don't understand why, because
# # the line uses isapprox for arrays of numbers.
# function isapprox(x::AbsArr{NTuple{N,T}}, y::AbsArr{NTuple{N,T}}; rtol::Real=Base.rtoldefault(Float), atol::Real=eps(Float)) where {N,T<:Number}
#     return isapprox_kernel(x, y, rtol, atol)
# end

function consolidate_similar!(l::AbsVecNumber, L::Number)
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

function findsim(l1::AbsVecNumber, l2::AbsVecNumber, L::Number)
    ind = Int[]
    for i = 1:length(l1)
        v1 = l1[i]
        for v2 = l2
            isapprox_wrt(v1, v2, L) && push!(ind, i)
        end
    end

    return ind
end
