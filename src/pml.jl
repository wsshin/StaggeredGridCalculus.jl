# About PML setup
#
# - Let's define the computational domain as the region where we perform computation, and
# the physical domain where actual physical phenomena are simulated.  The physical domain is
# the computational domain outside PML, because PML is an artificial region that is
# introduced to absorb physical waves.
#
# Currently, we specify the size of the computational domain first, and then specify PML
# such that it "eats in" the computational domain to leave the physical domain.  This means
# you always need to define the computational domain larger than the physical domain by the
# PML thickness.
#
# This seems inconvenient.  It may seem more convenient to specify the physical domain first,
# specify the PML thickness, and then construct the computational domain by attaching PML
# to the physical domain.  That way, we would be able to turn on and off the absorbing
# boundary without changing the size of the physical domain of interest.  Also, increasing
# the PML thickness would be done simply by increasing the PML thickness, wherease with the
# current method it requires increasing the computational domain as well to keep the size of
# the physical domain the same.
#
# However, there is a reason for the current method.  One of the most popular use cases of
# PML is the simulation of an infinite waveguide, where we need to define the waveguide
# inside PML.  Therefore, we cannot "hide" PML from the users: we have to let the users put
# objects inside PML.  In that case, it is better to let the users specify the size of the
# region including PML, because that way the users would know how deep they should place the
# waveguide.
#
# You way think that you hide PML from the users by automatically constructing the material
# parameters inside PML, by taking the material parameters on the boundary of the physical
# domain and extending it homogeneously into PML.  However, such a homogeneous extension is
# not always what we want.  For example, to construct PML for a phothonic crystal waveguide,
# we have to specify the photonic crystal structure inside PML.  In that case the material
# composition inside PML is not simply the homogeneous extension of their values at the PML
# boundary
#
# - Another question that could be raised about the current method is the way to set the
# PML thickness.  Currently we specify the number of grid cells inside PML and multiply the
# cell size ∆l to figure out the PML thickness, rather than specifying the PML thickness and
# divide it by ∆l to figure out the number of grid cells inside PML.
#
# The latter look like a better method, because the reflectance of PML is a function of the
# PML loss parameter and thickness.  Therefore, for a given PML loss parameter, the
# reflectance does not change with the spatial resolution of the grid if the PML thickness
# is maintained.  Conversely, if we fix the number of grid cells inside PML, the PML
# thickness changes with the spacial resolution, so the PML loss parameter needs to change
# as well to achieve the same PML reflectance.
#
# However, in my experience, when using iterative methods to solve Maxwell's equations, the
# number of grid cells inside PML affects the convergence of iterative methods significantly,
# regardless of the spatial resolution.  For example, when I doubled the spatial resolution
# while keeping the PML thickness and therefore doubled the number of grid cells inside PML,
# the iterative methods no longer converged.  On the other hand, even if I doubled the
# spatial resolution, if I keep the number of grid cells inside PML to halve the PML
# thickness, I still got convergence.
#
# Of course, I have to use a stronger PML loss parameter to achieve the same reflectance
# with a thinner PML, and if we use a too strong loss parameter, the discretized PML model
# deviates from the continuous model and we get stronger reflectance than the target value.
# Still, I found that the reflectance remains pretty low, and the convergence of iterative
# methods was more important.  So, I decided to specify the number of grid cells inside PML
# rather than the PML thickness.  This decision may change in the future.

export PMLParam
export create_stretched_∆l

mutable struct PMLParam
    m::Float64  # degree of polynomial grading for σ and κ
    R::Float64  # R: target reflection coefficient of PML
    κmax::Float64  # maximum of real stretch factor
    amax::Float64  # maximum of complex frequency shift
    ma::Float64  # degree of polynomial grading for a
    PMLParam() = new(4.0, exp(-16), 1.0, 0.0, 4.0)
end

function create_stretched_∆l(ω::Number, ∆l::NTuple{2,AbstractVector{<:Real}}, l::NTuple{2,AbstractVector{<:Real}},
                             bounds::NTuple{2,Real}, Npml::NTuple{2,Integer}, pml::PMLParam=PMLParam())
    s∆l = create_stretched_∆l(ω, tuple.(∆l), tuple.(l), SVector{1}.(bounds), SVector{1}.(Npml), pml)

    return (s∆l[nPR][1], s∆l[nDL][1])
end

# Wrapper for converting AbstractVector's to SVector's
create_stretched_∆l(ω::Number, grid::Grid{K}, Npml::NTuple{2,AbstractVector{<:Integer}}, pml::PMLParam=PMLParam()) where {K} =
    create_stretched_∆l(ω, grid, SVector{K}.(Npml), pml)

create_stretched_∆l(ω::Number, ∆l::NTuple{2,NTuple{K,AbstractVector{<:Real}}}, l::NTuple{2,NTuple{K,AbstractVector{<:Real}}},
                    bounds::NTuple{2,AbstractVector{<:Real}}, Npml::NTuple{2,AbstractVector{<:Integer}}, pml::PMLParam=PMLParam()) where {K} =
    create_stretched_∆l(ω, ∆l, l, SVector{K}.(bounds), SVector{K}.(Npml), pml)

# Create vectors containig a stretched version of a given ∆l.
create_stretched_∆l(ω::Number, grid::Grid{K}, Npml::NTuple{2,SVector{K,<:Integer}}, pml::PMLParam=PMLParam()) where {K} =
    create_stretched_∆l(ω, grid.∆l, grid.l, grid.bounds, Npml, pml)

function create_stretched_∆l(ω::Number,  # angular frequency
                             ∆l::NTuple{2,NTuple{K,AbstractVector{<:Real}}},  # grid point spacings at primal and dual grid point locations
                             l::NTuple{2,NTuple{K,AbstractVector{<:Real}}},  # locations of primal and dual grid points w/o ghost points
                             bounds::NTuple{2,SVector{K,<:Real}},  # bounds[NEG][k] = boundary of domain at (-) end in k-direction
                             Npml::NTuple{2,SVector{K,<:Integer}},  # Npml[NEG][k] = number of cells inside PML at (-) end in k-direction
                             pml::PMLParam=PMLParam()  # PML parameters
                             ) where {K}
    if all(iszero.(Npml))
        s∆lprim = ∆l[nPR]  # AbstractVector{<:Real}
        s∆ldual = ∆l[nDL]  # AbstractVector{<:Real}
    else
        lpml, Lpml = pml_loc(l[nPR], bounds, Npml)
        sfactor = create_sfactor(ω, l, lpml, Lpml, pml)

        s∆lprim = map((x,y)->x.*y, sfactor[nPR], ∆l[nPR])  # AbstractVector{<:Number}
        s∆ldual = map((x,y)->x.*y, sfactor[nDL], ∆l[nDL])  # AbstractVector{<:Number}
    end

    s∆l = (s∆lprim, s∆ldual)

    return s∆l
end

# Wrapper for converting AbstractVector's to SVector's
create_sfactor(ω::Number, l::NTuple{2,NTuple{K,AbstractVector{<:Real}}}, lpml::NTuple{2,AbstractVector{<:Real}},
               Lpml::NTuple{2,AbstractVector{<:Real}}, pml::PMLParam=PMLParam()) where {K} =
    create_sfactor(ω, l, SVector{K}.(lpml), SVector{K}.(Lpml), pml)

# Create vectors of stretching factors.
function create_sfactor(ω::Number,  # angular frequency
                        l::NTuple{2,NTuple{K,AbstractVector{<:Real}}},  # locations of primal and dual grid points
                        lpml::NTuple{2,SVector{K,<:Real}},  # locations of PML interfaces
                        Lpml::NTuple{2,SVector{K,<:Real}},  # thicknesses of PML
                        pml::PMLParam=PMLParam()  # PML parameters
                        ) where {K}
    N = length.(l[nPR])  # (Nx, Ny, Nz)
    sfactor = (ones.(ComplexF64,N), ones.(ComplexF64,N))
    for k = 1:K
        lpmlₙ, lpmlₚ = lpml[nN][k], lpml[nP][k]
        Lpmlₙ, Lpmlₚ = Lpml[nN][k], Lpml[nP][k]
        for ngt = nPD
            r = l[ngt][k]
            s = sfactor[ngt][k]
            for i = 1:N[k]
                r[i] < lpmlₙ && (s[i] = calc_sfactor(ω, lpmlₙ-r[i], Lpmlₙ, pml))
                r[i] > lpmlₚ && (s[i] = calc_sfactor(ω, r[i]-lpmlₚ, Lpmlₚ, pml))
            end
        end
    end

    return sfactor
end

# Calculate the stretch factor at a given depth into PML.
function calc_sfactor(ω::Number,  # angular frequency
                      d::Real,  # depth into PML
                      Lpml::Real,  # thickness of PML
                      pml::PMLParam)  # PML parameters
    σmax = -(pml.m+1) * log(pml.R) /2Lpml  # -(m+1) ln(R) / (2 η Lpml), where η = 1 in units of η₀
    σ = σmax * (d/Lpml)^pml.m
    κ = 1 + (pml.κmax-1) * (d/Lpml)^pml.m
    a = pml.amax * (1 - d/Lpml)^pml.ma

    s_factor = κ + σ / (a + im*ω)  # s = κ + σ/(a + i ω ε), where ε = 1 in units of ε₀
end

# Wrapper for converting AbstractVector's to SVector's
pml_loc(lprim::NTuple{K,AbstractVector{<:Real}}, bounds::NTuple{2,AbstractVector{<:Real}}, Npml::NTuple{2,AbstractVector{<:Integer}}) where {K} =
    pml_loc(lprim, SVector{K}.(bounds), SVector{K}.(Npml))

# Get the PML interface locations.
# The primal grid plane locations *without* the ghost plane locations are passed, so the
# positive-end boundaries of the computational domain cannot be retrieved from them.
# Therefore, the boundary information is passed separately.
function pml_loc(lprim::NTuple{K,AbstractVector{<:Real}},  # lprim[k] = primal vertex locations in k-direction (excluding ghost location at (+) end)
                 bounds::NTuple{2,SVector{K,<:Real}},  # bounds[NEG][k] = boundary of domain at (-) end in k-direction
                 Npml::NTuple{2,SVector{K,<:Integer}}  # Npml[NEG][k] = number of cells inside PML at (-) end in k-direction
                 ) where {K}
    # Below, if Npml[nP] = 0, lprim[end+1-Npml[nP]] = lprim[end+1] is out of bound.  Because
    # lprim[end+1] points to the ghost point, we return the positive-end boundary in this
    # case.
    lpml = (map((v,n)->v[1+n], lprim, Npml[nN]), map((v,b,n)->(n==0 ? b : v[end+1-n]), lprim, bounds[nP], Npml[nP]))  # NTuple{2,SFloat{K}}
    Lpml = (lpml[nN]-bounds[nN], bounds[nP]-lpml[nP])  # NTuple{2,SFloat{K}}

    return lpml, Lpml
end
