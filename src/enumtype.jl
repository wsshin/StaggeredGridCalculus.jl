export GridType, Sign  # enumerated types
export NP, nN, nP, nNP
export PD, nPR, nDL, nPD
export gt_w, numel, alter  # functions

# Signs
const nN, nP = 1, 2  # negative, positive directions
const nNP = SVector(nN,nP)  # tuple allocation is more efficient than array allocation
@enum Sign NEG=nN POS
const NP = SVector(NEG, POS)
for ins in instances(Sign); @eval export $(Symbol(ins)); end  # export all instances
Base.string(ins::Sign) = ins==NEG ? "negative" : "positive"
alter(ins::Sign) = ins==NEG ? POS : NEG


# Grid types
const nPR, nDL = 1, 2  # primal, dual grids
const nPD = SVector(nPR,nDL)  # tuple allocation is more efficient than array allocation
@enum GridType PRIM=nPR DUAL
const PD = SVector(PRIM, DUAL)
for ins in instances(GridType); @eval export $(Symbol(ins)); end  # export all instances
Base.string(ins::GridType) = ins==PRIM ? "primal" : "dual"
alter(ins::GridType) = ins==PRIM ? DUAL : PRIM

# Return grid types of the w-component of the field F, given the grid types gt₀ of the
# corners of the voxels whose edges are the field lines.
# If w is not one of the valid axes, return the voxel corner grid types gt₀.
gt_w(nw::Int,  # component index of field
     gt₀::SVector{K,GridType}  # grid type of corners of voxel whose edges are field lines
     ) where {K} =
    broadcast((a,w,g)->(a==w ? alter(g) : g), SVector(ntuple(identity, Val(K))), nw, gt₀)  # no change from gt₀ for nw ∉ {1,...,K}


# Functions for enumerated types
numel(::Type{T}) where {T<:Enum} = length(instances(T))
alter(n::Int) = 3-n
