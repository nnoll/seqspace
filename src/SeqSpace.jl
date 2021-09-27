module SeqSpace

using GZip
using BSON: @save
using LinearAlgebra: norm, svd, Diagonal
using Statistics: quantile, std
using Flux, Zygote

import BSON

include("io.jl")
include("rank.jl")
include("model.jl")
include("voronoi.jl")
include("distance.jl")
include("pointcloud.jl")

using .PointCloud, .DataIO, .SoftRank, .ML

export Result, HyperParams
export linearprojection, fitmodel, extendfit
export marshal, unmarshal

# ------------------------------------------------------------------------
# globals

# ------------------------------------------------------------------------
# types

mutable struct HyperParams
    dₒ :: Int          # output dimensionality
    Ws :: Array{Int,1} # (latent) layer widths
    BN :: Array{Int,1} # (latent) layers followed by batch normalization
    DO :: Array{Int,1} # (latent) layers followed by drop outs
    N  :: Int          # number of epochs to run
    δ  :: Int          # epoch subsample factor for logging
    η  :: Float64      # learning rate
    B  :: Int          # batch size
    V  :: Int          # number of points to partition for validation
    k  :: Int          # number of neighbors to use to estimate geodesics
    γₓ :: Float32      # prefactor of distance soft rank loss
    γᵤ :: Float32      # prefactor of uniform density loss
    γₗ :: Float32      # prefactor of latent space extra dimensions
    g  :: Function     # metric given to latent space
end

const euclidean²(x) = sum( (x[d,:]' .- x[d,:]).^2 for d in 1:size(x,1) )
function cylinder²(x)
    c = cos.(π.*(x[1,:]))
    s = sin.(π.*(x[1,:]))

    return (c' .- c).^2 .+ (s' .- s).^2 .+ euclidean²(x[2:end,:])
end

HyperParams(; dₒ=2, Ws=Int[], BN=Int[], DO=Int[], N=200, δ=10, η=1e-3, B=64, V=1, k=12, γₓ=1, γᵤ=1e-1, γₗ=100, g=euclidean²) = HyperParams(dₒ, Ws, BN, DO, N, δ, η, B, V, k, γₓ, γᵤ, γₗ, g)

struct Result
    param :: HyperParams
    loss  :: NamedTuple{(:train, :valid), Tuple{Array{Float64,1},Array{Float64,1}} }
    model
end

function marshal(r::Result)
    io = IOBuffer()
    BSON.bson(io, r.model)
    return Result(r.param,r.loss,take!(io))
end

function unmarshal(r::Result)
    io = IOBuffer(r.model)
    return Result(r.param,r.loss,BSON.load(io))
end

# ------------------------------------------------------------------------
# utility functions

# α := rescale data by
# δ := subsample data by
function pointcloud(;α=1, δ=1)
    verts, _ = open("$root/gut/mesh_apical_stab_000153.ply") do io
        read_ply(io)
    end

    return α*vcat(
        map(v->v.x, verts)',
        map(v->v.y, verts)',
        map(v->v.z, verts)'
    )[:,1:δ:end]
end

function expression(;raw=false)
    scrna, genes, _  = if raw
        GZip.open("$root/dvex/dge_raw.txt.gz") do io
            read_matrix(io; named_cols=false, named_rows=true)
        end
    else
        GZip.open("$root/dvex/dge_normalized.txt.gz") do io
            read_matrix(io; named_cols=true, named_rows=true)
        end
    end

    return scrna, genes
end

# ------------------------------------------------------------------------
# i/o

# assumes a BSON i/o
function load(io::IO)
    database = BSON.parse(io)
    result, input = database[:result], database[:in]
end

mean(x) = length(x) > 0 ? sum(x) / length(x) : 0
mean(x::Matrix;dims=1) = sum(x;dims=dims) / size(x,dims)

function cor(x, y)
    μ = (
        x=mean(x),
        y=mean(y)
    )
    var = (
       x=mean(x.^2) .- μ.x^2,
       y=mean(y.^2) .- μ.y^2
    )

    return (mean(x.*y) .- μ.x.*μ.y) / sqrt(var.x*var.y)
end

function buildloss(model, D², param)
    return function(x, i::T, log) where T <: AbstractArray{Int,1}
        z = model.pullback(x)
        y = model.pushforward(z)

        # reconstruction loss
        ϵᵣ = sum(sum((x.-y).^2, dims=2)) / sum(sum(x.^2,dims=2))

        # distance softranks
        Dz² = param.g(z)
        Dx² = D²[i,i]

        ϵₓ = mean(
            let
                dx, dz = Dx²[:,j], Dz²[:,j]
                rx, rz = softrank(dx ./ mean(dx)), softrank(dz ./ mean(dz))
                1 - cor(rx, rz)
            end for j ∈ 1:size(Dx²,2)
        )

        ϵᵤ = let
            a = Voronoi.volumes(z[1:2,:])
            std(a) / mean(a)
        end

        log && println(stderr, "ϵᵣ=$(ϵᵣ), ϵₓ=$(ϵₓ), ϵᵤ=$(ϵᵤ)")

        return ϵᵣ + param.γₓ*ϵₓ + param.γᵤ*ϵᵤ
    end
end

# ------------------------------------------------------------------------
# main functions

function linearprojection(x, d; Δ=1, Λ=nothing)
    Λ = isnothing(Λ) ? svd(x) : Λ

    ι = (1:d) .+ Δ
    ψ = Diagonal(Λ.S[ι])*Λ.Vt[ι,:]
    μ = mean(ψ, dims=2)

    x₀ = (Δ > 0) ? Λ.U[:,1:Δ]*Diagonal(Λ.S[1:Δ])*Λ.Vt[1:Δ,:] : 0
    return (
        projection = (ψ .- μ),
        embed = (x) -> (x₀ .+ (Λ.U[:,ι]*(x.+μ)))
    )
end

function fitmodel(data, param; D²=nothing)
    D² = isnothing(D²) ? geodesics(data, param.k).^2 : D²

    M = model(size(data,1), param.dₒ;
          Ws         = param.Ws,
          normalizes = param.BN,
          dropouts   = param.DO
    )

    nvalid = size(data,2) - ((size(data,2)÷param.B)-param.V)*param.B
    batch, index = validate(data, nvalid)

    loss = buildloss(M, D², param)
    E    = (
        train = Float64[],
        valid = Float64[]
    )

    log = (n) -> begin
        if (n-1) % param.δ == 0 
            @show n

            push!(E.train,loss(batch.train, index.train, true))
            push!(E.valid,loss(batch.valid, index.valid, true))
        end

        nothing
    end

    train!(M, batch.train, index.train, loss;
        η   = param.η,
        B   = param.B,
        N   = param.N,
        log = log
    )

    return Result(param, E, M), (batch=batch, index=index, D²=D², log=log)
end

function extendfit(result::Result, input, epochs)
    loss = buildloss(result.model, input.D², result.param)
    train!(result.model, input.y.train, input.index.train, loss; 
        η   = result.param.η,
        B   = result.param.B,
        N   = epochs,
        log = input.log
    )

    return result, input
end

end
