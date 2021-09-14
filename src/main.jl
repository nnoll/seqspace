module SeqSpace

using GZip
using BSON: @save
using LinearAlgebra: norm, svd, Diagonal
using Statistics: quantile, mean, std, cov, var
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
export run

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
    γₓ :: Float64      # prefactor of distance soft rank loss
    γᵤ :: Float64      # prefactor of uniform density loss
    γₗ :: Float64      # prefactor of latent space extra dimensions
    ψ  :: Function     # transformation applied to latent space before computing euclidean distance
end

HyperParams(; dₒ=3, Ws=Int[], BN=Int[], DO=Int[], N=200, δ=10, η=1e-3, B=64, V=81, k=12, γₓ=1, γᵤ=1e-1, γₗ=100, ψ=(x)->x) = HyperParams(dₒ, Ws, BN, DO, N, δ, η, B, V, k, γₓ, γᵤ, γₗ, ψ)

struct Result
    param :: HyperParams
    loss  :: NamedTuple{(:train, :valid), Tuple{Array{Float64,1},Array{Float64,1}} }
    model
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

ball(D, k) = [ sort(view(D,:,i))[k+1] for i in 1:size(D,2) ]

# ------------------------------------------------------------------------
# i/o

# TODO: add a save function

# assumes a BSON i/o
function load(io)
    database = BSON.parse(io)
    result, input = database[:result], database[:in]
end

# XXX: can you refactor to be less repetitive ?
function buildloss(model, D², param)
    if param.γᵤ == 0
        (x, i, log) -> let
            z = model.pullback(x)
            x̂ = model.pushforward(z)

            # reconstruction loss
            ϵᵣ = sum(sum((x.-x̂).^2, dims=2)) / sum(sum(x.^2,dims=2))

            # distance softranks
            D̂² = Distances.euclidean²(param.ψ(z))
            D̄² = D²[i,i]

            ϵₓ = mean(
                let
                    d, d̂ = D̄²[:,j], D̂²[:,j]
                    r, r̂ = softrank(d ./ mean(d)), softrank(d̂ ./ mean(d̂))
                    1 - cov(r, r̂)/sqrt(var(r)*var(r̂))
                end for j ∈ 1:size(D̂²,2)
            )

            if log
                @show ϵᵣ, ϵₓ
            end

            return ϵᵣ + param.γₓ*ϵₓ
        end
    else
        (x, i, log) -> let
            z = model.pullback(x)
            x̂ = model.pushforward(z)

            # reconstruction loss
            ϵᵣ = sum(sum((x.-x̂).^2, dims=2)) / sum(sum(x.^2,dims=2))

            # distance softranks
            D̂² = Distances.euclidean²(param.ψ(z))
            D̄² = D²[i,i]

            ϵₓ = mean(
                let
                    d, d̂ = D̄²[:,j], D̂²[:,j]
                    r, r̂ = softrank(d ./ mean(d)), softrank(d̂ ./ mean(d̂))
                    1 - cov(r, r̂)/sqrt(var(r)*var(r̂))
                end for j ∈ 1:size(D̂²,2)
            )

            # FIXME: allow for higher dimensionality than 2
            ϵᵤ = let
                a = Voronoi.areas(z[1:2,:])
                std(a) / mean(a)
            end

            ϵₗ = (size(z,1) ≤ 2) ? 0 : mean(sum(z[3:end,:].^2,dims=1))

            if log
                @show ϵᵣ, ϵₓ, ϵᵤ, ϵₗ
            end

            return ϵᵣ + param.γₓ*ϵₓ + param.γᵤ*ϵᵤ + param.γₗ*ϵₗ
        end
    end
end

# ------------------------------------------------------------------------
# main functions

function linearprojection(x, d; Δ=1, Λ=nothing)
    Λ = isnothing(Λ) ? svd(x) : Λ

    ι = (1:d) .+ Δ
    ψ = Diagonal(Λ.S[ι])*Λ.Vt[ι,:]
	μ = mean(ψ, dims=2)

    x₀ = Λ.U[:,1:Δ]*Diagonal(Λ.S[1:Δ])*Λ.Vt[1:Δ,:]
	return (
        projection = ψ .- μ,
        embed = (x) -> (x₀ .+ (Λ.U[:,ι]*(x.+μ)))
    )
end

function run(data, param; D²=nothing)
    D² = isnothing(D²) ? geodesics(data, param.k).^2 : D²

    M = model(size(data,1), param.dₒ;
          Ws         = param.Ws,
          normalizes = param.BN,
          dropouts   = param.DO
    )

    batch, index = validate(data, param.V)

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

function extendrun(result::Result, input, epochs)
    loss = buildloss(result.model, input.D², result.param)
    train!(result.model, input.y.train, input.index.train, loss; 
        η   = result.param.η, 
        B   = result.param.B, 
        N   = epochs,
        log = input.log
    )

    return result, input
end

function main(input, niter, output)
    params = Result[]
    open(input, "r") do io 
        params = eval(Meta.parse(read(io, String)))
    end

    result, data = run(params, niter)
    @save "$root/result/$output.bson" result data
end

end
