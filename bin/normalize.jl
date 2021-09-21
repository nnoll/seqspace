module Normalize

using JLD2, FileIO
using LinearAlgebra, SpecialFunctions, NMF
using Distributions, Statistics, StatsBase
using Plots

include("../src/scrna.jl")
include("../src/mle.jl")
include("../src/util.jl")
include("../src/pointcloud.jl")

include("utils.jl")
using .CommandUtility

# ------------------------------------------------------------------------
# variable inputs

mutable struct Parameters
    threshold :: NamedTuple{(:gene,:cell), Tuple{Float64,Float64}}
    subdir    :: Union{Nothing, AbstractString}
    filter    :: Function
    plots     :: Bool
end

const default = Parameters(
    (gene = 5e-3, cell = 2e-1),
    nothing,
    (X) -> X,
    true
)

Parameters(;
    threshold=default.threshold,
    subdir=default.subdir,
    filter=default.filter,
    plots=default.plots
) = Parameters(threshold, subdir, filter, plots)

# ------------------------------------------------------------------------
# plotting code

module Plot

using Statistics, StatsBase
using Plots, ColorSchemes

rank(x) = invperm(sortperm(x))

cdf(x;  kwargs...) = plot(sort(x),  range(0,1,length=length(x)); kwargs...)
cdf!(x; kwargs...) = plot!(sort(x), range(0,1,length=length(x)); kwargs...)

function marginals(matrix; ϵ=1e-6)
    p₁ = cdf(vec(mean(data,dims=1)).+ϵ, xscale=:log10, label="", linewidth=2)
    xaxis!("mean count/cell")
    yaxis!("CDF")

    p₂ = cdf(vec(mean(data,dims=2)).+ϵ, xscale=:log10, label="", linewidth=2)
    xaxis!("mean count/gene")
    yaxis!("CDF")

    return plot(p₁, p₂)
end

function qq(x, model)
    p = scatter(rank(x), model.cumulative(param), linewidth=2, label="")
    plot!(0:1, 0:1, color=:red, linewidth=2, linestyle=:dashdot, label="ideal")
    xaxis!("empirical quantile")
    yaxis!("model quantile")

    return p
end

function mlefits(matrix, param)
    mₓ = vec(mean(matrix, dims=2))
    Mₓ = vec(maximum(matrix, dims=2))

    p₁ = scatter(mₓ, param.α,
        alpha=0.1,
        xscale=:log10,
        marker_z=log10.(Mₓ),
        label=false
    )
    xaxis!("expression / cell")
    yaxis!("estimated α")

    p₂ = scatter(mₓ, param.β,
        alpha=0.1,
        xscale=:log10,
        marker_z=log10.(Mₓ),
        label=false
    )
    xaxis!("expression / cell")
    yaxis!("estimated β")

    p₃ = scatter(mₓ, param.γ,
        alpha=0.1,
        xscale=:log10,
        yscale=:log10,
        marker_z=log10.(Mₓ),
        label=false
    )
    xaxis!("expression / cell")
    yaxis!("estimated γ")

    return p₁, p₂, p₃
end

function mpmaxeigval(λ, x₀, rank)
    p = plot(λ, 1:length(λ), 
             xscale=:log10, 
             yscale=:log10, 
             linewidth=2, 
             label="Empirical distribution"
    )

    vline!([x₀], linewidth=2, linestyle=:dashdot, label="MP maximum eigenvalue")
    title!("scRNAseq data (k=$rank)")
    xaxis!("singular value")
    yaxis!("CDF")

    p
end

end

# ------------------------------------------------------------------------
# the bulk of the code

alert(msg) = println(stderr, msg)

function data(dir::AbstractString, subdir, filter::Function)
    N = if subdir === nothing
        scRNA.load(dir)
    else
        reduce(∪, scRNA.load("$dir/$d") for d ∈ readdir(dir) if occursin(subdir,d))
    end

    return filter(N)
end

function cutoff(counts, threshold)
    x = scRNA.filtergene(x) do gene, _
        mean(gene) >= threshold.gene && length(unique(gene)) > 3
    end

    x = scRNA.filtercell(x) do cell, _
        mean(cell) >= threshold.cell
    end
end

function normalize(dir::AbstractString, param::Parameters, figs::AbstractString)
    # load in raw data. impose data specific filter
    alert("--> loading raw data")
    counts = data(dir, param.subdir, param.filter)
    param.plot && savefig(Plot.marginals(counts), "$figs/raw_count_marginals.png")

    # impose (conservative) count-based cutoffs
    alert("--> threshold raw data")
    counts = cutoff(counts, param.threshold)
    param.plot && savefig(Plot.marginals(counts), "$figs/threshold_count_marginals.png")

    # estimate marginalized overdispersion distribution from highly expressed genes
    alert("--> estimating overdispersion prior")
    p₀ = MLE.fit_glm(:negative_binomial, X;
        Γ=(β̄=1, δβ¯²=10, Γᵧ=nothing),
        run=(x) -> mean(x) > 1
    )
    param.plot && savefig(
        Plot.cdf(p₀.γ, xlabel="estimated γ", ylabel="CDF", xscale=:log10, label="", linewidth=2),
        "$figs/overdispersion_distribution_estimate.png"
    )

    logγ  = log.(p₀.γ)
    model = MLE.generalized_normal(logγ)
    param = MLE.fit(model)
    param.plot && savefig(Plot.qq(logγ, model, param), "$figs/overdispersion_distribution_fig.png")

    # use overdispersion distribution as prior into the full fit
    alert("--> fitting overdispersion per gene")
    p₀ = MLE.fit_glm(:negative_binomial, X; Γ=(β̄=1, δβ¯²=10, Γᵧ=param))
    param.plot && let
        p = Plot.mlefits(counts, p₀)
        savefig(p[1], "$figs/nb_α_fit.png")
        savefig(p[2], "$figs/nb_β_fit.png")
        savefig(p[3], "$figs/nb_γ_fit.png")
    end

    # normalize variance of count matrix. estimate rank
    alert("--> normalizing count variance & estimating rank")
    N, σ², u², v² = let
        σ² = counts.*(counts.+p₀.γ) ./ (1 .+ p₀.γ)
        u², v², _ = Utility.sinkhorn(σ²)

        (Diagonal(.√u²) * counts * Diagonal(.√v²)), (Diagonal(u²) * σ² * Diagonal(v²)), u², v²
    end

    λ = svdvals(N)
    R = sum(λ .> (sqrt(size(N,1))+sqrt(size(N,2)))) + param.δ
    param.plot && savefig(Plot.mpmaxeigval(λ, √(size(N,1)) + √(size(N,2)), R), "$figs/rank_estimation.png")

    # reduce rank to signal directions
    N, c = let
        r = nnmf(N, R; alg=:cd)
        m = r.W*r.H
        m, cor(m[:], N[:])
    end

    # renormalize
    alert("--> normalizing reduced rank matrix")
    N, u, v = let
        u, v, _ = Utility.sinkhorn(Y)
        (Diagonal(u) * N * Diagonal(v)), u, v
    end

    return (
        raw       = counts.data,
        gene      = counts.gene,
        cell      = counts.cell,
        rank      = R,
        corr      = c,
        data      = N,
        varnorm   = (row=u², col=v²),
        scalenorm = (row=u,  col=v),
        mleparam  = p₀,
    )
end

# ------------------------------------------------------------------------
# main point of entry

if abspath(PROGRAM_FILE) == @__FILE__
    params = (
        p = Arg("inpath", loadparams, default),
        o = Arg("outpath", (path)->path, nothing)
    )

    args = argparse(ARGS,params)
    if length(args) > 1
        error("too many input paths")
    end

    input = args[1]
    !isdir(input) && error("directory $input not found")

    outdir = dirname(params.o.value)
    !isdir(outdir) && mkdir(outdir)

    figdir = "$(outdir)/figs/norm"
    !isdir(figdir) && mkpath(figdir)

    result = normalize(input, params.p.value, figdir)
    # TODO: output file to correct path
end

end
