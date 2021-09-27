module FitModel

using JLD2, FileIO
using LinearAlgebra
using Statistics, StatsBase
using Plots

import BSON

include("../src/scrna.jl")
include("../src/pointcloud.jl")
include("../src/distance.jl")
include("../src/SeqSpace.jl")

include("utils.jl")
using .CommandUtility, .SeqSpace

# ------------------------------------------------------------------------
# variable inputs

mutable struct Parameters
    hyper  :: HyperParams
    dim    :: Int
    plot   :: Bool
    metric :: Function
end

const default = Parameters(
    HyperParams(),
    50,
    true,
    Distances.euclidean
)

Parameters(;
    hyper=default.hyper,
    dim=default.dim,
    plot=default.plot,
    metric=default.metric,
) = Parameters(hyper,dim,plot,metric,niter)

# ------------------------------------------------------------------------
# plotting code

module Plot

using Plots

import ..PointCloud

function scaling(D, N; δ=2)
    ϕ, Rs = PointCloud.scaling(D, N)
    p = plot(Rs, ϕ[1:δ:end,:]',
           alpha=0.05,
           color=:red,
           xscale=:log10,
           yscale=:log10,
           label="",
    )
    plot!(Rs, 1e-4*Rs.^3,
           color=:black,
           linestyle=:dashdot,
           linewidth=2,
           label="",
    )
    xaxis!("ball size")
    yaxis!("number of points enclosed", (20,size(D,1)))

    return p
end

function isomapfit(ξ,D)
    c = [ cor(PointCloud.upper_tri(Distance.euclidean(ξ[:,1:i]')),
              PointCloud.upper_tri(D)) for i ∈ 1:10 ]
    p = plot(c, label="", linewidth=2, addmarker=true)

    xaxis!("number of isomap dimensions")
    yaxis!("pairwise distance correlation")

    return p
end

function fitequilibrate(loss)
    p = plot(loss.train,label="training",linewidth=3,yscale=:log10)
    plot!(loss.valid,label="validatation",linewidth=3,yscale=:log10)
    xlabel!("epoch")
    ylabel!("loss")

    return p
end

function fitgoodness(x, y)
    c = cor(x[:], y[:])
    p = histogram2d(x,y,title="corr=$(c)")
    xlabel!("data")
    ylabel!("predicted")

    return p
end

end

# ------------------------------------------------------------------------
# bulk functionality

function autoencode(input::AbstractString, param::Parameters, figs::AbstractString)
    alert("--> projecting data to linear subspace")
    data  = load(input, "data")
    input = linearprojection(data, param.dim)

    alert("--> computing geodesics")
    D₀ = param.metric(input.projection)
    χ  = percentile(upper_tri(D₀), 5)
    D  = geodesics(input.projection, param.k; D=D₀, accept=(d)->d≤χ)
    param.plot && let
        savefig(Plot.scaling(D₀, 100), "$figs/pointcloud_scaling_base.png")
        savefig(Plot.scaling(D,  100), "$figs/pointcloud_scaling_geodesic.png")
    end

    alert("--> computing isomap coordinates")
    ξ = real.(mds(D², 10))
    param.plot && savefig(Plot.isomapfit(ξ,D), "$figs/isomap_corr.png")

    alert("--> fitting autoencoder replicates")
    # TODO: add niter parameter
    result, metadata = fitmodel(input.projection, param.hyper; D²=D.^2)
    param.plot && let
        savefig(Plot.fitequilibrate(result.loss), "$figs/loss_dynamics.png")
        savefig(Plot.fitgoodness(input.projection, result.model.identity(input.projection)), "$figs/fit_residual.png")
    end

    return (
        input    = input,
        distance = D,
        isomap   = ξ,
        result   = marshal(result),
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
    length(args) > 1 && error("too many input paths")

    input = args[1]
    !isdir(input) && error("directory $input not found")

    outdir = dirname(params.o.value)
    !isdir(outdir) && mkdir(outdir)

    figdir = "$(outdir)/figs/model"
    !isdir(figdir) && mkpath(figdir)

    result = autoencode(input, params.p.value, figdir)
    # TODO: output file to correct path
end

end
