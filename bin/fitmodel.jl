module FitModel

using JLD2, FileIO
using LinearAlgebra
using Statistics, StatsBase
using Plots

import BSON

include("../src/scrna.jl")
include("../src/pointcloud.jl")
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
    PointCloud.geodesics
)

Parameters(;
    hyper=default.hyper,
    dim=default.dim,
    plot=default.plot,
    metric=default.metric,
) = Parameters(hyper,dim,plot)

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
end

end

# ------------------------------------------------------------------------
# bulk functionality

function autoencode(input::AbstractString, param::Parameters, figdir::AbstractString)
    data  = load(input, "data")
    input = linearprojection(data, param.dim)

    D = param.metric(input.projection, param.hyper.k)
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

    figdir = "$(outdir)/figs/model"
    !isdir(figdir) && mkpath(figdir)

    result = autoencode(input, params.p.value, figdir)
    # TODO: output file to correct path
end

end
