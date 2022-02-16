module Drosophila

using GZip
using LinearAlgebra
using Statistics, StatsBase

include("../src/scrna.jl")
include("../src/normalize.jl")

# ------------------------------------------------------------------------
# globals

const DATA = "proc/drosophila"

# ------------------------------------------------------------------------
# data i/o

module DataIO

function counts(io::IO)
	genes = String[]
	lines = [
		let
			entry = split(line)
			push!(genes, entry[1])
			parse.(Int, entry[2:end])
		end for line in eachline(io)
	]
	count = Matrix(reduce(hcat,lines)')
	return count, genes
end

function counts(dir::String, subdir::String)
	count = reduce(∪,
		scRNA.load("$dir/$d") for d ∈ readdir(dir) if occursin(subdir,d)
	)
			
    # remove any non melongaster cells (experiment specific)
	markers  = (
		dvir = scRNA.searchloci(count, "Dvir_")
	)

	count = scRNA.filtercell(count) do cell, _
		sum(cell[markers.dvir]) < .10*sum(cell)
	end
			
    # remove any low count genes or non melongaster specific genes
	count = scRNA.filtergene(count) do count, gene
		!occursin("Dvir_", gene) && sum(count) > 20
	end
				

    # remove yolk and pole cells
	markers  = (
		yolk = scRNA.locus(count, "sisA", "CG8129", "Corp", "CG8195", "CNT1", "ZnT77C"),
		pole = scRNA.locus(count, "pgc"),
	)
			
	count = scRNA.filtercell(count) do cell, _
		(sum(cell[markers.yolk]) < 10
	  && sum(cell[markers.pole]) < 3
	  && sum(cell) > 1e4)
	end

    # remove any lowly expressed genes
	count = scRNA.filtergene(count) do count, _
		sum(count) > 20
	end

	return count.data, count.gene
end

end

# ------------------------------------------------------------------------
# figures

module Figures

using LaTeXStrings
using Makie, CairoMakie
using Statistics, StatsBase

using ..Normalize

const FIG = "figs/drosophila"

function mkpath()
    isdir(FIG) || Base.mkpath(FIG)
end

function heteroskedastic(raw)
    fig = Figure(font="Latin Modern Math", fontsize=26)
	
	ax1 = Axis(
		fig[1,1],
		xscale=Makie.log10,
		xlabel="cell sequencing depth",
		ylabel="cumulative",
        yticks=([0, 0.25, 0.5, 0.75, 1.0], [L"0.0", L"0.25", L"0.50", L"0.75", L"1.00"])
	)

	ax2 = Axis(
		fig[1,2],
		xlabel="gene sequencing depth",
		xscale=Makie.log10,
        xticks=([1e1, 1e2, 1e3, 1e4, 1e5, 1e6], [L"10^1", L"10^2", L"10^3", L"10^4", L"10^5", L"10^6"]),
		ylabel="cumulative",
        yticks=([0, 0.25, 0.5, 0.75, 1.0], [L"0.0", L"0.25", L"0.50", L"0.75", L"1.00"])
	)

	depth = vec(sum(raw,dims=1))
	level = vec(sum(raw,dims=2))
	
	lines!(ax1, sort(depth), range(0,1,size(raw,2)),
		linewidth=2,
		color=:black
	)
    xlims!(ax1, (1e4, 3e5))
	
	lines!(ax2, sort(level), range(0,1,size(raw,1)),
		linewidth=2,
		color=:black
	)

    save("$FIG/heteroskedastic.png", fig, px_per_unit=2)
end

function overdispersed(raw)
	depth = sum(raw,dims=1)
    χ = Normalize.logmean(depth|>vec) .* (raw ./ depth)

	# figure 1
    fig = Figure(font="Latin Modern Math", fontsize=26)
	ax1 = Axis(fig[1,1],
		xscale=Makie.log10,
		yscale=Makie.log10,
		xlabel="mean / gene",
        xticks=([1e-2, 1e0, 1e2, 1e4], [L"10^{-2}", L"10^0", L"10^2", L"10^4"]),
		ylabel="variance / gene",
        yticks=([1e-2, 1e0, 1e2, 1e4], [L"10^{-2}", L"10^0", L"10^2", L"10^4"]),
    )

	scatter!(ax1,
		vec(mean(χ,dims=2)), vec(var(χ,dims=2)),
		color=(:black,0.01),
	)
	lines!(ax1,
		1e-2:1e4, 1e-2:1e4,
		linewidth=2,
		color=:red,
		linestyle=:dashdot,
		label="poisson",
	)
	ylims!(ax1, 1e-2, 1e4)
	axislegend(ax1, position = :rb)
    save("$FIG/overdispersed_mean_vs_variance.png", fig, px_per_unit=2)

	# figure 2
    fig = Figure(font="Latin Modern Math", fontsize=26)
	ax2 = Axis(fig[1,1],
		xscale=Makie.pseudolog10,
		xlabel="expected number of zeros (poisson)",
        xticklabelrotation=π/5,
		yscale=Makie.pseudolog10,
		ylabel="measured number of zeros",
    )

	g = vec(mean(χ,dims=2))
	z = vec(sum(χ.==0, dims=2))

	scatter!(ax2,
		size(χ,2)*exp.(-g),
		z,
		color=(:black,0.03),
	)
	lines!(ax2,
		1:1500, 1:1500,
		linewidth=2,
		color=:red,
		linestyle=:dashdot,
		label="poisson",
	)
	axislegend(ax2, position = :rb)

    save("$FIG/overdispersed_zeros.png", fig, px_per_unit=2)
end

function nb_uncertainty(raw, model)
    χ = [ gene |> vec |> Normalize.logmean for gene in eachrow(raw) ]
	
    # figure 1
	fig = Figure(font="Latin Modern Math", fontsize=26)
	ax1 = Axis(fig[1,1],
		xscale=Makie.log10,
		yscale=Makie.log10,
		xlabel=L"\chi",
        xticks=([1e-2, 1e0, 1e2], [L"10^{-2}", L"10^0", L"10^2"]),
		ylabel=L"Tr\left[\delta\Theta^2\right]",
        yticks=([1e0, 1e5, 1e10], [L"10^0", L"10^5", L"10^{10}"])
	)
	
    scatter!(ax1, χ, sqrt.(model.δΘ₁.^2 .+ model.δΘ₂.^2 .+ model.δΘ₃.^2),
		color=model.likelihood,
		colormap=:inferno
	)
	hlines!(ax1, [1e1], color=:turquoise, linewidth=4, linestyle=:dashdot)
    save("$FIG/nb_total_uncertainty_vs_expression.png", fig, px_per_unit=2)

    # figure 2
    fig = Figure(font="Latin Modern Math", fontsize=26)
	ax1 = Axis(fig[1,1],
		xscale=Makie.pseudolog10,
		yscale=Makie.log10,
		xlabel=L"\Theta_1",
        xticks=([-5, 0, +5], [L"-5", L"0", "+5"]),
		ylabel=L"\delta\Theta_1",
        yticks=([1e-4, 1e-3, 1e-2, 1e-1, 1e0], [L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
	)
	scatter!(ax1, model.Θ₁, model.δΘ₁, color=log.(χ), colormap=:inferno)
    save("$FIG/nb_1_uncertainty_vs_expression.png", fig, px_per_unit=2)

    # figure 3
    fig = Figure(font="Latin Modern Math", fontsize=26)
	ax1 = Axis(fig[1,1],
		xscale=Makie.pseudolog10,
		yscale=Makie.log10,
		xlabel=L"\Theta_2",
        xticks=([-1, 0, 1, 2, 3], [L"-1", L"0", L"1", L"2", L"3"]),
		ylabel=L"\delta\Theta_2",
        yticks=([1e-3, 1e-2, 1e-1, 1e0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
	)
	scatter!(ax1, model.Θ₂, model.δΘ₂, color=log.(χ), colormap=:inferno)
    save("$FIG/nb_2_uncertainty_vs_expression.png", fig, px_per_unit=2)

    # figure 4
    fig = Figure(font="Latin Modern Math", fontsize=26)
	ax1 = Axis(fig[1,1],
		xscale=Makie.pseudolog10,
		yscale=Makie.log10,
		xlabel=L"\Theta_3",
        xticks=([1e-2, 1e0, 1e2, 1e4], [L"10^{-2}", L"10^{0}", L"10^{2}", L"10^{4}"]),
		ylabel=L"\delta\Theta_3",
        yticks=([1e-5, 1e0, 1e5, 1e10], [L"10^{-5}", L"10^{0}", L"10^{+5}", L"10^{10}"]),
	)
	scatter!(ax1, model.Θ₃, model.δΘ₃, color=log.(χ), colormap=:inferno)
    save("$FIG/nb_3_uncertainty_vs_expression.png", fig, px_per_unit=2)
end

end

function main(path::String)
    # TODO: check for subdir vs directory
    isdir(DATA) || mkpath(DATA)
 
    # raw empirics
    raw, gene = GZip.open(DataIO.counts, path)

    Figures.mkpath()
    Figures.heteroskedastic(raw)
    Figures.overdispersed(raw)

    # negative binomial estimation
    model = if isfile("$DATA/nbmodel.jld2")
        load("$DATA/nbmodel.jld2", "model")
    else
        m = Normalize.glm(raw)
        save("$DATA/nbmodel.jld2", Dict("model"=>m))
        m
    end

    Figures.nb_uncertainty(raw, model)
    # Figures.nb_badfits()
    # Figures.nb_params()

    # bootstrap
end

if abspath(PROGRAM_FILE) == @__FILE__
    main("/home/nolln/mnt/data/drosophila/dvex/dge_raw.txt.gz")
end

end
