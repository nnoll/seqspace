module Drosophila

using GZip
using LinearAlgebra, NMF
using Statistics, StatsBase
using FileIO, JLD2

include("../src/util.jl")
include("../src/infer.jl")
include("../src/normalize.jl")
include("../src/pointcloud.jl")

# ------------------------------------------------------------------------
# globals

const DATA = "proc/drosophila"

# ------------------------------------------------------------------------
# data i/o

module DataIO

include("../src/scrna.jl")

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
		dvir = scRNA.searchloci(count, "Dvir_"),
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

    # remove any remaining lowly expressed genes
	count = scRNA.filtergene(count) do count, _
		sum(count) > 20
	end

	return count.data, count.gene
end

end

# ------------------------------------------------------------------------
# figures

module Figures

using Printf
using LaTeXStrings
using ColorSchemes
using Makie, CairoMakie
using Statistics, StatsBase
using SpecialFunctions: erf

using ..Normalize, ..PointCloud

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
        xticks=([10^4.0, 10^4.5, 10^5.0], [L"10^{4.0}", L"10^{4.5}", L"10^{5.0}"]),
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
		color=(:black,0.03),
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
		color=(:black,0.05),
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
        xticks=([-5, 0, +5], [L"-5", L"0", L"5"]),
		ylabel=L"\delta\Theta_L1",
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
        yticks=([1e-5, 1e0, 1e5, 1e10], [L"10^{-5}", L"10^{0}", L"10^{5}", L"10^{10}"]),
	)
	scatter!(ax1, model.Θ₃, model.δΘ₃, color=log.(χ), colormap=:inferno)
    save("$FIG/nb_3_uncertainty_vs_expression.png", fig, px_per_unit=2)
end

function nb_badfits(raw, model)
	fig = Figure(font="Latin Modern Math", fontsize=26)
	ax1 = Axis(
		fig[1,1],
		xlabel="gene expression",
		ylabel="cumulative",
        yticks=([0, 0.25, 0.5, 0.75, 1.0], [L"0.0", L"0.25", L"0.50", L"0.75", L"1.00"])
	)

    δΘ  = sqrt.(model.δΘ₁.^2 .+ model.δΘ₂.^2 .+ model.δΘ₃.^2)
    bad = findall(δΘ .> 1e1)

    for g in sample(bad, min(100, length(bad)); replace=false)
		lines!(ax1,
			sort(vec(raw[g,:])), range(0,1,size(raw,2)),
			color=(:black,.05),
		)
	end
    save("$FIG/nb_badfits.png", fig, px_per_unit=2)
end

function nb_params(raw, model)
    δΘ = sqrt.(model.δΘ₁.^2 .+ model.δΘ₂.^2 .+ model.δΘ₃.^2)
	ι  = findall(δΘ .< 1e1)

    # figure 1
    fig = Figure(font="Latin Modern Math", fontsize=26)
	ax1 = Axis(
		fig[1,1],
		xlabel=L"\Theta_2",
		ylabel="cumulative",
        yticks=([0, 0.25, 0.5, 0.75, 1.0], [L"0.0", L"0.25", L"0.50", L"0.75", L"1.00"])
	)
	lines!(ax1,
		sort(model.Θ₂[ι]), range(0,1,length(ι)),
		color=:black,
		linewidth=2,
	)
    save("$FIG/nb_param2.png", fig, px_per_unit=2)

    # figure 2
    fig = Figure(font="Latin Modern Math", fontsize=26)
	ax1 = Axis(
		fig[1,1],
		xlabel=L"\Theta_3",
        xscale=Makie.log10,
        xticks=([1e-2, 1e-1, 1e0, 1e1], [L"10^{-2}", L"10^{-1}", L"10^0", L"10^1"]),
		ylabel="cumulative",
        yticks=([0, 0.25, 0.5, 0.75, 1.0], [L"0.0", L"0.25", L"0.50", L"0.75", L"1.00"])
	)
	lines!(ax1,
		sort(model.Θ₃[ι]), range(0,1,length(ι)),
		color=:black,
		linewidth=2,
	)
    save("$FIG/nb_param3.png", fig, px_per_unit=2)
end

function bootstrap(count, model, fits, index)
	χ = [ gene |> vec |> Normalize.logmean for gene in eachrow(count) ]

	function kernel(i)
		μ = [ median(b[i]) for b in fits ]
		δ = [ var(b[i]) for b in fits ]
		return μ, δ
	end

	Θ₁, δΘ₁ = kernel(1)
	Θ₂, δΘ₂ = kernel(2)
	Θ₃, δΘ₃ = kernel(3)

    # figure 1
	fig = Figure(font="Latin Modern Math", fontsize=26)
	ax1 = Axis(
		fig[1,1],
		xlabel="MLE Θ₁",
		ylabel="bootstrap Θ₁",
	)
	scatter!(ax1, model.Θ₁[index], Θ₁, color=log.(χ[index]), colormap=:inferno)
	ax2 = Axis(
		fig[1,2],
		xlabel="MLE δΘ₁",
		xscale=Makie.log10,
        xticks=([1e-3, 1e-2, 1e-1, 1e0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
		ylabel="bootstrap δΘ₁",
		yscale=Makie.log10,
        yticks=([1e-3, 1e-2, 1e-1, 1e0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
	)
	scatter!(ax2, model.δΘ₁[index], δΘ₁, color=log.(χ[index]), colormap=:inferno)
    save("$FIG/bootstrap_1.png", fig, px_per_unit=2)

	fig = Figure(font="Latin Modern Math", fontsize=26)
	ax1 = Axis(
		fig[1,1],
		xlabel="MLE Θ₂",
		ylabel="bootstrap Θ₂",
	)
	scatter!(ax1, model.Θ₂[index], Θ₂, color=log.(χ[index]), colormap=:inferno)

	ax2 = Axis(
		fig[1,2],
		xlabel="MLE δΘ₂",
        xticks=([1e-3, 1e-2, 1e-1, 1e0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
		xscale=Makie.log10,
		ylabel="bootstrap δΘ₂",
		yscale=Makie.log10,
        yticks=([1e-3, 1e-2, 1e-1, 1e0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
	)
	scatter!(ax2, model.δΘ₂[index], δΘ₂, color=log.(χ[index]), colormap=:inferno)
    save("$FIG/bootstrap_2.png", fig, px_per_unit=2)

	fig = Figure(font="Latin Modern Math", fontsize=26)
	ax1 = Axis(
		fig[1,1],
		xlabel="MLE Θ₃",
		ylabel="bootstrap Θ₃",
		xscale=Makie.log10,
        xticks=([1e-1, 1e0, 1e1, 1e2], [L"10^{-1}", L"10^{0}", L"10^{1}", L"10^2"]),
		yscale=Makie.log10,
        yticks=([1e-2, 1e-1, 1e0, 1e1, 1e2], [L"10^{-2}", L"10^{-1}", L"10^{0}", L"10^{1}", L"10^2"]),
	)
	scatter!(ax1, model.Θ₃[index], Θ₃, color=log.(χ[index]), colormap=:inferno)
	ylims!(ax1, (1e-2, 1e2))
	
	ax2 = Axis(
		fig[1,2],
		xlabel="MLE δΘ₃",
        xticks=([1e-4, 1e-2, 1e0], [L"10^{-4}", L"10^{-2}", L"10^{0}"]),
		xscale=Makie.log10,
		ylabel="bootstrap δΘ₃",
		yscale=Makie.log10,
        yticks=([1e-4, 1e-2, 1e0, 1e2], [L"10^{-4}", L"10^{-2}", L"10^{0}", L"10^{2}"]),
	)
	scatter!(ax2, model.δΘ₃[index], δΘ₃, color=log.(χ[index]), colormap=:inferno)
	ylims!(ax2, (1e-5, 1e2))
    save("$FIG/bootstrap_3.png", fig, px_per_unit=2)
end

function estimated_rank(count, Λ)
    fig = Figure(font="Latin Modern Math", fontsize=26)
	ax1 = Axis(
		fig[1,1],
		xlabel="singular value",
		xscale=Makie.log10,
        xticks=([1e2, 10^(2.5), 1e3, 10^(3.5)], [L"10^{2.0}", L"10^{2.5}", L"10^{3.0}", L"10^{3.5}"]),
		ylabel="rank",
		yscale=Makie.log10,
        yticks=([1e0, 1e1, 1e2, 1e3], [L"10^{0}", L"10^{1}", L"10^{2}", L"10^{3}"]),
	)
	λ = sum(sqrt.(size(count)))
	k = sum(Λ.S .>= λ)
		
	lines!(ax1, Λ.S, 1:length(Λ.S), 
		color=:black,
		linewidth=2,
		label="empirical spectrum",
	)
	vlines!(ax1, [λ],
		color=:red,
		linestyle=:dashdot,
		linewidth=2,
		label="MP max value",
	)
	axislegend("Rank≈$(k)", position = :rt)
    save("$FIG/rank_estimate.png", fig, px_per_unit=2)

    fig = Figure(font="Latin Modern Math", fontsize=26)
	ax1 = Axis(
		fig[1,1],
		xlabel="principal component",
		xscale=Makie.log10,
        xticks=([1e0, 1e1, 1e2, 1e3], [L"10^0", L"10^1", L"10^2", L"10^3"]),
		ylabel="participation ratio",
		yscale=Makie.log10,
        yticks=([10^2.6, 10^2.8, 10^3.0, 10^3.2, 10^3.4], [L"10^{2.6}", L"10^{2.8}", L"10^{3.0}", L"10^{3.2}", L"10^{3.4}"]),
	)

    IPR = ( sum(Λ.U.^4, dims=1) ./ sum(Λ.U.^2,dims=1) ) |> vec

    lines!(ax1, 1 ./ IPR,
           color=:black,
           linewidth=2
    )
    save("$FIG/participation_ratio.png", fig, px_per_unit=2)
end

function gamma_fits(model, count)
	fig = Figure(font="Latin Modern Math", fontsize=26)
	ax1 = Axis(fig[1,1],
		xlabel="theoretical quantile",
        xticks=([0, 0.25, 0.5, 0.75, 1.0], [L"0.0", L"0.25", L"0.50", L"0.75", L"1.00"]),
		ylabel="empirical quantile",
        yticks=([0, 0.25, 0.5, 0.75, 1.0], [L"0.0", L"0.25", L"0.50", L"0.75", L"1.00"]),
	)

    Z = .5*(1 .+ erf.( (model.residual) ./ sqrt(2) ))
    χ = [ Normalize.logmean(gene |> vec) for gene in eachrow(count) ]

	for i in 1:10:size(model.residual,1)
        χ[i] ≥ 1 || continue

        x = sort(Z[i,:] |> vec)
		lines!(ax1, sort(x), range(0,1,length(x)), color=(:black,.05))
	end

    save("$FIG/gamma_qq.png", fig, px_per_unit=2)
end

function scaling(D, N; δ=2)
    ϕ, Rs = PointCloud.scaling(D, N)

	fig = Figure(font="Latin Modern Math", fontsize=26)
	ax1 = Axis(fig[1,1],
		xlabel="ball size",
		xscale=Makie.log10,
        xticks=([10^1, 10^1.5, 10^2.0, 10^2.5], [L"10^{1.0}", L"10^{1.5}", L"10^{2.0}", L"10^{2.5}"]),
		ylabel="points enclosed",
		yscale=Makie.log10,
        yticks=([10^1, 10^1.5, 10^2.0, 10^2.5, 10^3.0], [L"10^{1.0}", L"10^{1.5}", L"10^{2.0}", L"10^{2.5}", L"10^{3.0}"]),
	)
	for i in 1:δ:size(ϕ,1)
	    lines!(ax1, Rs, vec(ϕ[i,:]),
	           color=(:black,.01),
	           label="",
	    )
	end
    lines!(ax1, Rs, 6*size(ϕ,1)*(Rs ./ maximum(Rs)).^3,
           color=:red,
           linestyle=:dashdot,
           linewidth=2,
           label="3d scaling",
    )
	ylims!(ax1, (10, size(D,1)))

    save("$FIG/pointcloud_scaling.png", fig, px_per_unit=2)
end

function inverse(mapping, database, residual; len=100)
    cmean   = zeros(len)
    cstd    = zeros(len)
    entropy = Array{Array{Float64,1},1}(undef,len)

    βs = range(1,2000,len)
    for (i,β) in enumerate(βs)
        ψ = minimum(size(residual))*mapping.invert(β)
        ψ = ψ ./ sum(ψ,dims=1)
        G = residual*ψ'
        c = [cor(G[g,:] |> vec, database.expression.real[:,i] |> vec) for (i,g) in enumerate(mapping.index) if g !== nothing]

        cmean[i]   = mean(c)
        cstd[i]    = std(c)
        entropy[i] = -sum(ψ .* log2.(ψ), dims=1) |> vec
    end

    # figure 1
	fig = Figure(font="Latin Modern Math", fontsize=26)
	ax1 = Axis(fig[1,1],
        xlabel="β",
        ylabel="residual",
    )
    lines!(ax1, βs, cmean, color=:black, linewidth=2)
    errorbars!(ax1, βs, cmean, cstd./4, color=:red, linewidth=2)
    save("$FIG/bdtnp_fit.png", fig, px_per_unit=2)

    # figure 2
	fig = Figure(font="Latin Modern Math", fontsize=26)
	ax1 = Axis(fig[1,1],
        xlabel="entropy (bits)",
        ylabel="cumulative",
        yticks=([0, 0.25, 0.5, 0.75, 1.0], [L"0.0", L"0.25", L"0.50", L"0.75", L"1.00"])
    )

    δ = 10
    colors = cgrad(:matter,length(entropy[1:δ:end]), categorical=true)
    for (i,S) in enumerate(entropy[1:δ:end])
        lines!(ax1, sort(S), range(0,1,length(S));
               linewidth=2,
               color=colors[i],
               label=@sprintf "%.2f" βs[δ*(i-1)+1])
    end
    axislegend("β", position = :rb)
    save("$FIG/bdtnp_entropy.png", fig, px_per_unit=2)
end

function isomap(inverse, database, count, Z, D)
    # figure 1
	fig = Figure(font="Latin Modern Math", fontsize=26)
	ax1 = Axis(fig[1,1],
        xlabel="latent dimension",
        ylabel="distance correlation",
    )
    ξ = PointCloud.isomap(count,10;k=8);
    c = [
		let
			Dest = PointCloud.Distances.euclidean(ξ[:,1:i]')
			cor(PointCloud.upper_tri(Dest), PointCloud.upper_tri(D))
		end	for i in 1:size(ξ,2)
	]
    lines!(ax1, c, color=:black, linewidth=2)
    save("$FIG/isomap_dimension.png", fig, px_per_unit=2)

    ξ = PointCloud.isomap(Z,10;k=10);
    ψ = inverse.invert(150)
    X = ψ'*database.position

    # figure 2
    fig = Figure(font="Latin Modern Math", fontsize=26)
	ax1 = Axis3(fig[1,1],
		elevation=π/10,
		azimuth=5π/4,
        title="AP"
	)
	scatter!(ax1,
		ξ[:,1],ξ[:,2],ξ[:,3],
		color=X[:,1],
        colormap=:inferno,
        markersize=25
	)
	hidedecorations!(ax1)
    save("$FIG/isomap_AP.png", fig, px_per_unit=2)

    # figure 3
    fig = Figure(font="Latin Modern Math", fontsize=26)
	ax1 = Axis3(fig[1,1],
		elevation=π/10,
		azimuth=5π/4,
        title="DV"
	)
	scatter!(ax1,
		ξ[:,1],ξ[:,2],ξ[:,3],
        color=atan.(X[:,2],X[:,3]),
        colormap=:inferno,
        markersize=25
	)
	hidedecorations!(ax1)
    save("$FIG/isomap_DV.png", fig, px_per_unit=2)
end

const genelist = String["eve","odd","twi","ftz","zen"]

function gene_patterns(inverse, database, gene, count; genes=genelist)
    X = database.position
    ψ = inverse.invert(150)
    ψ = ψ ./ sum(ψ,dims=1)

    G = count*ψ'

    isdir("$FIG/gene") || Base.mkpath("$FIG/gene")

    for g in genes
        i = findfirst(gene .== g)

        fig = Figure(font="Latin Modern Math", fontsize=26)
        ax1 = Axis3(fig[1,1],
            elevation=0,
            azimuth=π/2,
        )
        scatter!(ax1,
            -X[:,1],X[:,2],X[:,3],
            color=G[i,:] |> vec,
            colormap=:inferno,
            markersize=15,
        )
        hidedecorations!(ax1)

        save("$FIG/gene/$g.png", fig, px_per_unit=2)
    end
end

end

# ------------------------------------------------------------------------
# main point of entry

function finalize(count, gene, model; cutoff=10)
    uncertainty = sqrt.(model.δΘ₁.^2 .+ model.δΘ₂.^2 .+ model.δΘ₃.^2)
	keep = findall(
		(uncertainty .<= cutoff) .&
		(vec(sum(count .== 0, dims=2)) .<= .99*size(count,2))
	)

	return count[keep,:], gene[keep], (
		likelihood = model.likelihood[keep],
		residual   = model.residual[keep, :],
		
		Θ₁ = model.Θ₁[keep],
		Θ₂ = model.Θ₂[keep],
		Θ₃ = model.Θ₃[keep],

		δΘ₁ = model.δΘ₁[keep],
		δΘ₂ = model.δΘ₂[keep],
		δΘ₃ = model.δΘ₃[keep],
	)
end

function bootstrap(count)
	χ = [ cell |> vec |> Normalize.logmean for cell in eachcol(count) ]
	ι = 1:10:size(count,1)

	data = Array{Tuple}(undef,length(ι))
	Threads.@threads for (n,i) in collect(enumerate(ι))
		data[n] = Normalize.bootstrap(vec(count[i,:]), χ)
	end
	
	return data, ι
end

function rescale(count, model)
	σ² = count.*(count.+model.Θ₃) ./ (1 .+ model.Θ₃)
	u², v², _ = Utility.sinkhorn(σ²)

	μ  = Diagonal(.√u²)*count*Diagonal(.√v²)
	return μ, .√u², .√v²
end

function main(root::String; subdir=nothing)
    isdir(DATA) || mkpath(DATA)
 
    # raw empirics
    count, gene = if subdir === nothing
        if isfile("$DATA/count_dvex.jld2")
            load("$DATA/count_dvex.jld2", "count", "gene")
        else
            c, g = open(DataIO.counts, root)
            save("$DATA/count_dvex.jld2", Dict("count"=>c, "gene"=>g))
            c, g
        end
    else
        if isfile("$DATA/count_raw.jld2")
            load("$DATA/count_raw.jld2", "count", "gene")
        else
            c, g = DataIO.counts(root, subdir)
            save("$DATA/count_raw.jld2", Dict("count"=>c, "gene"=>g))
            c, g
        end
    end

    Figures.mkpath()
    Figures.heteroskedastic(count)
    Figures.overdispersed(count)

    # negative binomial estimation
    model = if isfile("$DATA/nbmodel.jld2")
        load("$DATA/nbmodel.jld2", "model")
    else
        m = Normalize.glm(count)
        save("$DATA/nbmodel.jld2", Dict("model"=>m))
        m
    end

    Figures.nb_uncertainty(count, model)
    Figures.nb_badfits(count, model)
    Figures.nb_params(count, model)

    count, gene, model = finalize(count, gene, model)

    # bootstrap
    fits, index = if isfile("$DATA/bootstrap.jld2")
        load("$DATA/bootstrap.jld2", "fits", "index")
    else
        f, i = bootstrap(count)
        save("$DATA/bootstrap.jld2", Dict("fits"=>f, "index"=>i))
        f, i
    end
    Figures.bootstrap(count, model, fits, index)

    # rank of matrix
    norm, u, v, Λ = if isfile("$DATA/normalize.jld2")
        load("$DATA/normalized.jld2", "count", "u", "v", "Λ")
    else
        norm, u, v = rescale(count, model)
        Λ = svd(norm)
        save("$DATA/normalized.jld2", Dict("count"=>norm, "u"=>u, "v"=>v, "Λ"=>Λ))
        norm, u, v, Λ
    end

    Figures.estimated_rank(norm, Λ)

    # linear dimensional reduction
    μ, gene = if isfile("$DATA/nnmf.jld2")
        load("$DATA/nnmf.jld2", "matrix", "gene")
    else
        r = nnmf(norm,40;alg=:multmse,maxiter=200,init=:nndsvda)
        m = r.W*r.H
        save("$DATA/nnmf.jld2", Dict("matrix"=>m, "gene"=>gene))
        m, gene
    end

    # estimate differential expression
    model = if isfile("$DATA/gammamodel.jld2")
        load("$DATA/gammamodel.jld2", "model")
    else
        m = Normalize.glm(μ.+1e-8; stochastic=Normalize.gamma, ϵ=1e-8);
        save("$DATA/gammamodel.jld2", Dict("model"=>m))
        m
    end
    Figures.gamma_fits(model, count)

    # spatial inference
    Z = model.residual
    D = PointCloud.geodesics(μ,3)

    inverse, database = if isfile("$DATA/inverse.jld2")
        load("$DATA/inverse.jld2", "inverse", "database")
    else
        ν, ω = if isfile("$DATA/bdntp_params.jld2")
            # load("$DATA/bdntp_params.jld2", "nu", "omega")
            nothing, nothing
        else
            nothing, nothing
        end
        db = Inference.virtualembryo(directory="/home/nolln/mnt/data/drosophila/dvex");
        I  = Inference.inversion(count./sum(count,dims=1), gene; refdb=db, ν=ν, ω=ω);
        save("$DATA/inverse.jld2", Dict("inverse"=>I, "transform"=>I.invert(150), "database"=>db))
        I, db
    end

    Figures.scaling(D, 100)
    Figures.inverse(inverse, database, Z)
    Figures.isomap(inverse, database, μ, Z, D)
    Figures.gene_patterns(inverse, database, gene, count./sum(count,dims=1))

    ξ = PointCloud.isomap(Z,10;k=10)
    if !isfile("$DATA/isomap.jld2")
        save("$DATA/isomap.jld2", Dict(
            "latent"=>ξ,
            "distance"=>D,
        ))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main("/home/nolln/mnt/data/drosophila/raw"; subdir="rep")
end

end
