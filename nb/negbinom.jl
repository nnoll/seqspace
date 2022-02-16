### A Pluto.jl notebook ###
# v0.17.5

using Markdown
using InteractiveUtils

# ╔═╡ ae4c61cc-3149-4b10-a61f-125a6fd803a6
begin
	import Pkg;	Pkg.activate(Base.current_project())
	
	using LaTeXStrings
	using Makie, GLMakie
	using LinearAlgebra, SpecialFunctions
	using Statistics, StatsBase
	using Optim, NLSolversBase
end

# ╔═╡ e09c6169-4c0d-4534-abe9-cf791f185ffb
using NMF

# ╔═╡ f90042ca-4b48-4bd6-aa01-2a2889e3b1b2
import GZip

# ╔═╡ d0b7bd92-8605-11ec-2be3-3be13a4679ff
function ingredients(path::String)
	# this is from the Julia source code (evalfile in base/loading.jl)
	# but with the modification that it returns the module instead of the last object
	name = Symbol(basename(path))
	m = Module(name)
	Core.eval(m,
        Expr(:toplevel,
             :(eval(x) = $(Expr(:core, :eval))($name, x)),
             :(include(x) = $(Expr(:top, :include))($name, x)),
             :(include(mapexpr::Function, x) = $(Expr(:top, :include))(mapexpr, $name, x)),
             :(include($path))))
	m
end

# ╔═╡ 55dbd93d-1d78-4ea0-af78-9410a2a36630
md"""
# scRNAseq Normalization
"""

# ╔═╡ ff88e7f7-e370-41b2-b082-9b250ac0b186
Normalize = ingredients("../src/normalize.jl").Normalize

# ╔═╡ 51c001d2-e85c-4526-b380-044cc26d1e18
scRNA = ingredients("../src/scrna.jl").scRNA

# ╔═╡ b82a3bb4-45b8-4a90-8633-14b2074c9a45
begin
	
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
			
	markers  = (
		yolk = scRNA.locus(count, "sisA", "CG8129", "Corp", "CG8195", "CNT1", "ZnT77C"),
		pole = scRNA.locus(count, "pgc"),
		dvir = scRNA.searchloci(count, "Dvir_")
	)

	count = scRNA.filtercell(count) do cell, _
		sum(cell[markers.dvir]) < .10*sum(cell)
	end
			
	count = scRNA.filtergene(count) do count, gene
		!occursin("Dvir_", gene) && sum(count) > 20
	end
				
	markers  = (
		yolk = scRNA.locus(count, "sisA", "CG8129", "Corp", "CG8195", "CNT1", "ZnT77C"),
		pole = scRNA.locus(count, "pgc"),
	)
			
	count = scRNA.filtercell(count) do cell, _
		(sum(cell[markers.yolk]) < 10
	  && sum(cell[markers.pole]) < 3
	  && sum(cell) > 1e4)
	end
			
	count = scRNA.filtergene(count) do count, _
		sum(count) > 20
	end

	return count.data, count.gene
end
	
end

# ╔═╡ deb992f3-72ad-48cb-92f4-b1abd04cfdcb
raw, gene = counts("/home/nolln/mnt/data/drosophila/raw", "rep"); size(raw)
#raw, gene = GZip.open(counts,"/home/nolln/mnt/data/drosophila/dvex/dge_raw.txt.gz"); size(raw)

# ╔═╡ 8210f158-1d58-44e0-8733-586c584be4e1
md"""
## Negative binomial fit
As seen below, we have huge variability across sequencing depth and gene expression.
We fit a GLM negative-binomial model to each gene, controlling for the confounding efffects of sequencing depth.
"""

# ╔═╡ ba552cce-953f-4f62-bec1-d7c2e9358e6f
let
	fig = Figure(font="Latin Modern Math", fontsize=26)
	
	ax1 = Axis(
		fig[1,1],
		xscale=Makie.log10,
		xlabel="sequencing depth",
		ylabel="cumulative"
	)

	ax2 = Axis(
		fig[1,2],
		xscale=Makie.log10,
		xlabel="gene counts",
		ylabel="cumulative"
	)

	depth = vec(sum(raw,dims=1))
	level = vec(sum(raw,dims=2))
	
	lines!(ax1, sort(depth), range(0,1,size(raw,2)),
		linewidth=2,
		color=:black
	)
	
	lines!(ax2, sort(level), range(0,1,size(raw,1)),
		linewidth=2,
		color=:black
	)

	fig
end

# ╔═╡ 51cde4a1-28ab-413f-8627-d96f2c55be12
model = Normalize.glm(raw);

# ╔═╡ f1ee2764-c1af-4c9d-ad1e-35d629348fe7
md"""
### Filtering

We see a non-monotonic relationship between the uncertainty of our fits and the average expression for each gene. 
We filter on the total uncertainty to remove poorly estimated genes
"""

# ╔═╡ 65e87444-f8eb-4596-8af2-4c08995db91d
let
	χ = [ gene |> vec |> Normalize.logmean for gene in eachrow(raw) ]
	
	fig = Figure(font="Latin Modern Math", fontsize=26)
	ax1 = Axis(fig[1,1],
		xscale=Makie.log10,
		yscale=Makie.log10,
		xlabel=L"\chi",
		ylabel=L"Tr[\delta\Theta]",
	)
	
	scatter!(ax1, χ, model.δΘ₁ .+ model.δΘ₂ .+ model.δΘ₃, 
		color=model.likelihood,
		colormap=:inferno
	)
	hlines!(ax1, [1e1], color=:turquoise, linewidth=4, linestyle=:dashdot)
	fig
end

# ╔═╡ 593f3381-ac7a-4148-8733-99e0791ea587
md"""
#### Componentwise
For completeness, we break it down into components.
Each gene is colored by $\chi$, the scalar measure of sequencing depth
"""

# ╔═╡ 3b3c02da-d5a2-4304-acef-2c1aca63cb03
let
	χ = [ gene |> vec |> Normalize.logmean for gene in eachrow(raw) ]

	fig = Figure(font="Latin Modern Math", fontsize=26)
	ax1 = Axis(fig[1,1],
		xscale=Makie.pseudolog10,
		yscale=Makie.log10,
		xlabel=L"\Theta_1",
		ylabel=L"\delta\Theta_1",
	)
	scatter!(ax1, model.Θ₁, model.δΘ₁, color=log.(χ), colormap=:inferno)

	fig
end

# ╔═╡ b7d31f65-3b5e-4f64-8382-91ecd2d19c38
let
	χ = [ gene |> vec |> Normalize.logmean for gene in eachrow(raw) ]

	fig = Figure(font="Latin Modern Math", fontsize=26)
	ax1 = Axis(fig[1,1],
		xscale=Makie.pseudolog10,
		yscale=Makie.log10,
		xlabel=L"\Theta_2",
		ylabel=L"\delta\Theta_2",
	)
	scatter!(ax1, model.Θ₂, model.δΘ₂, color=log.(χ), colormap=:inferno)
	fig
end

# ╔═╡ fea31b0e-052d-401c-8814-68da136bdfc7
let
	χ = [ gene |> vec |> Normalize.logmean for gene in eachrow(raw) ]

	fig = Figure(font="Latin Modern Math", fontsize=26)
	ax1 = Axis(fig[1,1],
		xscale=Makie.log10,
		yscale=Makie.log10,
		xlabel=L"\Theta_3",
		ylabel=L"\delta\Theta_3",
	)
	scatter!(ax1, model.Θ₃, model.δΘ₃, color=log.(χ), colormap=:inferno)
	fig
end

# ╔═╡ 9d08d1ea-dc65-41ec-991e-e9229ce40cb4
md"""
#### Intuition
We see that the poorly fit genes are basically bernouilli trials.
Below we plot a representative sample of high uncertainty genes.
As seen, these genes have rather uninformative distributions and thus we can "throw" away the associated rows with little loss of information.
"""

# ╔═╡ def1d949-c5ef-4f5a-999c-42db1e6fa305
let
	fig = Figure(font="Latin Modern Math", fontsize=26)
	ax1 = Axis(
		fig[1,1],
		xlabel="gene expression",
		ylabel="cumulative"
	)

	for g in findall(model.δΘ₃ .> 1e5)
		lines!(ax1,
			sort(vec(raw[g,:])), range(0,1,size(raw,2)),
			color=(:black,.5),
		)
	end
	fig
end

# ╔═╡ fa4c4080-b794-4b69-bdee-b941d83e80c1
md"""
#### Scaling
We see an odd scaling with respect to gene counts and the estimated sequencing depth.
It scales like $n \sim \chi^{1.25}$ instead of proportionally.
"""

# ╔═╡ 4a53093f-702d-4c9d-85b0-e5f06bffada8
let
	fig = Figure(font="Latin Modern Math", fontsize=26)
	ax1 = Axis(
		fig[1,1],
		xlabel="Θ₂",
		ylabel="Cumulative",
	)

	ι = findall(model.δΘ₂ .< 1e-1)
	lines!(ax1,
		sort(model.Θ₂[ι]), range(0,1,length(ι)),
		color=:black,
		linewidth=2,
	)
	fig
end

# ╔═╡ f9469369-cf4e-4e56-8785-44d63e2af91e
md"""
#### Large variablility in Θ₃
Varies over 4 orders of magnitude
"""

# ╔═╡ 4610b310-f2f8-47aa-8158-ecc453c6e145
let
	fig = Figure(font="Latin Modern Math", fontsize=26)
	ax1 = Axis(
		fig[1,1],
		xlabel="Θ₃",
		xscale=Makie.log10,
		ylabel="Cumulative",
	)

	ι = findall(model.δΘ₃ .< 1e5)
	lines!(ax1,
		sort(model.Θ₃[ι]), range(0,1,length(ι)),
		color=:black,
		linewidth=2,
	)
	fig
end

# ╔═╡ 1fc1bdf0-a03b-4799-9a9c-ff2f7e95a36f
function filter_uncertain(count, gene, model; cutoff=50)
	uncertainty = model.δΘ₁ .+ model.δΘ₂ .+ model.δΘ₃
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

# ╔═╡ 77e68c01-447c-4bd9-b0a8-f6b82269c8e4
Counts, Genes, Model = filter_uncertain(raw, gene, model; cutoff=10); size(Counts)

# ╔═╡ d43e0889-6b07-48a5-bbc1-20c37089b50b
md"""
## Verification
To ensure our fits are adequate, we verify the parameter estimates via bootstrap.
"""

# ╔═╡ 5e4f0f97-80aa-4b09-8840-285428bb1faa
bootstrap, sampled_indices = let
	χ = [ cell |> vec |> Normalize.logmean for cell in eachcol(Counts) ]
	ι = 1:10:size(Counts,1)
	data = Array{Tuple}(undef,length(ι))
	
	Threads.@threads for (n,i) in collect(enumerate(ι))
		data[n] = Normalize.bootstrap(vec(Counts[i,:]), χ)
	end
	
	data,ι
end;

# ╔═╡ 5cf21f9c-0f18-4544-910a-53639859373a
let
	ι = sampled_indices
	χ = [ gene |> vec |> Normalize.logmean for gene in eachrow(Counts) ]

	function kernel(i)
		Θ̂  = [ median(b[i]) for b in bootstrap ]
		δΘ̂ = [ var(b[i])  for b in bootstrap ]
		return Θ̂, δΘ̂
	end

	Θ₁, δΘ₁ = kernel(1)
	Θ₂, δΘ₂ = kernel(2)
	Θ₃, δΘ₃ = kernel(3)

	fig = Figure(font="Latin Modern Math", fontsize=26)
	
	ax1 = Axis(
		fig[1,1],
		xlabel="MLE Θ₁",
		ylabel="bootstrap Θ₁",
	)
	scatter!(ax1, Model.Θ₁[ι], Θ₁, color=log.(χ[ι]), colormap=:inferno)

	ax2 = Axis(
		fig[1,2],
		xlabel="MLE δΘ₁",
		xscale=Makie.log10,
		ylabel="bootstrap δΘ₁",
		yscale=Makie.log10,
	)
	scatter!(ax2, Model.δΘ₁[ι], δΘ₁, color=log.(χ[ι]), colormap=:inferno)

	ax3 = Axis(
		fig[2,1],
		xlabel="MLE Θ₂",
		ylabel="bootstrap Θ₁",
	)
	scatter!(ax3, Model.Θ₂[ι], Θ₂, color=log.(χ[ι]), colormap=:inferno)

	ax4 = Axis(
		fig[2,2],
		xlabel="MLE δΘ₂",
		ylabel="bootstrap δΘ₂",
		xscale=Makie.log10,
		yscale=Makie.log10,
	)
	scatter!(ax4, Model.δΘ₂[ι], δΘ₂, color=log.(χ[ι]), colormap=:inferno)

	ax5 = Axis(
		fig[3,1],
		xlabel="MLE Θ₃",
		ylabel="bootstrap Θ₃",
		xscale=Makie.log10,
		yscale=Makie.log10,
	)
	scatter!(ax5, Model.Θ₃[ι], Θ₃, color=log.(χ[ι]), colormap=:inferno)
	ylims!(ax5, (1e-2, 1e2))
	
	ax6 = Axis(
		fig[3,2],
		xlabel="MLE δΘ₃",
		xscale=Makie.log10,
		ylabel="bootstrap δΘ₃",
		yscale=Makie.log10,
	)
	scatter!(ax6, Model.δΘ₃[ι], δΘ₃, color=log.(χ[ι]), colormap=:inferno)
	ylims!(ax6, (1e-5, 1e2))

	fig
end

# ╔═╡ ebd96e2d-1146-4f57-9fbd-8b0c7496b553
md"""
## Dimensional reduction
With the mean and variance estimated reliably, we now rescale with sinkhorn and provide a rank estimation.

### Random matrix theory
"""

# ╔═╡ c3820cbf-70e8-49e7-bfe7-032a5b39515a
Utility = ingredients("../src/util.jl").Utility

# ╔═╡ 36ba7c41-bc1f-480d-bd3c-dc64c795aea6
μ, σ², u, v = let
	σ² = Counts.*(Counts.+Model.Θ₃) ./ (1 .+ Model.Θ₃)
	u², v², _ = Utility.sinkhorn(σ²)

	μ  = Diagonal(.√u²)*Counts*Diagonal(.√v²)
	σ² = Diagonal(u²)*σ²*Diagonal(v²)
	μ, σ², .√u², .√v²
end;

# ╔═╡ cf324d59-c127-479f-8878-6bd2b17bce2e
Λ = svd(μ);

# ╔═╡ cd718366-107a-440f-a2bc-9f9a6ec7c96d
let
	fig = Figure(font="Latin Modern Math", fontsize=26)
	ax1 = Axis(
		fig[1,1],
		ylabel="rank",
		xlabel="singular value",
		xscale=Makie.log10,
		yscale=Makie.log10,
	)
	λ = sum(sqrt.(size(Counts)))
	k = sum(Λ.S .>= λ)
		
	lines!(ax1, Λ.S, 1:length(Λ.S), 
		color=:black,
		linewidth=2,
		label="spectrum",
	)
	vlines!(ax1, [λ],
		color=:red, 
		linestyle=:dashdot, 
		linewidth=2,
		label="MP max value",
	)
	
	axislegend("Rank≈$(k)", position = :rt)

	fig
end

# ╔═╡ 7dee363d-b606-4d74-b73c-80978de0ef58
μ̃ = let
	r = nnmf(μ,40;alg=:multmse,maxiter=200,init=:nndsvda)
	r.W*r.H
	#=
	m = Λ.U[:,1:40] * Diagonal(Λ.S[1:40]) * Λ.Vt[1:40,:]
	m[m .< 0] .= 0
	m
	=#
end;

# ╔═╡ 0326c120-1a3f-4ed6-aad8-dab05a1e1adf
MeanModel = Normalize.glm(μ̃.+1e-8; stochastic=Normalize.gamma, ϵ=1e-8);

# ╔═╡ 7acdbbcf-2e99-4664-943f-89787d7901d8
let
	fig = Figure(font="Latin Modern Math", fontsize=26)
	ax1 = Axis(fig[1,1],
		ylabel="empirical quantile",
		xlabel="theoretical quantile",
	)

	Z = erf.( (MeanModel.residual) ./ sqrt(2) )
	for i in 1:50:size(MeanModel.residual,1)
		χ = .5*(1 .+ Z[i,:])
		lines!(ax1, sort(χ), range(0,1,length(χ)), color=(:black,.05))
	end
	fig
end

# ╔═╡ af2f9edf-80b9-4b15-8056-b55482c10500
Z = let
#=
	F = svd(Model.residual)
	F.U[:,1:30]*Diagonal(F.S[1:30])*F.Vt[1:30,:]
=#
	#.5*(1 .+ erf.(MeanModel.residual ./ sqrt(2)))
	MeanModel.residual
end;

# ╔═╡ 79b02fc1-fb56-4814-ac19-882decc2f137
μ̂ = let
	μ = Λ.U[:,1:40]*Diagonal(Λ.S[1:40])*Λ.Vt[1:40,:]

# sinkhorn normalization
	u, v, _ = Utility.sinkhorn(μ)
	Diagonal(u)*μ*Diagonal(v)

#= unscaled pearson normalization 
	χ = log.([ cell |> vec |> Normalize.logmean for cell in eachcol(n) ])
	norms = [
		let
			n = (1 ./ u[i]) .* gene .* (1 ./ v)
			m = exp.( Model.Θ₁[i] .+ Model.Θ₂[i].*χ )
			s = m.*(1 .+ m./Model.Θ₃[i])
				
			v = (n .- m) ./ sqrt.(s)
			v[v.<-5]  .= -5
			v[v.>+5]  .= +5
			v
		end for (i,gene) in enumerate(eachrow(μ))
	]
	Matrix(
		reduce(hcat,norms)'
	)
=#	
end;

# ╔═╡ 3f2d6b35-7cfc-4c2a-a15e-8ce69afe0bfb
let
	fig = Figure()
	ax1 = Axis(fig[1,1],
		xscale=Makie.log10,
		yscale=Makie.log10,
	)
	ax2 = Axis(fig[1,2],
		xscale=Makie.log10,
		yscale=Makie.log10,
	)

	scatter!(ax1, sum(μ,dims=1) |> vec, -Λ.Vt[1,:]|> vec)
	scatter!(ax2, sum(μ,dims=2) |> vec, -Λ.U[:,1]|> vec)

	fig
end

# ╔═╡ 90c0c19f-99bd-4d42-95a9-3c4b872cfcf0
md"""
### Pearson residuals
Repeat the same analysis, but don't use random matrix theory first.
Use our estimated pearson residuals.
"""

# ╔═╡ dff854f7-97e7-402c-9476-1717159ab518
let
	fig = Figure(font="Latin Modern Math", fontsize=26)
	ax1 = Axis(
		fig[1,1],
		ylabel="rank",
		xlabel="singular value",
		xscale=Makie.log10,
		yscale=Makie.log10,
	)
	F = svd(Model.residual)
	Λ = sum(sqrt.(size(N)))
	k = sum(F.S .>= Λ)
		
	lines!(ax1, F.S, 1:length(F.S), 
		color=:black, 
		linewidth=2,
		label="spectrum",
	)
	vlines!(ax1, [Λ],
		color=:red, 
		linestyle=:dashdot, 
		linewidth=2,
		label="MP max value",
	)
	
	axislegend("Rank≈$(k)", position = :lb)

	fig
end

# ╔═╡ 0dec4b8a-5402-43b6-ab51-4d911c8fd4de
md"""
## Spatial inference
"""

# ╔═╡ 8a21602f-24fe-40cd-820f-271bbbe40935
Inference = ingredients("../src/infer.jl").Inference

# ╔═╡ 6bf4f2a5-04f2-43cc-995a-78f1c7d1abb3
database = Inference.virtualembryo(directory="/home/nolln/mnt/data/drosophila/dvex");

# ╔═╡ 98506181-1e09-432e-9e47-68eeda650507
inv_z = Inference.inversion(Z, Genes; refdb=database);

# ╔═╡ 5b74e210-9b3d-422c-8e89-525ccdd17c47
ψz = minimum(size(Counts))*inv_z.invert(80);

# ╔═╡ d2503eec-c26a-41a8-835c-2ddb7bafdec3
size(ψz)

# ╔═╡ 41e74b6b-8835-4013-b4eb-9cdc9318f8f0
S = let
	p = ψz ./ sum(ψz,dims=2)
	-sum( p .* log.(p), dims=2)
end

# ╔═╡ 7425feb4-5c17-4ece-a6d3-80d412481b56
Xz = ψz'*database.position;

# ╔═╡ 97affcd3-e919-48d9-a263-4fa4f81dc0a2
let
	fig = Figure(font="Latin Modern Math", fontsize=26)
	ax3 = Axis3(fig[1,1],
		aspect=:data,
		elevation=0,
		azimuth=π/2,
	)
	
	scatter!(ax3,
		-database.position[:,1],database.position[:,2],database.position[:,3],
		color=ψz*vec(Z[findfirst(Genes.=="odd"),:]),
		colormap=:inferno,
		markersize=2000,
	)
	hidedecorations!(ax3)
	fig
end

# ╔═╡ a803fc43-75de-44eb-9d23-6661fe44a01d
seq = let
	u,v = Utility.sinkhorn(μ)
	Diagonal(u)*μ*Diagonal(v)
end

# ╔═╡ ef87beae-4f24-46d1-87db-72b8e4c2e5aa
inv_m = Inference.inversion(seq, Genes; refdb=database);

# ╔═╡ 786ef949-248f-4079-a226-e2c6c827c0b4
ψm = minimum(size(Counts))*inv_m.invert(80);

# ╔═╡ 9e856728-34d2-4583-9299-828178e97d33
Xm = ψm'*database.position;

# ╔═╡ c82ad97b-2230-4431-8b89-8d51211cc8b0
let
	fig = Figure(font="Latin Modern Math", fontsize=26)
	ax3 = Axis3(fig[1,1],
		aspect=:data,
		elevation=0,
		azimuth=π/2,
	)
	
	scatter!(ax3,
		-database.position[:,1],database.position[:,2],database.position[:,3],
		color=ψm*vec(μ̂[findfirst(Genes.=="eve"),:]),
		colormap=:inferno,
		markersize=2000,
	)
	hidedecorations!(ax3)
	fig
end

# ╔═╡ 10af84b3-e81c-4e2f-b518-950d1b4aac81
md"""
## Manifold learning
### Isomap
"""

# ╔═╡ 6ef6791f-65be-47ca-8330-7f5f16e8f222
PointCloud = ingredients("../src/pointcloud.jl").PointCloud

# ╔═╡ d93b478b-a094-4edd-8aea-b84cd91cfa82
function scaling(D, N; δ=2)
    ϕ, Rs = PointCloud.scaling(D, N)

	fig = Figure(font="Latin Modern Math", fontsize=26)
	ax1 = Axis(fig[1,1],
		xlabel="ball size",
		ylabel="points enclosed",
		xscale=Makie.log10,
		yscale=Makie.log10,
	)
	for i in 1:δ:size(ϕ,1)
	    lines!(ax1, Rs, vec(ϕ[i,:]),
	           color=(:black,.01),
	           label="",
	    )
	end
    lines!(ax1, Rs, 10*size(ϕ,1)*(Rs ./ maximum(Rs)).^3,
           color=:red,
           linestyle=:dashdot,
           linewidth=2,
           label="3d scaling",
    )
	ylims!(ax1, (10, size(D,1)))
    return fig
end

# ╔═╡ 29cc3bbf-5da3-4a0e-9bcc-dd5bc0d3b53e
Dz = PointCloud.geodesics(Z,6);

# ╔═╡ ea0af752-58d4-4399-8d34-7eff34967a03
scaling(Dz,100)

# ╔═╡ c7a2a425-be2b-4a0d-969c-a8642d30ab58
ξz = PointCloud.isomap(Z, 10); size(ξz)

# ╔═╡ e62d07c8-37c0-440c-b520-4a57fdbcf5ef
let
	fig = Figure()
	ax1 = Axis3(fig[1,1])
	ax2 = Axis3(fig[1,2])
	
	scatter!(ax1, -ξz[:,1], ξz[:,2], ξz[:,3], markersize=4000,color=Xm[:,1], colormap=:inferno)
	scatter!(ax2, -ξz[:,1], ξz[:,2], ξz[:,3], markersize=4000,color=atan.(Xm[:,2],Xm[:,3]), colormap=:inferno)
	fig
end

# ╔═╡ 7cec641c-2311-4e2b-888c-03c7af82e07d
begin
	AP = Xz[:,1] .+ .5*ξz[:,3]
	DV = atan.(Xz[:,2],Xz[:,3]) .+ .013*ξz[:,1]
end;

# ╔═╡ 8cd10af6-db26-4846-b65d-66c0fc47a53e
let
	fig = Figure()
	ax1 = Axis(fig[1,1])
	ax2 = Axis(fig[1,2])
	
	scatter!(ax1, ξz[:,1], ξz[:,3], color=AP, colormap=:inferno)
	scatter!(ax2, ξz[:,1], ξz[:,3], color=DV, colormap=:inferno)
	fig
end

# ╔═╡ 474c8b35-4eef-4995-88f7-7b0355560a14
let
	c = [
		let
			Dest = PointCloud.Distances.euclidean(ξz[:,1:i]')
			cor(PointCloud.upper_tri(Dest), PointCloud.upper_tri(Dz))
		end	for i in 1:size(ξz,2)
	]

	lines(c)
end

# ╔═╡ 35cdebf3-5400-47cb-8b3d-759451adcef9
Dm = PointCloud.geodesics(μ̂,12);

# ╔═╡ 03ef65ec-6155-4705-8ab8-d6515610a535
scaling(Dm,100)

# ╔═╡ 975ad2b4-9a79-4b6f-ac4e-ff78b4c5fe1e
ξm = PointCloud.isomap(μ̂, 10); 

# ╔═╡ 5dac3ca1-f173-4885-ab39-748da6989e41
let
	fig = Figure()
	ax1 = Axis3(fig[1,1])
	ax2 = Axis3(fig[1,2])

	scatter!(ax1, ξm[:,1], -ξm[:,2], ξm[:,3], markersize=4000, color=Xm[:,1], colormap=:inferno)
	scatter!(ax2, ξm[:,1], -ξm[:,2], ξm[:,3], markersize=4000, color=atan.(Xm[:,2],Xm[:,3]), colormap=:inferno)
	fig
end

# ╔═╡ 06c01255-ecb7-4623-ab5b-8072f0d16553
let
	c = [
		let
			Dest = PointCloud.Distances.euclidean(ξm[:,1:i]')
			cor(PointCloud.upper_tri(Dest), PointCloud.upper_tri(Dm))
		end	for i in 1:size(ξm,2)
	]

	lines(c)
end

# ╔═╡ Cell order:
# ╠═ae4c61cc-3149-4b10-a61f-125a6fd803a6
# ╠═f90042ca-4b48-4bd6-aa01-2a2889e3b1b2
# ╟─d0b7bd92-8605-11ec-2be3-3be13a4679ff
# ╟─55dbd93d-1d78-4ea0-af78-9410a2a36630
# ╠═ff88e7f7-e370-41b2-b082-9b250ac0b186
# ╟─51c001d2-e85c-4526-b380-044cc26d1e18
# ╠═b82a3bb4-45b8-4a90-8633-14b2074c9a45
# ╠═deb992f3-72ad-48cb-92f4-b1abd04cfdcb
# ╟─8210f158-1d58-44e0-8733-586c584be4e1
# ╠═ba552cce-953f-4f62-bec1-d7c2e9358e6f
# ╠═51cde4a1-28ab-413f-8627-d96f2c55be12
# ╟─f1ee2764-c1af-4c9d-ad1e-35d629348fe7
# ╠═65e87444-f8eb-4596-8af2-4c08995db91d
# ╟─593f3381-ac7a-4148-8733-99e0791ea587
# ╠═3b3c02da-d5a2-4304-acef-2c1aca63cb03
# ╠═b7d31f65-3b5e-4f64-8382-91ecd2d19c38
# ╟─fea31b0e-052d-401c-8814-68da136bdfc7
# ╟─9d08d1ea-dc65-41ec-991e-e9229ce40cb4
# ╟─def1d949-c5ef-4f5a-999c-42db1e6fa305
# ╟─fa4c4080-b794-4b69-bdee-b941d83e80c1
# ╟─4a53093f-702d-4c9d-85b0-e5f06bffada8
# ╟─f9469369-cf4e-4e56-8785-44d63e2af91e
# ╟─4610b310-f2f8-47aa-8158-ecc453c6e145
# ╟─1fc1bdf0-a03b-4799-9a9c-ff2f7e95a36f
# ╠═77e68c01-447c-4bd9-b0a8-f6b82269c8e4
# ╟─d43e0889-6b07-48a5-bbc1-20c37089b50b
# ╠═5e4f0f97-80aa-4b09-8840-285428bb1faa
# ╟─5cf21f9c-0f18-4544-910a-53639859373a
# ╟─ebd96e2d-1146-4f57-9fbd-8b0c7496b553
# ╠═e09c6169-4c0d-4534-abe9-cf791f185ffb
# ╠═c3820cbf-70e8-49e7-bfe7-032a5b39515a
# ╠═36ba7c41-bc1f-480d-bd3c-dc64c795aea6
# ╠═cf324d59-c127-479f-8878-6bd2b17bce2e
# ╟─cd718366-107a-440f-a2bc-9f9a6ec7c96d
# ╠═7dee363d-b606-4d74-b73c-80978de0ef58
# ╠═0326c120-1a3f-4ed6-aad8-dab05a1e1adf
# ╠═7acdbbcf-2e99-4664-943f-89787d7901d8
# ╠═af2f9edf-80b9-4b15-8056-b55482c10500
# ╠═79b02fc1-fb56-4814-ac19-882decc2f137
# ╠═3f2d6b35-7cfc-4c2a-a15e-8ce69afe0bfb
# ╟─90c0c19f-99bd-4d42-95a9-3c4b872cfcf0
# ╟─dff854f7-97e7-402c-9476-1717159ab518
# ╟─0dec4b8a-5402-43b6-ab51-4d911c8fd4de
# ╠═8a21602f-24fe-40cd-820f-271bbbe40935
# ╠═6bf4f2a5-04f2-43cc-995a-78f1c7d1abb3
# ╠═98506181-1e09-432e-9e47-68eeda650507
# ╠═5b74e210-9b3d-422c-8e89-525ccdd17c47
# ╠═d2503eec-c26a-41a8-835c-2ddb7bafdec3
# ╠═41e74b6b-8835-4013-b4eb-9cdc9318f8f0
# ╠═7425feb4-5c17-4ece-a6d3-80d412481b56
# ╟─97affcd3-e919-48d9-a263-4fa4f81dc0a2
# ╠═a803fc43-75de-44eb-9d23-6661fe44a01d
# ╠═ef87beae-4f24-46d1-87db-72b8e4c2e5aa
# ╠═786ef949-248f-4079-a226-e2c6c827c0b4
# ╠═9e856728-34d2-4583-9299-828178e97d33
# ╟─c82ad97b-2230-4431-8b89-8d51211cc8b0
# ╟─10af84b3-e81c-4e2f-b518-950d1b4aac81
# ╟─6ef6791f-65be-47ca-8330-7f5f16e8f222
# ╟─d93b478b-a094-4edd-8aea-b84cd91cfa82
# ╠═29cc3bbf-5da3-4a0e-9bcc-dd5bc0d3b53e
# ╠═ea0af752-58d4-4399-8d34-7eff34967a03
# ╠═c7a2a425-be2b-4a0d-969c-a8642d30ab58
# ╠═e62d07c8-37c0-440c-b520-4a57fdbcf5ef
# ╟─7cec641c-2311-4e2b-888c-03c7af82e07d
# ╠═8cd10af6-db26-4846-b65d-66c0fc47a53e
# ╠═474c8b35-4eef-4995-88f7-7b0355560a14
# ╟─35cdebf3-5400-47cb-8b3d-759451adcef9
# ╠═03ef65ec-6155-4705-8ab8-d6515610a535
# ╠═975ad2b4-9a79-4b6f-ac4e-ff78b4c5fe1e
# ╠═5dac3ca1-f173-4885-ab39-748da6989e41
# ╠═06c01255-ecb7-4623-ab5b-8072f0d16553
