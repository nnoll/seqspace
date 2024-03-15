module DifferentialGeometry

using Match
using LinearAlgebra, Polynomials, Statistics

include("io.jl")
using .DataIO: read_obj, read_ply

export Mesh
export mesh

export Surface
export basis, pullback

export Manifold
export manifold, makefield

numdims(x) = sum(1 for d in 1:ndims(x) if size(x,d) > 1)

# ------------------------------------------------------------------------
# mesh type

"""
    struct Mesh{T <: Real}
        r   :: Array{T, 2}
        vₙ  :: Array{T, 2}
        tri :: Array{Int, 2}
    end

Store a 2 dimensional triangular mesh, embedded into arbitrary dimensions.
`r` and `vₙ` denote vertex positions and normals respectively.
`tri` denotes (non-oriented) triangular faces.
"""
struct Mesh{T <: Real}
    r   :: Array{T, 2}     # vertex positions
    vₙ  :: Array{T, 2}     # vertex normals
    tri :: Array{Int, 2}   # triangular faces
end

"""
    mesh(io::IO, type::Symbol=:obj)

Load a `Mesh` object from stream `io` formatted with `type`.
As of now, only `.obj` files are supported.
"""
function mesh(io::IO, type::Symbol=:obj)
    if type != :obj
        error("not implemented")
    end

    data = read_obj(io)

    return Mesh{eltype(data.verts[1])}(data.verts, data.norms, data.faces)
end

# ------------------------------------------------------------------------
# surface type

# internal functions
"""
    ellipse(r)

Fit an ellipse to 2D point cloud `r`.
`r` is assumed to be sized ``N \\times 2``.
"""
function ellipse(r)
    # constrained least squares (constraint: 4AC-B² = 1)
    D = hcat(r[:,1].*r[:,1], r[:,1].*r[:,2], r[:,2].*r[:,2], r[:,1], r[:,2], ones(size(r, 1),1))
    S = D'*D

    C      = zeros(6,6)
    C[1,3] = C[3,1] = 2
    C[2,2] = -1

    λ, ν = eigen(inv(S) * C)
    n    = argmax(λ)
    p    = real(ν[:,n])

    # equations taken from wikipedia
    A, B, C, D, E, F = p[1], p[2]/2, p[3], p[4]/2, p[5]/2, p[6]
    num = B^2 - A*C

    x₀ = (C*D-B*E)/num
    y₀ = (A*E-B*D)/num
    ϕ  = 0.5*atan(2*B/(A-C))

    up    = 2*(A*E^2+C*D^2+F*B^2-2*B*D*E-A*C*F)
    down1 = (B^2-A*C)*( (C-A)*√(1+4*B^2/((A-C)*(A-C)))-(C+A))
    down2 = (B^2-A*C)*( (A-C)*√(1+4*B^2/((A-C)*(A-C)))-(C+A))
    a, b  = √(abs(up/down1)), √(abs(up/down2))
    
    return [x₀, y₀, a, b, ϕ]
end

"""
    struct Surface{T <: Real}
        Θ  :: Array{T, 2}
        x  :: Array{T}
        L  :: T
        Λ  :: Array{Polynomial{T}}
        ∂Λ :: Array{Polynomial{T}}
    end

Store a representation of a 2D surface embedded into higher dimensional Euclidean space.
Fits the surface by:
  1. Partition the x axis so that each resultant interval has, on average, 20 points.
  2. Fit points within each partition to a 2D ellipse.
  3. Fit a polynomial function to each elliptical parameter over all partitions.
  4. Use polynomials to estimate tangent vectors.
`Θ` denotes the elliptical parameters fit per partition.
`x` denotes the input data.
`L` denotes the length along the x axis.
`Λ` denotes the polynomials for each elliptical parameter.
`∂Λ` denotes the derivative of polynomials for each elliptical parameter.
"""
struct Surface{T <: Real}
    Θ  :: Array{T, 2}
    x  :: Array{T}
    L  :: T
    Λ  :: Array{Polynomial{T}}
    ∂Λ :: Array{Polynomial{T}}

    # constructors
    Surface(Θ, x, L, Λ, ∂Λ) = new{eltype(x)}(Θ, x, L, Λ, ∂Λ)
 
    function Surface(x; order=12, debug=false)
        nbin = size(x,1)÷20
        xbin = LinRange(minimum(x[:,1]), maximum(x[:,1]), nbin+1)

        select(x, index) = hcat([x[index,d] for d in 2:size(x,2)]...)

        T  = eltype(x)
        x₀ = T[]
        Θ  = Array{T}[]
        for (lb, ub) in zip(xbin[1:end-1],xbin[2:end])
            ι = (x[:,1] .>= lb) .& (x[:,1] .<= ub)
            if sum(ι) < 6
                continue
            end

            push!(x₀, mean(x[ι,1]))
            push!(Θ,  ellipse(select(x, ι)))
        end

        Θ = hcat(Θ...)' # [x₀, y₀, a, b, ϕ]
        Λ = Array{Polynomial}(undef, size(Θ,2))
        L = (maximum(x₀) - minimum(x₀))./2
        for p in 1:size(Θ,2)
            M    = hcat([(x₀./L).^n for n in 0:order]...)
            α    = M \ Θ[:,p]
            Λ[p] = Polynomial(α)
            if debug
                χ = range(minimum(x₀), maximum(x₀), length=1000)
                Plots.scatter(x₀, Θ[:,p])
                Plots.plot!(χ, Λ[p].(χ./L), ylims=(minimum(Θ[:,p]),maximum(Θ[:,p])))
                gui()
                readline()
            end
        end

        ∂Λ = [derivative(Λ[i]) for i in 1:length(Λ)]

        new{T}(Θ, x₀, L, Λ, ∂Λ)
    end
end

"""
    function pullback(s::Surface, r)

Transform Euclidean point cloud `r` into a 2D cylindrical projection defined by `s`.
Cylindrical coordinates are [x,φ].
Assumes all points within `r` are distributed over the surface.
Assumes `r` is sized ``N \\times 3``.
"""
function pullback(s::Surface, r)
    z₀ = r[:,1] ./ s.L
    x₀ = s.Λ[1].(z₀)
    y₀ = s.Λ[2].(z₀)
    a  = s.Λ[3].(z₀)
    b  = s.Λ[4].(z₀)
    ϕ  = s.Λ[5].(z₀)

    v = Array{eltype(r),2}(undef, length(a), 2)
    for (i, p) in enumerate(ϕ)
        v[i,:] = [cos(p) +sin(p); -sin(p) cos(p)]*(r[i,2:3] .- [x₀[i], y₀[i]])
    end
    θ = mod2pi.(atan.(v[:,2]./b, v[:,1]./a) .+ π .- 1.65) .- π

    v[:,1] = z₀
    v[:,2] = θ

    return v
end

"""
    function basis(s::Surface, r)

Compute the tangent vectors ``\\hat{\\bm{e}}_\\phi, \\hat{\\bm{e}}_x`` associated to each point within `r`.
Assumes all points within `r` are distributed over the surface.
Assumes `r` is sized ``N \\times 3``.
"""
function basis(s::Surface, r; surface=false, normalized=true)
    x  = surface ? r : pullback(s, r)

    # function
    x₀ = s.Λ[1].(x[:,1])
    y₀ = s.Λ[2].(x[:,1])
    a  = s.Λ[3].(x[:,1])
    b  = s.Λ[4].(x[:,1])
    ϕ  = s.Λ[5].(x[:,1])

    # derivatives
    ∂x₀ = s.∂Λ[1].(x[:,1])./s.L
    ∂y₀ = s.∂Λ[2].(x[:,1])./s.L
    ∂a  = s.∂Λ[3].(x[:,1])./s.L
    ∂b  = s.∂Λ[4].(x[:,1])./s.L
    ∂ϕ  = s.∂Λ[5].(x[:,1])./s.L

    eθ = hcat(zeros(length(a)),
              -a.*cos.(ϕ).*sin.(x[:,2]) .- b.*sin.(ϕ).*cos.(x[:,2]) .+ x₀,
              -a.*sin.(ϕ).*sin.(x[:,2]) .+ b.*cos.(ϕ).*cos.(x[:,2]) .+ y₀)

    ez = hcat(ones(length(a)),
              +(∂a.*cos.(ϕ).-a.*sin.(ϕ).*∂ϕ).*cos.(x[:,2]) .- (∂b.*sin.(ϕ).+b.*cos.(ϕ).*∂ϕ).*sin.(x[:,2]) .+ ∂x₀,
              +(∂a.*sin.(ϕ).+a.*cos.(ϕ).*∂ϕ).*cos.(x[:,2]) .+ (∂b.*cos.(ϕ).-b.*sin.(ϕ).*∂ϕ).*sin.(x[:,2]) .+ ∂y₀)

    if normalized
        ez = ez ./ sqrt.(ez⋅ez)
        eθ = eθ ./ sqrt.(eθ⋅eθ)
    end

    return ez, eθ
end

# ------------------------------------------------------------------------
# manifold type

"""
    struct Manifold{T <: Real}
        mesh :: Mesh{T}
        surf :: Surface{T}
    end

Store the representation of a differential geometry object.
`mesh` is the empirical point cloud.
`surf` is the estimated differentiable surface.
"""
struct Manifold{T <: Real}
    mesh :: Mesh{T}
    surf :: Surface{T}
end


"""
    function pullback(ℳ::Manifold)

Returns the 2D cylindrical projection of the mesh, as estimated by the internal surface.
"""
function pullback(ℳ::Manifold)
    x    = pullback(ℳ.surf, ℳ.mesh.r)
    maxₗ = maximum(hcat(abs.(x[ℳ.mesh.tri[:,1],2]-x[ℳ.mesh.tri[:,2],2]), 
                        abs.(x[ℳ.mesh.tri[:,2],2]-x[ℳ.mesh.tri[:,3],2]),
                        abs.(x[ℳ.mesh.tri[:,3],2]-x[ℳ.mesh.tri[:,1],2])),
                   dims=2)
    return x, ℳ.mesh.tri[getindex.(findall(maxₗ .> π/2),1), :]
end

pullback(ℳ::Manifold, r) = pullback(ℳ.surf, r)

"""
    function order(tri, r)

Reorder the triangulation so that all labels are counterclockwise.
"""
function order(tri, r)
    r21 = r[tri[:,2],:] .- r[tri[:,1],:]
    r32 = r[tri[:,3],:] .- r[tri[:,2],:]

    det = r21[:,1].*r32[:,2] - r21[:,2].*r32[:,1] 
    idx = findall(det .< 0) 

    tmp = tri[idx, 2]
    tri[idx, 2] .= tri[idx,3]
    tri[idx, 3] .= tmp

    return tri
end

"""
    function triangulation(r₀, rᵢ)

Return the triangle containing each point given in `rᵢ`.
The triangulation used is delaunay triangulation defined by point cloud `r₀`.
"""
function triangulation(r₀, rᵢ)
    tri = order(delaunay(Array{eltype(r₀),2}(r₀'))', r₀)

    # compute containing faces
    t1  = r₀[tri[:,2],:] - r₀[tri[:,1],:]
    t2  = r₀[tri[:,3],:] - r₀[tri[:,2],:]
    t3  = r₀[tri[:,1],:] - r₀[tri[:,3],:]

    t1 /= sqrt.(t1⋅t1)
    t2 /= sqrt.(t2⋅t2)
    t3 /= sqrt.(t3⋅t3)

    t1  = t1*[0 +1; -1 0]
    t2  = t2*[0 +1; -1 0]
    t3  = t3*[0 +1; -1 0]

    e1  = .5*(r₀[tri[:,2],:] + r₀[tri[:,1],:])
    e2  = .5*(r₀[tri[:,3],:] + r₀[tri[:,2],:])
    e3  = .5*(r₀[tri[:,1],:] + r₀[tri[:,3],:])

    θ1  = t1*rᵢ' .- sum(t1.*e1,dims=2)
    θ2  = t2*rᵢ' .- sum(t2.*e2,dims=2)
    θ3  = t3*rᵢ' .- sum(t3.*e3,dims=2)

    τ(i)  = findfirst((θ1[:,i] .> 0) .& (θ2[:,i] .> 0) .& (θ3[:,i] .> 0))
    faces = [τ(i) for i in 1:size(rᵢ,1)]
    return tri, faces
end

function continuum(tri, ϕ, r)
    function fit(ϕ, r)
        M = hcat(r[:,1], r[:,2], ones(size(r,1)))
        ρ = M \ ϕ

        f(x) = ρ[1]*x[1] + ρ[2]*x[2] + ρ[3]
        return f
    end

    return [fit(ϕ[tri[t,:]], r[tri[t,:],:]) for t in 1:size(tri,1)]
end

"""
    function interpolate(M, ϕ, r)

Interpolate the tensor field `ϕ`, defined at points `r` onto the vertices of manifold `M`.
Interpolation is computed by finding the containing triangle within the mesh of `M` for each point `r`.
Linear interpolation is performed per face.
"""
function interpolate(M, ϕ, r)
    x, _ = pullback(M)
    y    = pullback(M.surf, r)

    tri, face = triangulation(y, x)
    dist = sum( (x[:,d] .- y[:,d]').^2 for d in 1:size(x,2) )

    @match numdims(ϕ) begin
        1 => begin
            field  = continuum(tri, ϕ, y)
            result = Array{eltype(ϕ)}(undef, size(x,1))
            for i in 1:size(x,1)
                if isnothing(face[i])
                    result[i] = ϕ[argmin(dist[i,:])]
                else
                    result[i] = field[face[i]](x[i,:])
                end
            end

            return result
        end
        2 => begin
            ψ = Array{eltype(ϕ)}(undef, size(x,1), 3)
            for d in 1:3
                field  = continuum(tri, ϕ[:,d], y)
                for i in 1:size(x,1)
                    if isnothing(face[i])
                        ψ[i,d] = ϕ[argmin(dist[i,:]), d]
                    else
                        ψ[i,d] = [field[face[i]](x[i,:]) for i in 1:size(x,1)]
                    end
                end
            end
            return ψ
        end
        _ => return error("unsupported field dimension = $(size(ϕ))")
    end
end

"""
    rescale(x)

Rescale array `x` to run between ``[0,1]``.
"""
function rescale(x)
    min = minimum(x)
    max = maximum(x)
    return (x.-min)./(max.-min)
end

"""
    scalar(M::Manifold, ϕ, field::Symbol)

Return a scalar field `ϕ` interpolated onto either the embedded space of manifold `M` or the pullback.
If `field` is :ℝ² the pullback is computed.
If `field` is :ℝ³ the embedding space is returned.
"""
function scalar(M::Manifold, ϕ, field::Symbol)
    @match field begin
        :ℝ² => begin
            q, tri = pullback(ℳ)
            return (
                x1  = q[:,1],
                x2  = q[:,2],
                tri = tri',
                ϕ   = ϕ,
            )
        end
        :ℝ³ => begin
            return (
                x1  = M.mesh.r[:,1],
                x2  = M.mesh.r[:,2],
                x3  = M.mesh.r[:,3],
                tri = M.mesh.tri',
                ϕ   = ϕ,
            )
        end
        _   => return error("unsupported field $(field)")
    end
end

"""
    scalar(M::Manifold, ϕ, field::Symbol)

Return a vector field field `ϕ` interpolated onto either the embedded space of manifold `M` or the pullback.
If `field` is :ℝ² the pullback is computed. Only the tangent space component is kept.
If `field` is :ℝ³ the embedding space is returned.
"""
function vector(ℳ::Manifold, ϕ, field::Symbol)
    @match field begin
        :ℝ² => begin
                q, tri = pullback(ℳ)
                ez, eθ = basis(ℳ.surf, ℳ.mesh.r)
                ϕz, ϕθ = sum(ϕ.*ez,dims=2), sum(ϕ.*eθ,dims=2)

                return (
                    x1  = q[:,1],
                    x2  = q[:,2],
                    tri = tri',
                    v1  = ϕz,
                    v2  = ϕθ,
                )
            end
        :ℝ³ => begin
                return (
                    x1  = M.mesh.r[:,1],
                    x2  = M.mesh.r[:,2],
                    x3  = M.mesh.r[:,3],
                    tri = M.mesh.tri',
                    v1  = ϕ[:,1],
                    v2  = ϕ[:,2],
                    v3  = ϕ[:,3],
                )
            end
        _   => return error("unsupported field $field")
    end
end

"""
    makefield(ℳ::Manifold, ϕ::AbstractArray; field::Symbol = :ℝ³)

Return an arbitrary tensor field defined over _either_ the embedding space or the cylindrical pullback of manifold `M`.
Higher level function than either [`scalar`](@ref) or [`vector`](@ref).
"""
function makefield(ℳ::Manifold, ϕ::AbstractArray; field::Symbol = :ℝ³)
    @match numdims(ϕ) begin
        1                         => return scalar(ℳ, ϕ, field)
        2, if size(ϕ,2) == 3 end  => return vector(ℳ, ϕ, field)
        _                         => return error("unsupported field dimension = $(size(ϕ))")
    end
end

function manifold(io::IO, type::Symbol=:obj)
    m = mesh(io, type)
    s = Surface(m.r)

    return Manifold(m, s)
end

end
