module Generate

using PyCall

# const map   = pyimport("mpl_toolkits.basemap")
# const Globe = map.Basemap()

"""
    sphere(N; R=1)

Generate a spherical point cloud of `N` points with extent of radius `R`.
"""
function sphere(N; R=1)
    θ = π  .* rand(Float64, N)
    ϕ = 2π .* rand(Float64, N)

    return R.*hcat(sin.(θ).*cos.(ϕ), sin.(θ).*sin.(ϕ), cos.(θ))', hcat(θ, ϕ)'
end

function spherical_distance(x; R=1)
    n̂ = x / R
    D = zeros(size(x,2),size(x,2))

    norm² = sum(n̂[i,:].^2 for i ∈ 1:size(x,1))
    cosΔψ = sum(n̂[i,:]' .* n̂[i,:] for i ∈ 1:size(x,1))
    sinΔψ = [norm(cross(n̂[:,i], n̂[:,i])) for i ∈ 1:size(x,2), j ∈ 1:size(x,2)]

    return R*atan.(sinΔψ, cosΔψ)
end


"""
    globe(N)

Generate a spherical point cloud of `N` points where all points are guaranteed to be sampled from land-masses from Earth.
Uses mpl.basemap Python library.
"""
function globe(N)
    Θ = Array{Float64}(undef, N)
    Φ = Array{Float64}(undef, N)
    for n in 1:N
    @label sample
        θ = π*rand(1)[1]
        ϕ = 2π*rand(1)[1]

        x, y = Globe(180*(ϕ.-π)/π, 180*(π/2 .- θ)/π)

        !Globe.is_land(x,y) && @goto sample

        Θ[n] = θ
        Φ[n] = ϕ
    end

    return hcat(sin.(Θ).*cos.(Φ), sin.(Θ).*sin.(Φ), cos.(Θ))', hcat(Θ, Φ)'
end

"""
    swissroll(N; z₀=10, R=1/20)

Generate a point cloud of `N` distributed on a swiss roll manifold with unit radius and length ``\frac{z₀}{R}``.
"""
function swissroll(N; z₀=10, R=1/20)
    z = (z₀/R)*rand(Float64, N)
    ϕ = 1.5π .+ 3π .* rand(Float64, N)

    return hcat(ϕ .* cos.(ϕ), ϕ .* sin.(ϕ), z)' .* R, hcat(ϕ, z)'
end

"""
    torus(N; R=2, r=1)

Generate a point cloud of `N` distributed on a torus, sized inner `r` and outer radius `R` respectively.
"""
function torus(N; R=2, r=1)
    θ = 2π  .* rand(Float64, N)
    ϕ = 2π .* rand(Float64, N)

    return hcat((R .+ r*cos.(θ)) .* cos.(ϕ), (R .+ r*cos.(θ)).* sin.(ϕ), r*sin.(θ))', hcat(ϕ, θ)'
end

end
