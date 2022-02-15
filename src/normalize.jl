module Normalize

using LinearAlgebra
using Optim, NLSolversBase
using Random, Statistics, StatsBase
using SpecialFunctions: loggamma

const ∞ = Inf

const logmean(x;ϵ=1) = exp.(mean(log.(x.+ϵ)))-ϵ
const logvar(x;ϵ=1)  = exp.(var(log.(x.+ϵ)))-ϵ

function negativebinomial(count, depth)
    loglikelihood = function(Θ)
        Θ₁,Θ₂,Θ₃ = Θ
        return -sum(
            loggamma(n+Θ₃)
          - loggamma(n+1)
          - loggamma(Θ₃)
          + n*(Θ₁+Θ₂*d)
          + Θ₃*log(Θ₃)
          - (n+Θ₃)*log(exp(Θ₁+Θ₂*d)+Θ₃)
          for (n,d) ∈ zip(count,depth)
        )
    end

    μ  = logmean(count)
    Θ₀ = [log(μ), 1.25, logvar(count)/μ - 1]
    if Θ₀[end] < 0 || isinf(Θ₀[end]) || isnan(Θ₀[end])
        Θ₀[end] = 1
    end

    return (
        Θ₀ = Θ₀,
        likelihood = TwiceDifferentiable(loglikelihood, Θ₀; autodiff=:forward),
        constraint = TwiceDifferentiableConstraints([-∞,-∞,0],[+∞,+∞,+∞]),
        residual   = function(Θ)
            Θ₁,Θ₂,Θ₃ = Θ
            μ = @. exp(Θ₁ + Θ₂*depth)
            σ = @. √(μ + μ^2/Θ₃)
            z = @. (count - μ) / σ

            z[ z .< -5 ] .= -5
            z[ z .> +5 ] .= +5

            return z
        end
    )
end

function gamma(count, depth)
    loglikelihood = function(Θ)
        Θ₁,Θ₂,Θ₃ = Θ
        return -sum(
            let
                α = Θ₃*exp(Θ₁+Θ₂*d)
                α*log(Θ₃) + (α-1)*log(x) - (Θ₃*x) - loggamma(α)
            end for (x,d) ∈ zip(count,depth)
        )
    end

    μ  = logmean(count; ϵ = 1e-10)
    Θ₀ = [log(μ), 1.25, μ/logvar(count; ϵ=1e-10)]
    if Θ₀[end] < 0 || isinf(Θ₀[end]) || isnan(Θ₀[end])
        Θ₀[end] = 1
    end

    return (
        Θ₀ = Θ₀,
        likelihood = TwiceDifferentiable(loglikelihood, Θ₀; autodiff=:forward),
        constraint = TwiceDifferentiableConstraints([-∞,-∞,0],[+∞,+∞,+∞]),
        residual   = function(Θ)
            Θ₁, Θ₂, Θ₃ = Θ
            μ = @. exp(Θ₁ + Θ₂*depth)
            σ = @. sqrt(μ / Θ₃)
            z = @. (count - μ) / σ

            z[ z .< -5 ] .= -5
            z[ z .> +5 ] .= +5

            return z
        end
    )
end

function prior(params)
    objective = function(Θ)
        Θ₁, Θ₂, Θ₃ = Θ
        return sum(
            loggamma(1/Θ₃) + log(Θ₂) .+ (abs.(params .- Θ₁)./ Θ₂).^Θ₃ .- log(Θ₃)
        )
    end

    Θ₁ = mean(params)
    Θ₂ = std(params)
    Θ₀ = [Θ₁,Θ₂,2]

    likelihood = TwiceDifferentiable(f,Θ₀;autodiff=:forward)
    constraint = TwiceDifferentiableConstraints([0,0,0],[+∞,+∞,+∞])
    hyperparam = optimize(likelihood, constraint, Θ₀, IPNewton())

    return Optim.minimizer(hyperparam)
end

function fit(stochastic, count, depth)
    model = stochastic(count, depth)
    param = optimize(model.likelihood, model.constraint, model.Θ₀, IPNewton())

    return (
        likelihood  = Optim.minimum(param),
        parameters  = Optim.minimizer(param),
        uncertainty = diag(inv(hessian!(model.likelihood, Optim.minimizer(param)))),
        residual    = model.residual(Optim.minimizer(param))
    )
end

function bootstrap(count, depth; stochastic=negativebinomial, samples=50)
    N = length(depth)

    Θ₁ = Array{Float64}(undef,samples)
    Θ₂ = Array{Float64}(undef,samples)
    Θ₃ = Array{Float64}(undef,samples)
    δL = Array{Float64}(undef,samples)

    for n in 1:samples
        ι = randperm(N)[1:2*N÷3]
        f = fit(stochastic, count[ι],depth[ι])

        δL[n] = f.likelihood
        Θ₁[n], Θ₂[n], Θ₃[n] = f.parameters
    end

    return Θ₁, Θ₂, Θ₃, δL
end

function glm(data; stochastic=negativebinomial, ϵ=1)
    average(x) = logmean(x; ϵ=ϵ)
    depth = map(eachcol(data)) do col
        col |> vec |> average
    end

    fits = [fit(stochastic,vec(gene),depth) for gene in eachrow(data)]

    return (
        likelihood = map((f)->f.likelihood,  fits),
        residual   = Matrix(reduce(hcat, map((f)->f.residual, fits))'),

        Θ₁  = map((f)->f.parameters[1],  fits),
        Θ₂  = map((f)->f.parameters[2],  fits),
        Θ₃  = map((f)->f.parameters[3],  fits),

        δΘ₁ = map((f)->f.uncertainty[1], fits),
        δΘ₂ = map((f)->f.uncertainty[2], fits),
        δΘ₃ = map((f)->f.uncertainty[3], fits),
    )
end

end
