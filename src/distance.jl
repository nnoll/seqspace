module Distances

using LinearAlgebra
using Statistics, StatsBase

export euclidean, jensen_shannon

"""
    euclidean²(X)

Compute the pairwise squared distances between points `X` according to the Euclidean metric.
Assumes `X` is sized ``d \times N`` where ``d`` and ``N`` denote dimensionality and cardinality respectively.
"""
function euclidean²(X)
    dotprod = X'*X
    vecnorm = vec(diag(dotprod))

    return (vecnorm' .+ vecnorm) .- 2*dotprod
end

"""
    euclidean(X)

Compute the pairwise distances between points `X` according to the Euclidean metric.
Assumes `X` is sized ``d \times N`` where ``d`` and ``N`` denote dimensionality and cardinality respectively.
"""
euclidean(X) = .√euclidean²(X)

"""
    kullback_liebler(p, q)
Compute the pairwise distances between probability distributions `p` and `q` according to the Kullback-Liebler divergence.
Assumes `p` and `q` are normalized.
"""
kullback_liebler(p, q) = sum(x .* log(x ./ y) for (x,y) ∈ zip(p,q) if x > 0 && y > 0)

"""
    jensen_shannon(P)
Compute the pairwise distances between probability distributions `P` according to the Jensen-Shannon divergence.
Assumes `P` is sized ``d \times N`` where ``d`` and ``N`` denote dimensionality and cardinality respectively.
"""
function jensen_shannon(P)
    D = zeros(size(P,2), size(P,2))
    Threads.@threads for i in 1:(size(P,2)-1)
        for j in (i+1):size(P,2)
            M = (P[:,i] + P[:,j]) / 2
            D[i,j] = (kullback_liebler(P[:,i],M) + kullback_liebler(P[:,j],M)) / 2
            D[j,i] = D[i,j]
        end
    end

    D = .√D
    return D / mean(D[:])
end

end
