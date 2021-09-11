module Voronoi

using MiniQhull
using LinearAlgebra

import ChainRulesCore: rrule, NO_FIELDS

∧(x,y) = x[1,:].*y[2,:] .- x[2,:].*y[1,:]

function boundary(d)
    d == 2 && return [[-1;-1] [-1;+1] [+1;+1] [+1;-1]]

    b = zeros(d, 2^d)
    for i in 0:(2^d-1)
        for n in 0:(d-1)
            b[n+1,i+1] = (((i >> n) & 1) == 1) ? +1 : -1
        end
    end
    return b
end

const NB = 4
function tessellation(q)
    # q = hcat(boundary, x)
    triangulation = delaunay(q)

    # order triangulation
    areas = (q[:,triangulation[2,:]] .- q[:,triangulation[1,:]]) ∧ (q[:,triangulation[3,:]] .- q[:,triangulation[2,:]])
    orientation = sign.(areas)
    for i in findall(orientation .== -1)
        triangulation[2,i], triangulation[3,i] = triangulation[3,i], triangulation[2,i] 
    end

    # compute voronoi vertices
    r = zeros(eltype(q), size(q,1), size(triangulation,2))
    for t in 1:size(triangulation,2)
        for i in 1:3
            j = (i % 3) + 1
            k = (j % 3) + 1

            α,β,γ = triangulation[i,t], triangulation[j,t], triangulation[k,t]

            qα² = sum((q[:,α]).^2)
            r[1,t] += qα²*(q[2,γ] - q[2,β])
            r[2,t] -= qα²*(q[1,γ] - q[1,β])
        end
    end

    r[1,:] = r[1,:] ./ abs.(areas)
    r[2,:] = r[2,:] ./ abs.(areas)

    triangulation, r
end

function areas(x)
    q = hcat(boundary(2), x)
    triangulation = delaunay(q)

    a = 0.5*[ 
        let
            q[1,t[1]]*(q[2,t[2]]-q[2,t[3]]) + 
            q[1,t[2]]*(q[2,t[3]]-q[2,t[1]]) + 
            q[1,t[3]]*(q[2,t[1]]-q[2,t[2]])
        end for t in eachcol(triangulation) 
    ]
    s = sign.(a)

    return s.*a
end

function rrule(::typeof(areas), x)
    q = hcat(boundary(2), x)
    triangulation = delaunay(q)

    a = 0.5*[ 
        let
            q[1,t[1]]*(q[2,t[2]]-q[2,t[3]]) + 
            q[1,t[2]]*(q[2,t[3]]-q[2,t[1]]) + 
            q[1,t[3]]*(q[2,t[1]]-q[2,t[2]])
        end for t in eachcol(triangulation) 
    ]
    s = sign.(a)

    return s.*a, (∂a) -> let
        ∂x = zeros(size(x))

        for (i,t) in enumerate(eachcol(triangulation))
            if t[1] > NB
                ∂x[1,t[1]-NB] += (q[2,t[2]]-q[2,t[3]])*∂a[i]*s[i]
                ∂x[2,t[1]-NB] -= (q[1,t[2]]-q[1,t[3]])*∂a[i]*s[i]
            end
            if t[2] > 4
                ∂x[1,t[2]-NB] += (q[2,t[3]]-q[2,t[1]])*∂a[i]*s[i]
                ∂x[2,t[2]-NB] -= (q[1,t[3]]-q[1,t[1]])*∂a[i]*s[i]
            end
            if t[3] > 4
                ∂x[1,t[3]-NB] += (q[2,t[1]]-q[2,t[2]])*∂a[i]*s[i]
                ∂x[2,t[3]-NB] -= (q[1,t[1]]-q[1,t[2]])*∂a[i]*s[i]
            end

        end

        (NO_FIELDS, 0.5*∂x)
    end

end

# derivative of det given by adjugate (which is just rescaled inverse)
# see https://en.wikipedia.org/wiki/Adjugate_matrix

volume(simplex) = det(vcat(simplex, ones(1, size(simplex,2))))

function volumes(x)
    d = size(x,1)
    q = hcat(boundary(d), x)

    simplices = delaunay(q)

    Ω = [ volume(hcat((q[:,i] for i in simplex)...)) for simplex in eachcol(simplices) ]
    return Ω ./ factorial(d)
end

end
