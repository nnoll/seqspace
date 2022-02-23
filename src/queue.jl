# Used for Dijisktra's algorithm
module PriorityQueue

import Base:
    ∈, length, minimum, 
    sizehint!, push!, take!, insert!

export RankedQueue
export update!

"""
    parent(i)

Return the index of the parent of node `i`.
"""
parent(i) = i÷2

"""
    left(i)

Return the index of the left child of node `i`.
"""
left(i)   = 2*i
"""
    right(i)

Return the index of the right child of node `i`.
"""
right(i)  = 2*i+1

"""
    struct RankedQueue{T <: Real, S <: Any}
        rank :: Array{T, 1}
        data :: Array{S, 1}
    end

Maintains a priority queue of `data`.
Each datum has a `rank` that determines it's priority in the queue.
`rank` and `data` are sorted in ascending order.
"""
struct RankedQueue{T <: Real, S <: Any}
    rank :: Array{T, 1}
    data :: Array{S, 1}
end

∈(x::S, q::RankedQueue{T,S}) where {T <: Real, S <: Any} = x ∈ q.data

function sizehint!(q::RankedQueue, n)
    sizehint!(q.rank, n)
    sizehint!(q.data, n)
end

"""
    rotateup!(q::RankedQueue, i)

Modify the ranked queue by pushing up the node at index `i` until the priority is sorted again.
"""
function rotateup!(q::RankedQueue, i)
    i == 1 && return i

    while i > 1 && (p=parent(i); q.rank[i] < q.rank[p])
        q.rank[i], q.rank[p] = q.rank[p], q.rank[i]
        q.data[i], q.data[p] = q.data[p], q.data[i]
        i = p
    end

    return i
end

"""
    rotateup!(q::RankedQueue)

Modify the ranked queue by pushing up the last element until the priority is sorted.
"""
rotateup!(q::RankedQueue) = rotateup!(q, length(q.rank))

"""
    rotatedown!(q::RankedQueue, i)

Modify the ranked queue by pushing down the node at index `i` until the priority is sorted.
"""
function rotatedown!(q::RankedQueue, i)
    left(i) > length(q) && return i

    child(i) = (right(i) > length(q) || q.rank[left(i)] < q.rank[right(i)]) ? left(i) : right(i)
    while left(i) <= length(q) && (c=child(i); q.rank[i] > q.rank[c])
        q.rank[i], q.rank[c] = q.rank[c], q.rank[i]
        q.data[i], q.data[c] = q.data[c], q.data[i]
        i = c
    end

    return i
end

"""
    rotatedown!(q::RankedQueue)

Modify the ranked queue by pushing down the root until the priority is sorted.
"""
rotatedown!(q::RankedQueue) = rotatedown!(q, 1)

function RankedQueue(X::Tuple{S, T}...) where {T <: Real, S <: Any}
    q = RankedQueue{T,S}([x[2] for x in X], [x[1] for x in X])
    i = length(X) ÷ 2
    while i >= 1
        rotatedown!(q, i)
        i -= 1
    end

    return q
end

length(q::RankedQueue) = length(q.rank)

# external methods
# XXX: there be dragons - we don't check for duplicated data being passed to us

minimum(q::RankedQueue) = (data=q.data[1], rank=q.rank[1])

"""
    insert!(q::RankedQueue{T}, data::S, rank::T) where {T <: Real, S <: Any}

Push a new element `data` with priority `rank` onto the ranked queue `q`.
Rotates the queue until priority is sorted in ascending order.
"""
function insert!(q::RankedQueue{T}, data::S, rank::T) where {T <: Real, S <: Any}
    push!(q.rank, rank)
    push!(q.data, data)

    rotateup!(q)
end

"""
    take!(q::RankedQueue)

Pop off the element with element with lowest rank/highest priority.
"""
function take!(q::RankedQueue)
    r, q.rank[1] = q.rank[1], q.rank[end]
    pop!(q.rank)

    d, q.data[1] = q.data[1], q.data[end]
    pop!(q.data)

    rotatedown!(q)

    return (data=d, rank=r)
end

"""
    update!(q::RankedQueue{T, S}, data::S, new::T) where {T <: Real, S <: Any}

Change the priority of element `data` to rank `new`.
Will panic if `data` is not contained in queue `q`.
"""
function update!(q::RankedQueue{T, S}, data::S, new::T) where {T <: Real, S <: Any}
    (i = findfirst(d -> d == data, q.data)) == nothing && panic("attempting to update a non-existent data value")

    old = q.rank[i]
    q.rank[i] = new

    return if new > old
               rotatedown!(q, i)
           elseif new < old
               rotateup!(q, i)
           else
               i
           end
end

function test()
    Q = RankedQueue((0,1), (1,2), (2,10), (3,23), (4,0))
    @show Q
    insert!(Q, 5, -1)
    @show Q

    nothing
end

end
