# Manifold learning

## Empirical analysis

* Point cloud scaling
* Physically close cells are close in expression
* Isomap analysis picks up spatial gradients

## Initial considerations

Want:
* Dimensional reduction
* Nonlinear
* Differentiable
* Generalizable
* Unsupervised

Natural choice for an autoencoder.
Utilize known positional labels as a validation step.
Not used for training purposes.

## Network architecture

How to pick depth? Width?
Worry about overfitting: enter dropout and batch normalization.
Vanilla autoencoder: latent space is not readily interpretable.

### Topological conservation
In order to learn an interpretable latent space representation of the intrinsic gene expression manifold, we wish to constrain the estimated pullback to conserve the topology of the input scRNAseq data.
This immediately poses the question: what topological features do we wish to preserve from the data to latent space and how do we empirically measure them?
The answers immediately determine the additional terms one must add to the objective function used for training.

We opt to utilize an explicitly _geometric_ formalism that will implicitly constrain _topology_.
The intuition for this choice is guided by _Differential Geometry_: a metric tensor uniquely defines a distance function between any two points on a manifold; the topology induced by this distance function will always coincide with the original topology of the manifold.
Thus, by imposing preservation of pairwise distances in the latent space relative to the data, we implicitly conserve topology.
It is important to note that this assumes our original scRNAseq data is sampled from a metric space that we have access to.
We note that there have been recent promising attempts at designing loss functions parameterized by explicit topological invariants formulated by _Topological Data Analysis_, e.g. persistent homology.
Lastly, one could envision having each network layer operate on a simplicial complex, rather than a flat vector of real numbers, however it is unclear how to parameterize the feed-forward function.

Thus the first task is to formulate an algorithm to approximate the metric space the point cloud is sampled from and subsequently utilize our estimate to compute all pairwise distances.
Again we proceed guided by intuition gleaned from _Differential Geometry_: pairwise distances within local neighborhoods are expected to be well-described by a Euclidean metric in the tangent space.
Conversely, macroscopic distances can only be computed via integration against the underlying metric along the corresponding geodesic.
As such, we first estimate the local tangent space of our input data by computing pairwise distances within local neighborhoods around each point, either defined by a fixed radius or fixed number of neighbors.
This defines a sparse, undirected graph in which edges only exist within our estimated tangent spaces and are weighted by the euclidean distance within the embedded space.
The resultant neighborhood graph serves as the basis for many dimensional reduction algorithms, such as **Isomap**, **UMAP** and **tSNE**.
Pairwise distances between _any_ two points in the original dataset can then be found by simple graph traversal to find the shortest possible path between two graph vertices, the discrete analog of a continuum geodesic.
It has been shown that the distance estimated by this algorithm asymptotically approaches the true distance as the number of data points sampled increases.
We denote ``D_{\alpha\beta}`` as the resultant pairwise distances between cell ``\alpha,\beta``.

#### Isometric formulation
The most straightforward manner to preserve distances between the input data and the latent representation is to impose isometry, i.e. distances in both spaces quantitatively agree.
This would be achieved by supplementing the objective function with the term
```math
E_{iso} = \displaystyle\sum\limits_{\alpha,\beta} \left(D_{\alpha\beta} - \left|\left| \xi_\alpha - \xi_\beta \right|\right| \right)^2
```
Utilizing this term is problematic for several reasons:
1. Large distances dominate the energetics and as such large-scale features of the intrinsic manifold will be preferentially fit.
2. Generically, ``d`` dimensional manifolds can not be isometrically embedded into ``\mathbb{R}^d``, e.g. the sphere into the plane.
3. It trusts the computed distances quantitatively. We simply want close cells to be close in the resultant latent space.

#### Differentiable ranking
Consider a vector ``\psi_\alpha`` of scores of length ``n`` we wish to rank.
Furthermore, define ``\sigma \in \Sigma_n`` to be an arbitrary permutation of ``n`` such scores.
We define the **argsort** to be the permutation that sorts ``\psi`` in descending order
```math
    \bar{\sigma}\left(\bm{\psi}\right) \equiv \left(\sigma_1\left(\bm{\psi}\right),...,\sigma_n\left(\bm{\psi}\right)\right) \qquad \text{such that} \qquad
    \psi_{\bar{\sigma}_1} \ge \psi_{\bar{\sigma}_1} \ge ... \ge \psi_{\bar{\sigma}_n}
```

The definition of the **sorted** vector of scores ``\bar{\bm{\psi}}_\alpha \equiv \psi_{\bar{\sigma}_\alpha}`` thus follows naturally.
Lastly, the **rank** of vector ``\bm{\psi}`` is defined as the inverse permutation of **argsort**.
```math
    R\left(\bm{\psi}\right) \equiv \bar{\sigma}^{-1}\left(\bm{\psi}\right)
```
We wish to devise an objective function that contains functions of the rank of some latent space variables.
However, ``R(\bm{\psi})`` is a non-differentiable function; it maps a vector in ``\mathbb{R}^n`` to a permutation of ``n`` items.
Hence, we can not directly utilize the rank in a loss function as there is no way to backpropagate gradient information to the network parameters.
In order to rectify this limitation, we first reformulate the ranking problem as a linear programming problem that permits efficient regularization.
Note, the presentation here follows closely the original paper [^1]

[^1]: [Fast Differentiable Sorting and Ranking](https://arxiv.org/abs/2002.08871)

##### Linear program formulation

The **sorting** and **ranking** problem can be formulated as discrete optimization over the set of n-permutations ``\Sigma_n``
```math
    \bar{\sigma}\left(\bm{\psi}\right) \equiv \underset{\bm{\sigma}\in\Sigma_n}{\mathrm{argmax}} \ \displaystyle\sum\limits_{\alpha} \psi_{\sigma_\alpha} \rho_{\alpha}
```
```math
    R\left(\bm{\psi}\right)
    \equiv \bar{\sigma}\left(\bm{\psi}\right)^{-1} 
    \equiv \left[\underset{\bm{\sigma}\in\Sigma_n}{\mathrm{argmax}} \ \displaystyle\sum\limits_{\alpha} \psi_{\sigma_\alpha} \rho_{\alpha} \right]^{-1}
    \equiv \left[\underset{\bm{\sigma^{-1}}\in\Sigma_n}{\mathrm{argmax}} \ \displaystyle\sum\limits_{\alpha} \psi_{\alpha} \rho_{\sigma^{-1}_\alpha} \right]^{-1}
    \equiv \underset{\bm{\pi}\in\Sigma_n}{\mathrm{argmax}} \ \displaystyle\sum\limits_{\alpha} \psi_\alpha \rho_{\pi(\alpha)}
```
where ``\rho_\alpha \equiv \left(n, n-1, ..., 1\right)``
In order to regularize the problem, and thus allow for continuous optimization, we imagine the convex hull of all permutations induced by an arbitrary vector ``\bm{\omega} \in \mathbb{R}^n``.
```math
    \Omega\left(\bm{\omega}\right) \equiv \text{convhull}\left[\left\{\bm{\omega}_{\sigma_\alpha}: \sigma \in \Sigma_n \right\}\right] \subset \mathbb{R}^n
```
This is often referred to as the _permutahedron_ of ``\bm{\omega}``; it is a convex polytope in n-dimensions whose vertices are the permutations of ``\bm{\omega}``
It follows directly from the fundamental theorem of linear programming, that the solution will almost surely be achieved at the vertex.
Thus the above discrete formulation can be rewritten as an optimization over continuous vectors contained on the _permutahedron_
```math
    \bm{\psi}_{\bar{\sigma}\left(\bm{\psi}\right)} \equiv \underset{\bm{\omega}\in\Omega\left(\bm{\psi}\right)}{\mathrm{argmax}} \ \bm{\omega}\cdot\bm{\rho}
    \qquad
    \bm{\rho}_{R\left(\bm{\psi}\right)} \equiv \underset{\bm{\omega}\in\Omega\left(\bm{\rho}\right)}{\mathrm{argmax}} \ \bm{\psi}\cdot\bm{\omega}
```
Utilizing the fact that ``\rho_{R\left(\bm{\psi}\right)} = R\left(-\bm{\psi}\right)``
```math
    R\left(\bm{\psi}\right) \equiv -\underset{\bm{\omega}\in\Omega\left(\bm{\rho}\right)}{\mathrm{argmax}} \ \bm{\psi}\cdot\bm{\omega}
```
Unfortunately, since ``\bm{\psi}`` appears in the **rank** objective function, any small perturbation in ``\bm{\psi}`` can force the solution of the linear program to discontinuously transition to another vertex.
As such, in its current form, it is still not differentiable.
Note, this is not true for the sorted vector, it appears in the constraint polyhedron; it has a unique Jacobian and can be directly used in neural networks.
The only way to proceed is to introduce convex regularization.

##### Regularization
We revise our objective function by Euclidean projection and thus introduce quadratic regularization on the norm of the solution.
Specifically, we define the **soft rank** operators as the extrema of the objective function
```math
    \tilde{R}\left(\bm{\psi}\right) \equiv \underset{\bm{\omega}\in\Omega\left(\bm{\rho}\right)}{\mathrm{argmax}} \left[ -\bm{\psi}\cdot\bm{\omega}
    - \frac{\epsilon}{2}\left|\left|\omega\right|\right|^2 \right]
```
Note that the limit ``\epsilon \rightarrow 0`` reproduces the linear programming formulation of the rank operator introduced above.
Conversely, in the limit ``\epsilon \rightarrow \infty``, the solution will go to a constant vector that has the smallest modulus on the _permutahedron_.

##### Solution
It has been demonstrated before that the above problem reduces to simple isotonic regression[^1][^2].
Specifically,
```math
R\left(\bm{\psi}\right) = -\frac{\bm{\psi}}{\epsilon} -
    \left[\underset{\omega_1 \ge \omega_2 \ge ... \ge \omega_n}{\mathrm{argmin}}
    \frac{1}{2} \left|\left|\bm{\omega} + \bm{\rho} + \frac{\bm{\psi}}{\epsilon} \right|\right|^2\right]_{\sigma^{-1}(\bm{\psi})}
\equiv -\frac{\bm{\psi}}{\epsilon} - \tilde{\bm{\omega}}\left(\bm{\psi},\bm{\rho}\right)
```
Importantly, isotonic regression is well-studied and can be solved in linear time.
Furthermore, the solution admits a simple, calculatable Jacobian
```math
\partial_{\psi_\alpha} R_\beta\left(\bm{\psi}\right)
= \frac{-\delta_{\alpha\beta}}{\epsilon} - \partial_{\psi_\alpha}\tilde{\omega_\beta}\left(\bm{\psi},\bm{\rho}\right)
= \frac{-\delta_{\alpha\beta}}{\epsilon} - 
    \begin{pmatrix}
    \bm{B}_1 & \bm{0} & \bm{0} \\
    \bm{0}   & \ddots & \bm{0} \\
    \bm{0}   & \bm{0} & \bm{B}_m \\
    \end{pmatrix}_{\alpha\beta}
```
where ``\bm{B}_i`` denotes the matrix corresponding to the `i^{th}` block obtained during isotonic regression.
It is a constant matrix whose number of rows and columns equals the size of the block, and whose values all sum to 1.

[^2]: [SparseMAP: Differentiable Sparse Structured Inference](https://arxiv.org/abs/1802.04223)

#### Loss function

### Uniform sampling of latent space

## Results
