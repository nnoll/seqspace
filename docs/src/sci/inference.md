# scRNAseq Spatial Inference

## Introduction

In order to understand the regulation of morphogenesis, it is paramount to understand both the dynamics of gene expression at the single-cell level, as well as, interactions between the transcriptomic state of neighboring cells.
With the advent of single-cell sequencing technology, it has now become possible to directly measure the transcriptome of cellular aggregates at cellular resolution, however, parameterizing such data by _space_ remains challenging.
Specifically, in the process of isolating individual cells for downstream sequencing, embryonic cells must be dissociated and suspended in liquid.
This prevents straightforward application of scRNAseq technology to the problems of Developmental Biology.

The straightforward resolution to this conundrum is to leverage pre-existing databases of known in-situ markers.
Such curated datasets should provide the spatial expression profiles for a small subset of genes that could be matched against scRNAseq counts measured from the same organism at the same stage of development.
Once such a subset of genes are "matched" between the in-situ and scRNAseq data, the spatial pattern for each gene for the remainder of the transcriptome would come for "free."
The collection of estimated gene expression patterns would function as a high-resolution atlas over physical space of the embryo's shape; a potentially important resource for researchers.

Additionally, such an inference would provide us with an estimate of the position on the embryo each cell was sampled from.
Assuming one has the ability to detect the intrinsic manifold of gene expression, as detailed elsewhere, such positional labels would provide an interesting overlay to attempt to learn the genotype to space map encoded by the genome.
Below we provide an (incomplete) overview of past attempts at this problem, as well as detail our attempt to infer the position of scRNAseq cells.

## Overview of current methods

**FILL ME OUT**
There are published techniques that attempt to solve this problem, however our attempts to utilize them were unsuccessful.

## Our approach

### Objective function
Let ``\alpha`` and ``a`` denote indices over scRNAseq cells and embryonic spatial positions.
Our goal is to solve for the sampling probability distribution ``\rho_{a\alpha}``, i.e. the probability that cell ``\alpha`` was sampled from position ``a``.
We pose the language of regularized optimal transport, and thus the solution found at the extrema of the following objective function
```math
    \tag{1} F(\vec{g}_\alpha, \vec{G}_a) \equiv \displaystyle\sum\limits_{\alpha,a} C_{a\alpha}\left(\vec{g}_\alpha,\vec{G}_a\right)\rho_{a\alpha} + T\rho_{a\alpha}\log\left(\rho_{a\alpha}\right)
```
where ``\vec{g}_\alpha`` and ``\vec{G}_a`` denote the transcriptomic state of cell ``\alpha`` and position ``a`` respectively.
In the parlance of optimal transport, ``C_{a\alpha}`` is a cost matrix - it denotes the energy required to map cell \alpha onto position a.
For now, we leave the functional form general but understood to implicitly depend upon the gene expression of both sequenced cell and in-situ position.
``T`` is a hyperparameter akin to thermodynamic "temperature"; it controls the precision that we demand of the inferred position for each sequenced cell.
As ``T \rightarrow 0``, each cell must bijectively map onto a spatial position; the problem reduces to the assignment problem
Conversely, as ``T \rightarrow \infty``, the entropic term dominates; the solution is the uniform distribution.

If extremized as written, Eq. (1) would not result in a well-formed probability distribution.
Specifically, we must constrain the row and column sum of ``\rho_{a\alpha}`` to have correctly interpreted marginals.
In order for ``\rho_{a\alpha}`` to be interpreted as the probability that cell ``\alpha`` was sampled from position ``a``, ``\forall \alpha \ \sum_a \rho_{a\alpha} = 1`` must hold.
Additionally, we assume there were no biases in cellular isolation and thus each position was sequencing uniformally, and thus impose uniform coverage ``\forall a \ \sum_\alpha \rho_{a\alpha} = \frac{N_c}{N_x}``.
``N_x`` and ``N_c`` denote the number of in-situ positions and sequenced cells respectively.
Taken together, our full Free Energy is of the form
```math
    \tilde{F}(\vec{g}_\alpha,\vec{G}_a)\!\equiv\!\displaystyle\sum\limits_{\alpha,a} C_{a\alpha}\left(\vec{g}_\alpha,\vec{G}_a\right)\rho_{a\alpha} + T\rho_{a\alpha}\log\left(\rho_{a\alpha}\right) + \displaystyle\sum_a\Lambda_a\left[\frac{N_c}{N_x}\!-\!\sum_\alpha \rho_{a\alpha} \right] + \displaystyle\sum_\alpha \lambda_\alpha\left[1\!-\!\sum_a\rho_{a\alpha} \right]
```

The solution is found to be
```math
    \tag{2} \rho_{a\alpha}^* = e^{\lambda_a} e^{-T^{-1}\left(C_{a\alpha}\left(\vec{g}_\alpha,\vec{G}_a\right)-1\right)} e^{\Lambda_\alpha}
```
where ``\lambda_a`` and ``\Lambda_\alpha`` can be found by utilizing the Sinkhorn-Knopp algorithm in conjunction with the marginal constraints prescribed above.
Eq. (2) provides a fast, scalable algorithm to estimate the sampling posterior given a cost matrix ``C_{a\alpha}(\vec{g}_a,\vec{G}_\alpha)``.
All that remains is to formulate an explicit model.

### Cost matrix
As seen by Eq. (2), the cost matrix can be viewed as the energy of a Boltzmann distribution: ``C_{a\alpha}-1 \equiv E\left(\vec{g}_\alpha, \vec{G}_a\right)``
Hence, an obvious interpretation of ``E(\vec{g}_\alpha, \vec{G}_a)`` is as the negative log-likelihood that ``\vec{g}_\alpha`` and ``\vec{G}_a`` were sampled from the sample entity.
Our first simplifying assumption is that genes within the database are statistically independent of each other and thus the energy is additive
```math
    E(\vec{g}_\alpha, \vec{G}_a) = \frac{1}{N_g}\displaystyle\sum\limits_{i=1}^{N_g} \varepsilon\left(g_{\alpha i}, G_{ai}\right)
```
where ``i`` indexes genes and ``\varepsilon`` denotes the single-body energetics.
Thus the problem has been reduced to parameterizing the log-likelihood that ``g_\alpha`` and ``G_a`` where sampled from the sample underlying cell.
However, the complication is that our scRNAseq data and the in-situ expression database are not directly relatable; both data are in manifestly different unit systems!
Furthermore, there is no _a priori_ obvious functional relationship one can use for regression to ultimately transform counts in one dataset to another.

Overview:
+ Look at each CDF of individual gene.
+ Optimal transport of each 1D distribution: can be solved analytically.
+ Denote cdf of ``i^{th}`` gene of scRNA and in-situ by ``\phi_i`` and ``\Phi_i`` respectively
+ Provides us a unique map from one space to another that minimizes distortions in the observed distributions.
+ Assume Gaussian sampling probability.
+ Mean is given by the in-situ database.
+ Variance of Gaussian simply renormalizes the temperature of the original problem. Can unambiguously set to 1.
+ Result is
```math
    \varepsilon\left(g_{\alpha i}, G_{ai}\right) \equiv \left(\Phi^{-1}_i\left(\phi_i\left(g_{\alpha i}\right)\right) - G_{ai}\right)^2
```

## Results

## Discussion

Unfortunately, such databases only exist for a select few model organisms and thus limit the applicability of this approach.
Our hope is to leverage phenomenology gleaned from the correlations between expression and space for organisms with such a database that 
