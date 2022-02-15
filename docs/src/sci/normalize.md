# scRNAseq Normalization

## Introduction

scRNA sequencing is a relatively novel technique added to the Biologist's toolkit that allows one to quantitatively probe the "Statistical Mechanics" of Biology.
In contrast to traditional bulk RNAseq techniques, which provide population averages, scRNAseq technology measures the complete ensemble of cellular gene expression.
Consequently, considerable activity has been focused on mapping the taxonomy of cell states; projects such as the Human Cell Atlas promise to provide a complete enumeration of microscopic cell types within the human body.
However, despite recent progress, scRNAseq data have many numerous technical limitations and sources of statistical bias that must be considered before any downstream analysis.
The sources of noise within the data include, but are not limited to:
  * **Sequence depth bias:** PCR and reverse transcription efficiency will vary across reactions, resulting in artificial variation of sequencing depth across different cells.
  * **Amplification bias:** PCR primers are not perfectly random. As a result, some genes will amplify preferentially over others. This will distort the underlying expression distribution.
  * **Batch effect:** scRNAseq runs have non-trivial distortions causing cells within a given sequencing run to have greater correlation within than across batches.
  * **Dropout:** Due to molecular competition in the underlying amplification reactions, some genes will fail to amplify during PCR due to early losses. This will lead to more zeros in the resulting count matrix than you would expect by chance.

These biases are approximately rectified by a preprocessing step known as **normalization**.
At a coarse level, all such methods can be thought of as transforming the obtained count matrix from absolute numbers into an estimate of differential expression, i.e. a comparison of a count *relative* to the measured distribution.
Such a procedure is analogous to that of the z-score of a univariate normally-distributed random variable; however, in the case of scRNAseq data, we don't have access to the underlying sampling distributions *a priori*, it must be estimated empirically.
The choice of sampling prior, and details of how the distribution is estimated, delineate normalization practices.

## Overview of current methods

**TODO:** FILL OUT

## Our approach

We model the count matrix ``n_{\alpha,i}``, where ``\alpha`` and ``i`` indexes cells and genes respectively, obtained from scRNA sequencing as an unknown, low-rank mean ``\mu_{\alpha,i}`` with additive full-rank noise ``\delta_{\alpha,i}`` that captures all unknown technical noise and bias.
```math
    \tag{1} n_{\alpha,i} = \mu_{\alpha,i} + \delta_{\alpha,i} 
```
Our goal during the normalization procedure is two-fold:
  1. Estimate the low-rank mean ``\mu_{\alpha,i}``.
  2. Estimate the sampling variance ``\langle\delta_{\alpha,i}^2\rangle`` of the experimental noise. Brackets denote an average over (theoretical) realizations of sequencing. This will eventually require an explicit model.

Given both quantities, normalization over gene *and* cell-specific biases can be achieved by imposing the variances of all marginal distributions to be one.
Specifically, we rescale all counts by cell-specific ``c_\alpha`` and gene-specific ``g_i`` factors ``\tilde{n}_{\alpha,i} \equiv c_\alpha n_{\alpha,i} g_i`` such that
```math
    \tag{2} \displaystyle\sum\limits_{\alpha} \langle \tilde{\delta}_{\alpha,i}^2 \rangle = \displaystyle\sum\limits_{\alpha}c_\alpha^2 \langle \delta_{\alpha,i}^2 \rangle g_i^2 = N_g \quad \text{and} \quad \displaystyle\sum\limits_{i} \langle \tilde{\delta}_{\alpha,i}^2 \rangle = \displaystyle\sum\limits_{i}c_\alpha^2 \langle \delta_{\alpha,i}^2 \rangle g_i^2 = N_c
```
This system of equations can be solved by the Sinkhorn-Knopp algorithm, provided we have a model for ``\langle\delta_{\alpha,i}^2\rangle`` parameterized by measurables.
Within this formalism, this choice of model fully determines the normalization scheme.
Owing to dropout and other sources of overdispersion, we model scRNAseq counts as sampled from an empirically estimated [Heteroskedastic Negative Binomial](@ref) distribution.

### Case studies

Before detailing our explicit algorithm, it is helpful to consider a few simpler examples.
In the following sections, homoskedastic is used to denote count matrices whose elements are
 independent and identically distributed (*iid*), while heteroskedastic denotes more complicated scenarios where each element has a unique distribution.

#### Homoskedastic Gaussian

This is slight perturbation of canonical [random matrix theory](https://en.wikipedia.org/wiki/Random_matrix) and will serve as a useful pedagogical starting point.
Assume ``\mu_{\alpha,i}`` is a quenched, low-rank matrix and each element of ``\delta_{\alpha,i}`` is sampled from a Gaussian with zero mean and variance ``\sigma^2``.
In this limit, Eq.(1) reduces to the well-studied spiked population covariance model.

If ``\mu_{\alpha,i} = 0``, the spectral decomposition of the count matrix ``n_{\alpha,i}`` would be given by the [Marchenko-Pastur distribution](https://en.wikipedia.org/wiki/Marchenko–Pastur_distribution) asymptotically.
As such, singular values would be bounded by ``\bar{\lambda} \equiv 1+\sigma\sqrt{N_g/N_c}``.
Now consider the case of a rank 1 mean, i.e. 1 cell type in the data.
```math
    n_{\alpha,i} = \gamma x_\alpha \bar{x}_i + \delta_{\alpha,i}
```
It has been shown[^1][^2] that this model exhibits the following asymptotic phase transition:
  * If ``\gamma \le \bar{\lambda}``, the top singular value of ``n_{\alpha,i}`` converges to ``\bar{\lambda}``. Additionally, the overlap of the left and right eigenvector with ``x`` and ``\bar{x}`` respectively converge to 0.
  * If ``\gamma > \bar{\lambda}``, the top singular value of ``n_{\alpha,i}`` converges to ``\gamma + \bar{\lambda}/\gamma``. Additionally, the overlap of the left and right eigenvector with ``x`` and ``\bar{x}`` respectively converge to ``1-(\bar{\lambda}/\gamma)^2``
This procedure can generalized to higher rank spike-ins; sub-leading principal components can be found by simply subtracting the previously inferred component from the count matrix ``n_{\alpha,i}``.
As such, we can only expect to meaningful measure the principal components of $\mu_{\alpha,i}$ that fall above the sampling noise floor, given by the Marchenko-Pastur distribution.
Consequently, this forces us to define the statistically significant **rank** of the count matrix.

[^1]: [The singular values and vectors of low rank perturbations of large rectangular random matrices](https://arxiv.org/abs/1103.2221)
[^2]: [The largest eigenvalue of rank one deformation of large Wigner matrices](https://arxiv.org/abs/math/0605624)

#### Heteroskedastic Poisson

The above results have been shown to hold for non-Gaussian Wigner noise matrices[^3], provided each element is still *iid*.
This is manifestly *not* the case we care about; the normalization procedure must account for heterogeneous sampling variances across both cells and genes.
However, it has been shown[^4][^5] that the distribution of eigenvalues converges almost surely to the Marchenko-Pastur distribution, provided the constraint of Eq.(2) is satisfied!
In other words, the eigenvalues of a heteroskedastic matrix is expected converge to that of a random Wigner matrix, provided the row and column variances are uniform (set to unity for convenience).

For the remainder of this section, assume the count matrix is sampled from a Poisson distribution, such that ``\langle\delta_{\alpha,i}^2\rangle = \mu_{\alpha,i}``.
As ``n_{\alpha,i}`` is an unbiased estimator for the mean ``\mu_{\alpha,i}``, Eq (2) reduces to
```math
\displaystyle\sum\limits_{\alpha}c_\alpha^2 n_{\alpha, i} g_i^2 = N_g \quad \text{and} \quad \displaystyle\sum\limits_{i}c_\alpha^2 n_{\alpha,i} g_i^2 = N_c
```
which provides an explicit system of equations to estimate the scaling factors ``c_\alpha`` and ``g_i``.
Once obtained, ``\tilde{\mu}_{\alpha,i}`` can be estimated via singular value decomposition of ``c_\alpha n_{\alpha,i} g_i``; all components with singular value greater than ``\bar{\lambda} = \sqrt{N_c}+\sqrt{N_g}`` can be confidently attributed to the "true mean" ``\mu_{\alpha,i}`` while all other components fall amongst the noise.
An example is shown below:

**TODO**: ADD FIGURE

[^3]: [Asymptotics of Sample Eigenstructure for a Large Dimensional Spiked Covariance Model](http://www3.stat.sinica.edu.tw/statistica/oldpdf/A17n418.pdf)
[^4]: [Biwhitening Reveals the Rank of a Count Matrix](https://arxiv.org/abs/2103.13840)
[^5]: [A Review of Matrix Scaling and Sinkhorn's Normal Form for Matrices and Positive Maps](https://arxiv.org/abs/1609.06349)

#### Heteroskedastic Negative Binomial

A negative binomial distribution is often used to model overdispersed count data, i.e. a process in which the variance grows superlinearly with the mean.
Canonically the distribution arises from the distribution of the number of successes (with probabiliy ``p``) obtained after ``\phi`` failures of a Bernoulli process.
However, the generative stochastic process can equivalently be modelled as an underlying Poisson process in which the emission rate is itself a stochastic variable drawn from a Gamma distribution.
This allows us to analytically continue ``\phi`` from an integer to the reals.
Provided we have estimated the mean ``\mu`` and the overdisperson factor ``\phi``, the unbiased estimator for the variance is given by
```math
    \langle \delta^2 \rangle= = \mu\frac{1 + \mu\phi}{1 + \phi}
```
Direct substitution into Eq. (2) would provide the necessary cell-specific $c_\alpha$ and gene-specific $g_i$ scaling factors.

## Empirical Sampling Distribution Estimation

As such, in order to normalize the estimated sampling variance, we must formulate a method to fit a negative binomial to the measured count data _per gene_.
We parameterize the distribution as follows:
```math
    p\left(n|\mu,\phi\right) = \frac{\Gamma\left(n+\phi\right)}{\Gamma\left(n+1\right)\Gamma\left(\phi\right)}\left(\frac{\mu}{\mu+\phi}\right)^n\left(\frac{\phi}{\mu+\phi}\right)^\phi
```

### Generalized Linear Model

A central complication in modeling counts of a given gene _across_ cells with the above function is that there are confounding variables that require explicit consideration.
For the present discussion, we explictly model the hetereogeneous sequencing depth across cells: naively we expect that if we sequenced a given cell with ``2x`` depth, each gene should scale accordingly.
As such, we formulate a generalized linear model
```math
    \log \mu_{i\alpha} = A_i + B_i \log n_{\alpha}
```
where ``n_\alpha`` denotes the total sequencing depth of cell ``\alpha``.
We note this formulation can be easily extended to account for other confounds such as batch effect or cell type.
The likelihood function for cell ``\alpha``, gene ``i`` is
```math
    p\left(n_{i\alpha}|A_i, B_i, n_\alpha, \phi_i\right) = \frac{\Gamma\left(n_{i\alpha}+\phi_i\right)}{\Gamma\left(n_{i\alpha}+1\right)\Gamma\left(\phi_i\right)}\left(\frac{\mu_{i\alpha}}{\mu_{i\alpha}+\phi_i}\right)^{n_{i\alpha}}\left(\frac{\phi_i}{\mu_{i\alpha}+\phi_i}\right)^{\phi_{i}}
```

### Maximum Likelihood Estimation
``\{A_i, B_i, \phi_i\}`` represent ``3N_g`` parameters we must infer from the data.
A priori this seems underdetermined given our ``N_c \times N_g`` sized matrix and thus we attempt to estimate all parameters within a maximum likelihood framework.
This is equivalent to minimizing (for each gene independently)
```math
    \mathcal{L} = \displaystyle\sum\limits_{\alpha}
        \left(n_{i\alpha} + \phi_i\right)\log\left(e^{A_{i}}n_\alpha^{B_i} + \phi_i\right) 
      - \phi_i\log\left(\phi_i\right)
      - n_{i\alpha}\left(A_i + B_i\log n_\alpha\right)
      - \log\left(\frac{\Gamma\left(n_{i\alpha}+\phi_i\right)}{\Gamma\left(n_{i\alpha}+1\right)\Gamma\left(\phi_i\right)}\right)
```

### Overfitting

### Maximum A Posterior

#### Empirical Estimation of Priors

## Results
