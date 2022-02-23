# SeqSpace

A **fast, self-contained** Julia library to normalize scRNAseq data and learn its underlying geometric structure in both a supervised and unsupervised fashion.

## Overview

**SeqSpace** is an _experimental_ Julia library and command line tool suite that will both normalize and subsequently learn/parameterize scRNAseq data to a low-dimensional manifold.
Our approach is orthogonal to traditional scRNAseq pipelines such as _UMAP_ or _tSNE_ that focus primarily on clustering low-dimensional embeddings;
we do not assume our data is drawn from categorical cell types.
Instead, our methodology is intended to be used on datasets that are well described by a small number of continuous degrees of freedom, for example to measure:
  * positional information of cells undergoing morphogenesis
  * time within cell cycle
  * cellular aging
However, at present, it has only been empirically validated on scRNAseq data obtained during early Drosophila embryogenesis.

There are many loosely connected library modules that are designed to help parameterize low-dimensional scRNAseq data in some capacity .
This documentation is written to both help navigate across these different modules, as well as to describe and motivate the algorithmic design in detail.
The main functionality contained within the codebase is:
  1. scRNAseq normalization
  2. scRNAseq pointcloud operations
  3. supervised scRNAseq spatial mapping
  4. unsupervised scRNAseq manifold learning

We refer the interested reader to both our in-depth algorithmic expositions as well as the library API documentation for more details.

## Installation

There are multiple ways to install the SeqSpace library

### From Julia REPL
```julia
    (@v1.x) pkg> add https://github.com/nnoll/seqspace.git
```

### From Command Line
```bash
    julia -e 'using Pkg; Pkg.add("https://github.com/nnoll/seqspace.git"); Pkg.build()'
```

### Local Environment

Clone the repository.
```bash
    git clone https://github.com/nnoll/seqspace.git && cd seqspace
```

Build the package. This will create a seperate Julia environment for SeqSpace
```bash
    julia --project=. -e 'using Pkg; Pkg.build()'
```

Enter the REPL
```bash
    julia --project=.
```

## Citing
TBA
