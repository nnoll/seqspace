# SeqSpace

[![Documentation](https://img.shields.io/badge/Documentation-Link-blue.svg)](https://nnoll.github.io/seqspace/)

> a Julia library to normalize scRNAseq expression data and discover its low-dimensional latent space parameterization

## Overview

**SeqSpace** provides both a Julia library, as well as a command line interface, to both normalize scRNAseq data and learn, if applicable, the underlying low-dimensional geometry.
Our methodology is intended to be used on datasets that are well described by a small number of continuous degrees of freedom and thus represents an _orthogonal_ viewpoint to that usually taken by traditional cell atlases.
The inference from gene expression to continuous variables can be accomplished in either a supervised fashion, by mapping to a user-supplied database, or via an unsupervised approach that relies upon a novel machine learning architecture.
**SeqSpace** is a standalone tool that we anticipate will be useful in analyzing scRNAseq data of developing systems.

## Installation

The core algorithm and command line tools are self-contained and require no additional dependencies.
The library is written in and thus requires Julia to be installed on your machine.
Julia binaries for all operating systems can be found [here](https://julialang.org/downloads/).

### Library

#### Local Environment

Clone the repository
```bash
    git clone https://github.com/nnoll/seqspace.git && cd seqspace
```

Build the package. This will create a separate Julia environment for **SeqSpace**
```bash
    julia --project=. -e 'using Pkg; Pkg.build()'
```

Enter the REPL
```bash
    julia --project=.
```

#### Global Package

**Important** please do not mix this method with that described above.
Instead of creating a _local_ SeqSpace specific environment, this method will install into the Julia base environment.
We recommend, unless for a specific reason, to default to installing within a local environment.
However, if needed, global installation can be achieved by running

```bash
    julia -e 'using Pkg; Pkg.add(url="https://github.com/nnoll/seqspace.git")'
```

The SeqSpace package will now be available globally within the Julia REPL.

## Citing
TBA

## License

[MIT License](LICENSE)
