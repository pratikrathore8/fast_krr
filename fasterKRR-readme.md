# fasterKRR

R translation of the [fast_krr](https://github.com/pratikrathore8/fast_krr) Python package by an anonymous contributor who goes by John Slade.

## Installation

### 1. Install dependencies

From the R console:

```r
install.packages("CVST", dep = TRUE)
```

Optional packages for additional functionality:

```r
install.packages(c("RSpectra", "knitr", "rmarkdown", "testthat"), dep = TRUE)
```

### 2. Install fasterKRR

From your shell, in the repository root:

```sh
R CMD INSTALL fasterKRR
```

## Usage

```r
require("fasterKRR")

# View the package manual
help(package = "fasterKRR")

# Browse the tutorial vignette
vignette("fasterKRR-tutorial")

# Run the demo
demo("fasterKRR-demo", package = "fasterKRR")
```
