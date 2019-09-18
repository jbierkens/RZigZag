# RZigZag: Zig-Zag sampling


Implements the Zig-Zag algorithm (Bierkens, Fearnhead, Roberts, 2016) [https://arxiv.org/abs/1607.03188] and Bouncy Particle Sampler as applied to Bayesian logistic regression, Gaussian target distributions and Student-t distributions.

## Installation

Install R packages `Rcpp` and `RcppEigen`:

```r
install.packages("Rcpp")
install.packages("RcppEigen")
```

Then either
1. download one of the .tar.gz files in the RZigZag repository and install `RZigZag` from the command line using
```
R CMD INSTALL RZigZag_[version].tar.gz
```
or 
2. (recommended) install from R using
```r
install.packages("RZigZag")
```

## Further documentation
```r
help(RZigZag)
```
