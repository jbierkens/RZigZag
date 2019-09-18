# RZigZag: Zig-Zag sampling


Implements the Zig-Zag algorithm with subsampling and control variates (ZZ-CV) of (Bierkens, Fearnhead, Roberts, 2016) [https://arxiv.org/abs/1607.03188] as applied to Bayesian logistic regression, as well as basic Zig-Zag for a Gaussian target distribution.

The Zig-Zag algorithm is an MCMC algorithm which allows for exact subsampling and can therefore be very efficient in situations with large amounts of data.

## Installation

Install R packages `Rcpp` and `RcppEigen`:

```r
install.packages("Rcpp")
install.packages("RcppEigen")
```

Then either
(a) download one of the .tar.gz files in the RZigZag repository and install `RZigZag` from the command line using
```
R CMD INSTALL RZigZag_[version].tar.gz
```
or 
(b) (recommended) install from R using
```r
install.packages("RZigZag")
```

## Further documentation
```r
help(RZigZag)
```
