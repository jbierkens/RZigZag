# RZigZag: Zig-Zag sampling


Implements the Zig-Zag algorithm with subsampling and control variates (ZZ-CV) of (Bierkens, Fearnhead, Roberts, 2016) [https://arxiv.org/abs/1607.03188] as applied to Bayesian logistic regression, as well as basic Zig-Zag for a Gaussian target distribution.

The Zig-Zag algorithm is an MCMC algorithm which allows for exact subsampling and can therefore be very efficient in situations with large amounts of data.

## Installation

Install R packages `Rcpp` and `RcppEigen`:

```r
install.packages("Rcpp")
install.packages("RcppEigen")
```

Then install `RZigZag` from the command line using
```
R CMD INSTALL RZigZag_0.1.1.tar.gz
```
or from R using
```r
install.packages("RZigZag")
```

## Further documentation
```r
help(RZigZag)
help(ZigZagLogistic)
help(ZigZagGaussian)
```

## Examples

### Zig-Zag for logistic regression

```r
require("RZigZag")
generate.logistic.data <- function(beta, nobs) {
  ncomp <- length(beta)
  dataX <- matrix(rnorm((ncomp -1) * nobs), nrow = ncomp -1);
  vals <- beta[1] + colSums(dataX * as.vector(beta[2:ncomp]))
  generateY <- function(p) { rbinom(1, 1, p)}
  dataY <- sapply(1/(1 + exp(-vals)), generateY)
  return(list(dataX, dataY))
}

beta <- c(1,2)
data <- generate.logistic.data(beta, 1000)
result <- ZigZagLogistic(data[[1]], data[[2]], 1000, n_samples = 100)
plot(result$skeletonPoints[1,], result$skeletonPoints[2,],type='l',asp=1)
points(result$samples[1,], result$samples[2,], col='magenta')
```

### Zig-Zag for a Gaussian target distribution

```r
require("RZigZag")
V <- matrix(c(3,1,1,3),nrow=2,ncol=2)
mu <- c(2,2)
result <- ZigZagGaussian(V, mu, 100, n_samples = 10)
plot(result$skeletonPoints[1,], result$skeletonPoints[2,],type='l',asp=1)
points(result$samples[1,], result$samples[2,], col='magenta')
