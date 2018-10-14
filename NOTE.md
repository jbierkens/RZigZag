# Note on R workflow
Because I am forgetting this, here is the workflow from a clean git clone into the development branch

1) from R, run
> Rcpp::compileAttributes("RZigZag")

> roxygen2::roxygenize("RZigZag")
2) then on the command line
> R CMD INSTALL RZigZag

## submitting to CRAN
* to build package zipfile for submission
> R CMD build
* to check the zip file
> R CMD check --as-cran ZIPFILE
