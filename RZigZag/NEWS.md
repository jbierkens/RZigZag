* TODO: update help
* TODO: allow for user provided starting direction
* TODO: R helper function to 
  - extract samples from derived skeleton
  - estimate moments from skeleton
  - perform batch means on skeleton
* TODO: flexibility with control variates (see correspondence with Divakar Kumar)
* TODO: general implementation for user provided gradient + Lipschitz bound
* TODO: multivariate T distribution
* TODO: add positive excess switching rate
* TODO: rethink memory usage

# RZigZag 0.1.7
* transposed dataX matrix in logistic regression: now the different rows of dataX represent different observations 
* allow user to provide starting direction and custom reference points for control variates (ZigZagLogistic)
* allow for V (Gaussian case) to be column vector containing diagonal of precision matrix

# RZigZag 0.1.6
* Allows for fixed time horizon (in additional to fixed number of iterations)
* Preprocessing of input to logistic regression disabled. Adding intercept to design matrix and possible recentering is up to user.

# RZigZag 0.1.5
* Implements Bouncy Particle Sampler for Gaussian target
* Allows to specify starting value for ZigZagGaussian

# RZigZag 0.1.4
* Changed from n_epochs argument to n_iter, representing number of iterations, as this is more flexible. This update is not downwards compatible, but if you want to run ZigZagLogistic when subsampling (with or without control variates) with n_epochs gradient evaluations, simply pick n_iter = n_epochs * (number of observations).
* Resolved ambiguities in overloaded functions which previously resulted in a compile error on Solaris.

# RZigZag 0.1.3
* Corrected arXiv reference

# RZigZag 0.1.1
* This is the first version, which introduces the R functions `ZigZagLogistic` and `ZigZagGaussian`.
