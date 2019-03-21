// RZigZag.cpp : implements Zig-Zag and other PDMP samplers
//
// Copyright (C) 2017--2019 Joris Bierkens
//
// This file is part of RZigZag.
//
// RZigZag is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// RZigZag is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with RZigZag.  If not, see <http://www.gnu.org/licenses/>.

#include "RZigZag.h"

List SkeletonToList(const Skeleton& skel) {
  // output: R list consisting of Times, Positions and Velocities
  return List::create(Named("Times") = skel.getTimes(), Named("Positions") = skel.getPositions(), Named("Velocities") = skel.getVelocities());
}

Skeleton ListToSkeleton(const List& list) {
  return Skeleton(list["Times"], list["Positions"], list["Velocities"]);
}

//' ZigZagGaussian
//' 
//' Applies the Zig-Zag Sampler to a Gaussian target distribution, as detailed in Bierkens, Fearnhead, Roberts, The Zig-Zag Process and Super-Efficient Sampling for Bayesian Analysis of Big Data, 2016.
//' Assume potential of the form \deqn{U(x) = (x - mu)^T V (x - mu)/2,} i.e. a Gaussian with mean vector \code{mu} and covariance matrix \code{inv(V)}
//'
//' @param V the inverse covariance matrix (or precision matrix) of the Gaussian target distribution; if V is a matrix consisting of a single column, it is interpreted as the diagonal of the precision matrix.
//' @param mu mean of the Gaussian target distribution
//' @param n_iterations Number of algorithm iterations; will result in the equivalent amount of skeleton points in Gaussian case because no rejections are needed.
//' @param finalTime If provided and nonnegative, run the sampler until a trajectory of continuous time length finalTime is obtained (ignoring the value of \code{n_iterations})
//' @param x0 starting point (optional, if not specified taken to be the origin)
//' @param v0 starting direction (optional, if not specified taken to be +1 in every component)
//' @return Returns a list with the following objects:
//' @return \code{Times}: Vector of switching times
//' @return \code{Positions}: Matrix whose columns are locations of switches. The number of columns is identical to the length of \code{skeletonTimes}. Be aware that the skeleton points themselves are NOT samples from the target distribution.
//' @return \code{Velocities}: Matrix whose columns are velocities just after switches. The number of columns is identical to the length of \code{skeletonTimes}.
//' @examples
//' V <- matrix(c(3,1,1,3),nrow=2)
//' mu <- c(2,2)
//' result <- ZigZagGaussian(V, mu, 100)
//' plot(result$Positions[1,], result$Positions[2,],type='l',asp=1)
//' 
//' V <- matrix(rep(1,100),nrow=100) # this will be interpreted as a diagonal matrix
//' mu <- numeric(100)
//' result <- ZigZagGaussian(V, mu, 1000)
//' plot(result$Positions[1,], result$Positions[2,],type='l',asp=1)
//' @export
// [[Rcpp::export]]
List ZigZagGaussian(const Eigen::MatrixXd V, const Eigen::VectorXd mu, int n_iter = -1, double finalTime = -1, const NumericVector x0 = NumericVector(0), const NumericVector v0 = NumericVector(0)) {
  if (finalTime >= 0)
    n_iter = -1;
  else if (n_iter >= 0)
    finalTime = -1;
  else
    stop("Either finalTime or n_iter must be specified.");
  
  const int dim = V.rows();
  VectorXd x, v;
  if (x0.size() < dim)
    x = VectorXd::Zero(dim);
  else
    x = as<Eigen::Map<VectorXd> >(x0);
  if (v0.size() < dim)
    v = VectorXd::Ones(dim);
  else
    v = as<Eigen::Map<VectorXd> >(v0);
  
  GaussianZZ sampler(V, mu, x, v);
  Skeleton skel(ZigZag(sampler, n_iter, finalTime));
  return SkeletonToList(skel);
}

//' ZigZagLogistic
//'
//' Applies the Zig-Zag Sampler to logistic regression, as detailed in Bierkens, Fearnhead, Roberts, The Zig-Zag Process and Super-Efficient Sampling for Bayesian Analysis of Big Data, 2019.
//'
//' @param dataX Design matrix containing observations of the independent variables x. The i-th row represents the i-th observation with components x_{i,1}, ..., x_{i,d}.
//' @param dataY Vector of length n containing {0, 1}-valued observations of the dependent variable y.
//' @param n_iterations Number of algorithm iterations; will result in the equivalent amount of skeleton points in Gaussian case because no rejections are needed.
//' @param finalTime If provided and nonnegative, run the sampler until a trajectory of continuous time length finalTime is obtained (ignoring the value of \code{n_iterations})
//' @param x0 starting point (optional, if not specified taken to be the origin)
//' @param v0 starting direction (optional, if not specified taken to be +1 in every component)
//' @param cv optional boolean to indicate the use of subsampling with control variates
//' @return Returns a list with the following objects:
//' @return \code{Times}: Vector of switching times
//' @return \code{Positions}: Matrix whose columns are locations of switches. The number of columns is identical to the length of \code{skeletonTimes}. Be aware that the skeleton points themselves are NOT samples from the target distribution.
//' @return \code{Velocities}: Matrix whose columns are velocities just after switches. The number of columns is identical to the length of \code{skeletonTimes}.
//' @examples
//' require("RZigZag")
//'
//' generate.logistic.data <- function(beta, n.obs) {
//'   dim <- length(beta)
//'   dataX <- cbind(rep(1.0,n.obs), matrix(rnorm((dim -1) * n.obs), ncol = dim -1));
//'   vals <- dataX %*% as.vector(beta)
//'     generateY <- function(p) { rbinom(1, 1, p)}
//'   dataY <- sapply(1/(1 + exp(-vals)), generateY)
//'     return(list(dataX = dataX, dataY = dataY))
//' }
//'
//' beta <- c(1,2)
//' data <- generate.logistic.data(beta, 1000)
//' result <- ZigZagLogistic(data$dataX, data$dataY, 1000)
//' plot(result$Positions[1,], result$Positions[2,],type='l',asp=1)
//' @export
// [[Rcpp::export]]
List ZigZagLogistic(const Eigen::MatrixXd& dataX, const Eigen::VectorXi& dataY, int n_iter = -1, double finalTime = -1, const NumericVector x0 = NumericVector(0), const NumericVector v0 = NumericVector(0), bool cv = false) {

  if (finalTime >= 0)
    n_iter = -1;
  else if (n_iter >= 0)
    finalTime = -1;
  else
    stop("Either finalTime or n_iter must be specified.");
  
  const int dim = dataX.cols();
  VectorXd x, v;
  if (x0.size() < dim)
    x = VectorXd::Zero(dim);
  else
    x = as<Eigen::Map<VectorXd> >(x0);
  if (v0.size() < dim)
    v = VectorXd::Ones(dim);
  else
    v = as<Eigen::Map<VectorXd> >(v0);
  
  if (cv) {
    LogisticCVZZ sampler(dataX, dataY, x, v);
    Skeleton skel(ZigZag(sampler, n_iter, finalTime));
    return SkeletonToList(skel);
  }
  else {
    LogisticZZ sampler(dataX, dataY, x, v);
    Skeleton skel(ZigZag(sampler, n_iter, finalTime));
    return SkeletonToList(skel);
  }
}

//' EstimateESS
//' 
//' Estimates the effective sample size (ESS) of a piecewise deterministic skeleton
//' 
//' @param skeletonList a piecewise deterministic skeleton (consisting of Times, Points and Velocities) returned by a sampler
//' @param n_batches optional argument indicating the number of batches to use in the batch means estimation method
//' @param coordinate if specified, only estimate the ESS of the specified coordinate, otherwise estimate the ESS of all coordinates
//' @return Returns a vector containing the estimated asymptotic variance and ESS
//' @export
// [[Rcpp::export]]
List EstimateESS(const List& skeletonList, int n_batches = 100, int coordinate = -1) {
  
  Skeleton skel = ListToSkeleton(skeletonList);
  if (coordinate > 0)
    coordinate--; // convert R to C++ number
  VectorXd asvar(skel.estimateAsymptoticVariance(n_batches, coordinate));
  VectorXd ESS(skel.estimateESS(n_batches, coordinate, asvar));
  return List::create(Named("AsVar") = asvar, Named("ESS") = ESS);
}


// 
// void Skeleton::GaussianZZ(const MatrixXd& V, const VectorXd& mu, const int n_iter, const double finalTime, const VectorXd& x0, const VectorXd& v0) {
//   // Gaussian skeleton
//   // input: V precision matrix (inverse covariance), mu mean, x0 initial condition, n_iter number of switches
//   // invariant: w = V theta, z = V (x-mu)
//   
//   GaussianData data(&V, mu);
//   GaussianBound bound(data.getGaussianBoundObject(x0, v0));
//   ZigZag(data, bound, x0, v0, n_iter, finalTime, true); // rejectionFree = true
// }
// 
// VectorXd resampleVelocity(const int dim, const bool unit_velocity = true) {
//   // helper function for GaussianBPS
//   VectorXd v = as<Eigen::Map<VectorXd> >(rnorm(dim));
//   if (unit_velocity)
//     v.normalize();
//   return v;
// }
// 
// void Skeleton::GaussianBPS(const MatrixXd& V, const VectorXd& mu, const int n_iter, const double finalTime, const VectorXd& x0, const double refresh_rate, const bool unit_velocity) {
//   
//   // Gaussian skeleton using BPS
//   // input: V precision matrix (inverse covariance), mu mean, x0 initial condition, n_iter number of switches
//   
//   if (refresh_rate < 0)
//     stop("GaussianBPS error: refresh_rate should be non-negative.");
// 
//   const int dim = V.rows();
//   if (x0.size() != dim)
//     stop("GaussianBPS error: dimension of starting position x0 should agree with dimensions of precision matrix V.");
//   
//   bool productForm = false;
//   ArrayXd diagV;
//   if (V.cols() != V.rows())
//   {
//     productForm = true;
//     diagV = V.array();
//   }
// 
//   VectorXd x(x0);
//   VectorXd v = resampleVelocity(dim, unit_velocity);
// 
//   RNGScope scp; // initialize random number generator
//   
//   double t, t_reflect, t_refresh;
//   double currentTime = 0;
//   
//   VectorXd gradient, w;
//   if (productForm) {
//     gradient = diagV * (x - mu).array();
//     w = diagV * v.array(); // useful invariant
//   }
//   else {
//     gradient = V * (x - mu);
//     w = V * v; // useful invariant
//   }
//   double a = v.dot(gradient);
//   double b = v.dot(w);
//   
//   int iteration = 0;
//   
//   Initialize(dim, n_iter);
//   Push(currentTime, x, v);
// 
//   while (iteration < n_iter || currentTime < finalTime) {
// //    Rprintf("iteration: %d, n_iter: %d, currentTime: %g, finalTime: %g\n", iteration, n_iter, currentTime, finalTime);
//     ++iteration;
//     NumericVector U(runif(2));
//     t_reflect = getRandomTime(a, b, U(0));
//     if (refresh_rate <= 0) {
//       t_refresh = -1; // indicating refresh rate = infinity
//       t = t_reflect;
//     }
//     else {
//       t_refresh = -log(U(1))/refresh_rate;
//       t = (t_reflect < t_refresh ? t_reflect : t_refresh);
//     }
//     currentTime = currentTime + t;
//     x = x + t * v;
//     gradient = gradient + t * w;
// 
//     if (t_refresh < 0 || t_reflect < t_refresh) {
//       VectorXd normalized_gradient = gradient.normalized(); // for projection
//       VectorXd delta_v = - 2 * (v.dot(normalized_gradient)) * normalized_gradient;
//       v = v + delta_v;
//     }
//     else
//       v = resampleVelocity(dim, unit_velocity);
// //    Rprintf("%g\n", v.norm());
//     if (productForm)
//       w = diagV * v.array();
//     else
//       w = V * v; // preserves invariant for w
//     a = v.dot(gradient);
//     b = v.dot(w);
//     Push(currentTime, x, v, finalTime);
//   }
//   ShrinkToCurrentSize();
// }
// 

// 
// 
// 
// //' BPSGaussian
// //' 
// //' Applies the BPS Sampler to a Gaussian target distribution, as detailed in Bouchard-Côté et al, 2017.
// //' Assume potential of the form \deqn{U(x) = (x - mu)^T V (x - mu)/2,} i.e. a Gaussian with mean vector \code{mu} and covariance matrix \code{inv(V)}
// //'
// //' @param V the inverse covariance matrix (or precision matrix) of the Gaussian target distribution; if V is a matrix consisting of a single column, it is interpreted as the diagonal of the precision matrix.
// //' @param mu mean of the Gaussian target distribution
// //' @param n_iterations Number of algorithm iterations; will result in the equivalent amount of skeleton points in Gaussian case because no rejections are needed.
// //' @param x0 starting point
// //' @param finalTime If provided and nonnegative, run the BPS sampler until a trajectory of continuous time length finalTime is obtained (ignoring the value of \code{n_iterations})
// //' @param refresh_rate \code{lambda_refresh}
// //' @param unit_velocity TRUE indicates velocities uniform on unit sphere, FALSE indicates standard normal velocities
// //' @param n_samples Number of discrete time samples to extract from the skeleton.
// //' @param n_batches If non-zero, estimate effective sample size through the batch means method, with n_batches number of batches.
// //' @param computeCovariance Boolean indicating whether to estimate the covariance matrix.
// //' @return Returns a list with the following objects:
// //' @return \code{skeletonTimes}: Vector of switching times
// //' @return \code{skeletonPoints}: Matrix whose columns are locations of switches. The number of columns is identical to the length of \code{skeletonTimes}. Be aware that the skeleton points themselves are NOT samples from the target distribution.
// //' @return \code{skeletonDirections}: Matrix whose columns are directions just after switches. The number of columns is identical to the length of \code{skeletonTimes}.
// //' @return \code{samples}: If \code{n_samples > 0}, this is a matrix whose \code{n_samples} columns are samples along the Zig-Zag trajectory.
// //' @return \code{mode}: Not used for a Gaussian target.
// //' @return \code{batchMeans}: If \code{n_batches > 0}, this is a matrix whose \code{n_batches} columns are the batch means
// //' @return \code{means}: If \code{n_batches > 0}, this is a vector containing the means of each coordinate along the Zig-Zag trajectory 
// //' @return \code{covariance}: If \code{n_batches > 0} or \code{computeCovariance = TRUE}, this is a matrix containing the sample covariance matrix along the trajectory
// //' @return \code{asVarEst}: If \code{n_batches > 0} this is an estimate of the asymptotic variance along each component
// //' @return \code{ESS}: If \code{n_batches > 0} this is an estimate of the effective sample size along each component
// //' @examples
// //' V <- matrix(c(3,1,1,3),nrow=2)
// //' mu <- c(2,2)
// //' x0 <- c(0,0)
// //' result <- BPSGaussian(V, mu, 100, x0, n_samples = 10)
// //' plot(result$skeletonPoints[1,], result$skeletonPoints[2,],type='l',asp=1)
// //' points(result$samples[1,], result$samples[2,], col='magenta')
// //' V <- matrix(rep(1,100),nrow=100)
// //' mu <- numeric(100)
// //' x0 <- numeric(100)
// //' result <- BPSGaussian(V, mu, 1000, x0, n_samples = 10)
// //' plot(result$skeletonPoints[1,], result$skeletonPoints[2,],type='l',asp=1)
// //' points(result$samples[1,], result$samples[2,], col='magenta')
// //' @export
// // [[Rcpp::export]]
// List BPSGaussian(const Eigen::MatrixXd V, const Eigen::VectorXd mu, int n_iterations, const Eigen::VectorXd x0, const double finalTime = -1.0, const double refresh_rate = 1, const bool unit_velocity = true, const int n_samples=0, const int n_batches=0, bool computeCovariance=false) {
// 
//   Skeleton skeleton;
//   if (finalTime >= 0)
//     n_iterations = -1;
//   skeleton.GaussianBPS(V, mu, n_iterations, finalTime, x0, refresh_rate, unit_velocity);
//   if (n_samples > 0)
//     skeleton.sample(n_samples);
//   if (n_batches > 0)
//     skeleton.computeBatchMeans(n_batches);
//   if (computeCovariance)
//     skeleton.computeCovariance();
//   return skeleton.toR();
// }
// 
