// RZigZag.cpp : implements Zig-Zag and other PDMP samplers
//
// Copyright (C) 2017--2018 Joris Bierkens
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

// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

using namespace Rcpp;
 
using Eigen::ArrayXd;
using Eigen::MatrixXd;
using Eigen::VectorXd;

#include "RZigZag.h"

#include "LogisticData.h"

inline double pospart(const double a) {
  if (a > 0)
    return a;
  else
    return 0;
}

void Skeleton::Initialize(const int dim, int initialSize) {
  if (initialSize >= 0)
    initialSize++; // allow for one extra point (the starting point)
  if (initialSize < 0) // i.e. no valid initialSize is provided
    initialSize = DEFAULTSIZE;
  Points = MatrixXd(dim, initialSize);
  Directions = MatrixXd(dim,initialSize);
  dimension = dim;
  Times = VectorXd(initialSize);
  capacity = initialSize;
  currentSize = 0;
}

void Skeleton::Resize(const int factor) {
  capacity *= factor;
  //        Rprintf("Resizing to size %d...\n", max_switches);
  Times.conservativeResize(capacity);
  Points.conservativeResize(dimension, capacity);
  Directions.conservativeResize(dimension, capacity);
}

void Skeleton::Push(const double time, const VectorXd& point, const VectorXd& direction, const double finalTime) {
  if (currentSize >= capacity)
    Resize();
  Directions.col(currentSize) = direction;
  if (finalTime < 0 || time < finalTime) {
    Times[currentSize] = time;
    Points.col(currentSize) = point;
  }
  else {
    Times[currentSize] = finalTime;
    double previousTime = Times[currentSize-1];
    VectorXd previousPoint = Points.col(currentSize-1);
    Points.col(currentSize) = previousPoint + (finalTime - previousTime) * (point - previousPoint) / (time - previousTime);
  }
  currentSize++;
}
  
void Skeleton::ShrinkToCurrentSize() {
  Times.conservativeResize(currentSize);
  Points.conservativeResize(dimension, currentSize);
  Directions.conservativeResize(dimension, currentSize);
}

void Skeleton::computeBatchMeans(const int n_batches) {
  if (n_batches == 0)
    stop("n_batches should be positive.");
  const int n_points = Times.size();
  const int dim = Points.rows();
  const double t_max = Times[n_points-1];
  const double batch_length = t_max / n_batches;
  
  double t0 = Times[0];
  VectorXd x0 = Points.col(0);
  
  batchMeans = MatrixXd(dim, n_batches);
  
  int batchNr = 0;
  double t_intermediate = batch_length;
  VectorXd currentBatchMean = VectorXd::Zero(dim);
  
  for (int i = 1; i < n_points; ++i) {
    double t1 = Times[i];
    VectorXd x1 = Points.col(i);
    
    while (batchNr < n_batches - 1 && t1 > t_intermediate) {
      VectorXd x_intermediate = x0 + (t_intermediate - t0) / (t1 - t0) * (x1 - x0);
      batchMeans.col(batchNr) = currentBatchMean + (t_intermediate - t0) * (x_intermediate + x0)/(2 * batch_length);
      
      // initialize next batch
      currentBatchMean = VectorXd::Zero(dim);
      batchNr++;
      t0 = t_intermediate;
      x0 = x_intermediate;
      t_intermediate = batch_length * (batchNr + 1);
    }
    currentBatchMean += (t1 - t0) * (x1 + x0)/(2 * batch_length);
    t0 = t1;
    x0 = x1;
  }
  batchMeans.col(batchNr) = currentBatchMean;
  
  computeCovariance();
  
  MatrixXd meanZeroBatchMeans = batchMeans.colwise() - means;
  asVarEst = batch_length * meanZeroBatchMeans.rowwise().squaredNorm()/(n_batches - 1);
  ESS = (covarianceMatrix.diagonal().array()/asVarEst.array() * t_max).matrix();
}

void Skeleton::computeCovariance() {
  const int n_points = Times.size();
  const int dim = Points.rows();
  const double t_max = Times[n_points-1];
  
  double t0 = Times[0];
  VectorXd x0 = Points.col(0);
  MatrixXd cov_current = x0 * x0.transpose();
  
  covarianceMatrix = MatrixXd::Zero(dim, dim);
  means = VectorXd::Zero(dim);
  
  for (int i = 1; i < n_points; ++i) {
    double t1 = Times[i];
    VectorXd x1 = Points.col(i);
    // the following expression equals \int_{t_0}^{t_1} x(t) (x(t))^T d t
    covarianceMatrix += (t1 - t0) * (2 * x0 * x0.transpose() + x0 * x1.transpose() + x1 * x0.transpose() + 2 * x1 * x1.transpose())/(6 * t_max);
    means += (t1 - t0) * (x1 + x0) /(2 * t_max);
    t0 = t1;
    x0 = x1;
  }
  covarianceMatrix -= means * means.transpose();
}

void Skeleton::ZZStepAffineBound(VectorXd& a, const VectorXd& b, const DataObject& data, double& currentTime, VectorXd& position, VectorXd& direction, const double intendedFinalTime) {
  
  const int dim = position.size(); 

  NumericVector U(runif(dim));
  int i0 = -1;
  double simulatedTime, minTime;

  for (int i = 0; i < dim; ++i) {
    simulatedTime = getRandomTime(a(i), b(i), U(i));
    if (simulatedTime > 0 && (i0 == -1 || simulatedTime < minTime)) {
      i0 = i;
      minTime = simulatedTime;
    }
  }
  if (minTime < 0) {
    stop("ZZStepAffineBound: Zigzag wandered off to infinity.");
  }
  else {
    currentTime = currentTime + minTime;
    position = position + minTime * direction;
    a = a + b * minTime;
    double derivative = data.getDerivative(position, i0);
    double V = runif(1)(0);
    if (V <= direction(i0) * derivative/a(i0)) {
      direction(i0) = -direction(i0);
      Push(currentTime, position, direction, intendedFinalTime);
    }
    a(i0) = derivative * direction(i0);
  }
}


void Skeleton::ZZStepConstantBound(const VectorXd& a, const DataObject& data, double& currentTime, VectorXd& position, VectorXd& direction, const double intendedFinalTime) {
  
  const int dim = position.size(); 
  
  NumericVector U(runif(dim));
  int i0 = -1;
  double simulatedTime, minTime;
  
  for (int i = 0; i < dim; ++i) {
    simulatedTime = getRandomTime(a(i), 0, U(i));
    if (simulatedTime > 0 && (i0 == -1 || simulatedTime < minTime)) {
      i0 = i;
      minTime = simulatedTime;
    }
  }
  if (minTime < 0) {
    stop("ZZStepConstantBound: Zigzag wandered off to infinity.");
  }
  else {
    currentTime = currentTime + minTime;
    position = position + minTime * direction;
    double derivative = data.getDerivative(position, i0);
    double V = runif(1)(0);
    if (V <= direction(i0) * derivative/a(i0)) {
      direction(i0) = -direction(i0);
      Push(currentTime, position, direction, intendedFinalTime);
    }
  }
}


void Skeleton::LogisticBasicZZ(const MatrixXd& dataX, const VectorXi& dataY, const int n_iter, const double finalTime, const VectorXd& x0) {
  
  const LogisticData logisticData(dataX, dataY);
  const int dim = dataX.rows();
  
  MatrixXd Q(domHessianLogistic(dataX));
  
  if (x0.size() != dim)
    stop("LogisticBasicZZ error: dimension of starting position should be equal to number of covariates in data.");

  VectorXd x(x0);
  VectorXd theta(VectorXd::Constant(dim, 1)); // initialize theta at (+1,...,+1)
  
  const VectorXd b(sqrt((double)dim) * Q.rowwise().norm());
  VectorXd a(dim);
  for (int k = 0; k < dim; ++k) 
    a(k) = theta(k) * logisticData.getDerivative(x, k);
  
  
  RNGScope scp; // initialize random number generator
  
  double minTime, simulatedTime;
  double currentTime = 0;
  int iteration = 0;
  
  Initialize(dim, n_iter);
  Push(currentTime, x, theta);
  
  while (currentTime < finalTime || iteration < n_iter) {
    ++iteration;
    ZZStepAffineBound(a, b, logisticData, currentTime, x, theta, finalTime);
  }
  ShrinkToCurrentSize();
  Rprintf("LogisticBasicZZ: Fraction of accepted switches: %g\n", double(currentSize)/(iteration));
}

void Skeleton::LogisticUpperboundZZ(const MatrixXd& dataX, const VectorXi& dataY, const int n_iter, const double finalTime, const VectorXd& x0) {
  
  const LogisticData logisticData(dataX, dataY);

  const int dim = dataX.rows();
  if (x0.size() != dim)
    stop("LogisticUpperboundZZ error: dimension of starting position should be equal to number of covariates in data.");
  
  const int n_observations = dataX.cols();
  VectorXd theta(VectorXd::Constant(dim, 1)); // initialize theta at (+1,...,+1)
  double currentTime = 0;
  VectorXd x(x0);
  const VectorXd a(n_observations * logisticUpperbound(dataX));
  
  RNGScope scp; // initialize random number generator
  
  double minTime, simulatedTime;
  
  int i0;
  Initialize(dim, n_iter);
  Push(currentTime, x, theta);

  int iteration = 0;
  while (iteration < n_iter || currentTime < finalTime) {
    ++iteration;
    ZZStepConstantBound(a, logisticData, currentTime, x, theta, finalTime);
  }
  ShrinkToCurrentSize();
  Rprintf("LogisticUpperboundZZ: Fraction of accepted switches: %g\n", double(currentSize)/(iteration));
}

void Skeleton::LogisticSubsamplingZZ(const MatrixXd& dataX, const VectorXi& dataY, const int n_iter, const double finalTime, const VectorXd& x0) {
  
  const LogisticData logisticData(dataX, dataY);
  
  const int dim = dataX.rows();
  const int n_observations = dataX.cols();
  
  VectorXd theta(VectorXd::Constant(dim,1)); // initialize theta at (+1,...,+1)
  double currentTime = 0;
  if (x0.size() != dim)
    stop("LogisticSubsamplingZZ error: dimension of starting position should be equal to number of covariates in data.");
  VectorXd x(x0);
  VectorXd upperbound(n_observations * logisticUpperbound(dataX));
  
  RNGScope scp; // initialize random number generator
  
  double minTime, simulatedTime;
  Initialize(dim,n_iter);
  Push(currentTime, x, theta);

  int i0;
  int switches = 0;
  int iteration = 0;
  while (iteration < n_iter || currentTime < finalTime) {
    ++iteration;
    for (int i = 0; i < dim; ++i) {
      simulatedTime = rexp(1, upperbound(i))(0);
      if (i == 0 || simulatedTime < minTime) {
        i0 = i;
        minTime = simulatedTime;
      }
    }
    currentTime = currentTime + minTime;
    x = x + minTime * theta;
    double derivative = logisticData.subsampledDerivative(x, i0);
    double V = runif(1)(0);
    if (derivative > upperbound(i0)) {
      Rprintf("LogisticSubsamplingZZ:: Error: derivative larger than its supposed upper bound.\n");
      Rprintf("  Upper bound: %g, actual derivative: %g.\n", upperbound(i0), derivative);
      Rprintf("  Index: %d, X(0): %g, X(1): %g\n", i0, x(0), x(1));
      break;
    }
    if (V <= theta(i0) * derivative/upperbound(i0)) {
      theta(i0) = -theta(i0);
      ++switches;
      Push(currentTime, x, theta);
    }
  }
  ShrinkToCurrentSize();
  Rprintf("LogisticSubSampling: Fraction of accepted switches: %g\n", double(switches)/iteration);
}

void Skeleton::LogisticCVZZ(const MatrixXd& dataX, const VectorXi& dataY, const int n_iter, const double finalTime, const VectorXd& x0) {
  
  LogisticData data(dataX, dataY);
  const int dim = dataX.rows();
  const int n_observations = dataX.cols();
  const double precision = 1e-10;
  const int max_iter = 1e2;
  mode = VectorXd::Zero(dim);
  newtonLogistic(data, mode, precision, max_iter);
  VectorXd x(x0);
  if (x.rows()==0)
    x = mode;
  
  VectorXd theta(VectorXd::Constant(dim,1)); // initialize theta at (+1,...,+1)
  double currentTime = 0;
  const VectorXd uniformBound(cvBound(dataX) * n_observations);
  const VectorXd b(sqrt((double)dim) * uniformBound);
  VectorXd a((x-mode).norm() * uniformBound);
  
  RNGScope scp; // initialize random number generator
  
  double minTime, simulatedTime;
  Initialize(dim, n_iter);
  Push(currentTime, x, theta);

  int i0;
  int switches = 0;
  int iteration = 0;
  while (iteration < n_iter || currentTime < finalTime) {
    ++iteration;
    NumericVector U(runif(dim));
    i0 = -1;
    for (int i = 0; i < dim; ++i) {
      simulatedTime = getRandomTime(a(i), b(i), U(i));
      if (simulatedTime > 0 && (i0 == -1 || simulatedTime < minTime)) {
        i0 = i;
        minTime = simulatedTime;
      }
    }
    if (minTime < 0) {
      // this means that we simulated T = inf
      stop("Zig zag wandered off to infinity.");
    }
    else {
      currentTime = currentTime + minTime;
      x = x + minTime * theta;
      // TODO implement subsampling with control variates using logisticData object
      int J = floor(n_observations*runif(1)(0)); // randomly select observation
      double switch_rate = theta(i0) * n_observations * dataX(i0,J) * (1.0/(1.0+exp(-dataX.col(J).dot(x)))-1.0/(1.0+exp(-dataX.col(J).dot(mode))));
      double simulated_rate = a(i0) + b(i0) * minTime;
      if (switch_rate > simulated_rate) {
        stop("LogisticCVZZ:: Error: switch rate larger than its supposed upper bound.");
        break;
      }
      double V = runif(1)(0);
      if (V <= switch_rate/simulated_rate) {
        theta(i0)=-theta(i0);
        ++switches;
        Push(currentTime, x, theta, finalTime);
      }
      a = (x-mode).norm() * uniformBound;
    }
  }
  ShrinkToCurrentSize();
  Rprintf("LogisticCVZZ: Fraction of accepted switches: %g\n", double(switches)/iteration);
}

void Skeleton::GaussianZZ(const MatrixXd& V, const VectorXd& mu, const int n_iter, const double finalTime, const VectorXd& x0) {
  // Gaussian skeleton
  // input: V precision matrix (inverse covariance), mu mean, x0 initial condition, n_iter number of switches
  // invariant: w = V theta, z = V (x-mu)

  const int dim = V.rows();
  bool productForm = false;
  ArrayXd diagV;
  if (V.cols() != V.rows())
  {
    productForm = true;
    diagV = V.array();
  }

  if (x0.size() != dim)
    stop("GaussianZZ error: dimension of starting position x0 should agree with dimensions of precision matrix V.");
  
  VectorXd x(x0);
  VectorXd theta = VectorXd::Constant(dim, 1); // initialize theta at (+1,...,+1)
  ArrayXd w, z;
  if (productForm) {
    w = diagV * theta.array();
    z = diagV * (x-mu).array();
  }
  else {
    w = (V * theta).array();
    z = (V * (x - mu)).array();
  }
  ArrayXd a(theta.array() * z), b(theta.array() * w); // convert to array for pointwise multiplication

  RNGScope scp; // initialize random number generator

  double minTime, simulatedTime;
  int i0;
  double currentTime = 0;
  int iteration = 0;

  Initialize(dim, n_iter);
  Push(currentTime, x, theta);
  
  while (iteration < n_iter || currentTime < finalTime) {
    ++iteration;
    NumericVector U(runif(dim));
    i0 = -1;
    for (int i = 0; i < dim; ++i) {
      simulatedTime = getRandomTime(a(i), b(i), U(i));
      if (simulatedTime > 0 && (i0 == -1 || simulatedTime < minTime)) {
        i0 = i;
        minTime = simulatedTime;
      }
    }
    if (minTime < 0) {
      // this means that we simulated T = inf
      stop("Zig zag wandered off to infinity.");
    }
    else {
      currentTime = currentTime + minTime;
      x = x + minTime * theta;
      theta(i0) = -theta(i0);
      z = z + w * minTime; // preserve invariant  z = V (x-mu)
      if (productForm)
        w(i0) = w[i0] + 2 * theta(i0) * V(i0);
      else
        w = w + 2 * theta(i0) * V.col(i0).array(); // preserve invariant w = V theta
      a = theta.array() * z;
      b = theta.array() * w;
      Push(currentTime, x, theta, finalTime);
    }
  }
  ShrinkToCurrentSize();
}

VectorXd resampleVelocity(const int dim, const bool unit_velocity = true) {
  // helper function for GaussianBPS
  VectorXd v = as<Eigen::Map<VectorXd> >(rnorm(dim));
  if (unit_velocity)
    v.normalize();
  return v;
}

void Skeleton::GaussianBPS(const MatrixXd& V, const VectorXd& mu, const int n_iter, const double finalTime, const VectorXd& x0, const double refresh_rate, const bool unit_velocity) {
  
  // Gaussian skeleton using BPS
  // input: V precision matrix (inverse covariance), mu mean, x0 initial condition, n_iter number of switches
  
  if (refresh_rate < 0)
    stop("GaussianBPS error: refresh_rate should be non-negative.");

  const int dim = V.rows();
  if (x0.size() != dim)
    stop("GaussianBPS error: dimension of starting position x0 should agree with dimensions of precision matrix V.");
  
  bool productForm = false;
  ArrayXd diagV;
  if (V.cols() != V.rows())
  {
    productForm = true;
    diagV = V.array();
  }

  VectorXd x(x0);
  VectorXd v = resampleVelocity(dim, unit_velocity);

  RNGScope scp; // initialize random number generator
  
  double t, t_reflect, t_refresh;
  double currentTime = 0;
  
  VectorXd gradient, w;
  if (productForm) {
    gradient = diagV * (x - mu).array();
    w = diagV * v.array(); // useful invariant
  }
  else {
    gradient = V * (x - mu);
    w = V * v; // useful invariant
  }
  double a = v.dot(gradient);
  double b = v.dot(w);
  
  int iteration = 0;
  
  Initialize(dim, n_iter);
  Push(currentTime, x, v);

  while (iteration < n_iter || currentTime < finalTime) {
//    Rprintf("iteration: %d, n_iter: %d, currentTime: %g, finalTime: %g\n", iteration, n_iter, currentTime, finalTime);
    ++iteration;
    NumericVector U(runif(2));
    t_reflect = getRandomTime(a, b, U(0));
    if (refresh_rate <= 0) {
      t_refresh = -1; // indicating refresh rate = infinity
      t = t_reflect;
    }
    else {
      t_refresh = -log(U(1))/refresh_rate;
      t = (t_reflect < t_refresh ? t_reflect : t_refresh);
    }
    currentTime = currentTime + t;
    x = x + t * v;
    gradient = gradient + t * w;

    if (t_refresh < 0 || t_reflect < t_refresh) {
      VectorXd normalized_gradient = gradient.normalized(); // for projection
      VectorXd delta_v = - 2 * (v.dot(normalized_gradient)) * normalized_gradient;
      v = v + delta_v;
    }
    else
      v = resampleVelocity(dim, unit_velocity);
//    Rprintf("%g\n", v.norm());
    if (productForm)
      w = diagV * v.array();
    else
      w = V * v; // preserves invariant for w
    a = v.dot(gradient);
    b = v.dot(w);
    Push(currentTime, x, v, finalTime);
  }
  ShrinkToCurrentSize();
}

List Skeleton::toR() {
  // output: R list consisting of Points and Points, and if samples are collected these too
  return List::create(Named("skeletonTimes") = Times, Named("skeletonPoints") = Points, Named("skeletonDirections") = Directions, Named("samples") = samples, Named("mode") = mode, Named("batchMeans") = batchMeans, Named("means") = means, Named("covariance") = covarianceMatrix, Named("asVarEst") = asVarEst, Named("ESS") = ESS);
}

void Skeleton::sample(const int n_samples) {
  
  const int n_steps = Times.size();
  const int dim = Points.rows();
  const double t_max = Times(n_steps-1);
  const double dt = t_max / (n_samples+1);
  
  double t_current = dt;
  double t0 = Times(0);
  double t1;
  VectorXd x0(Points.col(0));
  VectorXd x1(dim);
  samples = MatrixXd(dim, n_samples);
  int n_sampled = 0; // number of samples collected
  
  for (int i = 1; i < n_steps; ++i) {
    x1 = Points.col(i);
    t1 = Times(i);
    while (t_current < t1 && n_sampled < n_samples) {
      samples.col(n_sampled) = x0 + (x1-x0) * (t_current - t0)/(t1-t0);
      ++n_sampled;
      t_current = t_current + dt;
    }
    x0 = x1;
    t0 = t1;
  }
}

//' ZigZagLogistic
//' 
//' Applies the Zig-Zag Sampler to logistic regression, as detailed in Bierkens, Fearnhead, Roberts, The Zig-Zag Process and Super-Efficient Sampling for Bayesian Analysis of Big Data, 2016.
//'
//' @param dataX Matrix containing the independent variables x. The i-th column represents the i-th observation with components x_{1,i}, ..., x_{d,i}.
//' @param dataY Vector of length n containing {0, 1}-valued observations of the dependent variable y.
//' @param n_iterations Integer indicating the number of iterations, i.e. the number of proposed switches.
//' @param x0 Optional argument indicating the starting point for the Zig-Zag sampler
//' @param finalTime If provided and nonnegative, run the sampler until a trajectory of continuous time length finalTime is obtained (ignoring the value of \code{n_iterations})
//' @param subsampling Boolean. Use Zig-Zag with subsampling if TRUE. 
//' @param controlvariates Boolean. Use Zig-Zag with subsampling combined with control variates if TRUE (overriding any value of \code{subsampling}).
//' @param n_samples Number of discrete time samples to extract from the Zig-Zag skeleton.
//' @param n_batches If non-zero, estimate effective sample size through the batch means method, with n_batches number of batches.
//' @param computeCovariance Boolean indicating whether to estimate the covariance matrix.
//' @param upperbound Boolean. If TRUE, sample without subsampling and using a constant upper bound instead of a linear Hessian dependent upper bound
//' @return Returns a list with the following objects:
//' @return \code{skeletonTimes}: Vector of switching times
//' @return \code{skeletonPoints}: Matrix whose columns are locations of switches. The number of columns is identical to the length of \code{skeletonTimes}. Be aware that the skeleton points themselves are NOT samples from the target distribution.
//' @return \code{skeletonDirections}: Matrix whose columns are directions just after switches. The number of columns is identical to the length of \code{skeletonTimes}.
//' @return \code{samples}: If \code{n_samples > 0}, this is a matrix whose \code{n_samples} columns are samples at fixed intervals along the Zig-Zag trajectory. 
//' @return \code{mode}: If \code{controlvariates = TRUE}, this is a vector containing the posterior mode obtained using Newton's method. 
//' @return \code{batchMeans}: If \code{n_batches > 0}, this is a matrix whose \code{n_batches} columns are the batch means
//' @return \code{means}: If \code{n_batches > 0}, this is a vector containing the means of each coordinate along the Zig-Zag trajectory 
//' @return \code{covariance}: If \code{n_batches > 0} or \code{computeCovariance = TRUE}, this is a matrix containing the sample covariance matrix along the trajectory
//' @return \code{asVarEst}: If \code{n_batches > 0} this is an estimate of the asymptotic variance along each component
//' @return \code{ESS}: If \code{n_batches > 0} this is an estimate of the effective sample size along each component
//' @examples
//' require("RZigZag")
//' generate.logistic.data <- function(beta, n.obs) {
//'   dim <- length(beta)
//'   dataX <- rbind(rep(1, n.obs), matrix(rnorm((dim -1) * n.obs), nrow = dim -1));
//'   vals <- colSums(dataX * as.vector(beta))
//'   generateY <- function(p) { rbinom(1, 1, p)}
//'   dataY <- sapply(1/(1 + exp(-vals)), generateY)
//'   return(list(dataX, dataY))
//' }
//'
//' beta <- c(1,2)
//' data <- generate.logistic.data(beta, 1000)
//' result <- ZigZagLogistic(data[[1]], data[[2]], 1000, n_samples = 100)
//' plot(result$skeletonPoints[1,], result$skeletonPoints[2,],type='l',asp=1)
//' points(result$samples[1,], result$samples[2,], col='magenta')
//' @export
// [[Rcpp::export]]
List ZigZagLogistic(const Eigen::MatrixXd dataX, const Eigen::VectorXi dataY, int n_iterations, const NumericVector x0 = NumericVector(0), const double finalTime = -1, const bool subsampling = true, const bool controlvariates = true, const int n_samples = 0, const int n_batches = 0, const bool computeCovariance = false, const bool upperbound = false) {
  
  const int dim = dataX.rows();
  VectorXd x(dim);
  if (x0.size() < dim)
    x = VectorXd::Zero(dim);
  else
    for (int i = 0; i < dim; ++i)
      x[i] = x0[i];
  Skeleton skeleton;
  if (finalTime >= 0)
    n_iterations = -1;

  if (upperbound)
    skeleton.LogisticUpperboundZZ(dataX, dataY, n_iterations, finalTime, x);
  else if (controlvariates) 
    skeleton.LogisticCVZZ(dataX, dataY, n_iterations, finalTime, x);
  else if (subsampling && !controlvariates) 
    skeleton.LogisticSubsamplingZZ(dataX, dataY, n_iterations, finalTime, x);
  else 
    skeleton.LogisticBasicZZ(dataX, dataY, n_iterations, finalTime, x);
  
  if (n_samples > 0)
    skeleton.sample(n_samples);
  if (n_batches > 0)
    skeleton.computeBatchMeans(n_batches);
  if (computeCovariance)
    skeleton.computeCovariance();
  return skeleton.toR();
}

//' ZigZagGaussian
//' 
//' Applies the Zig-Zag Sampler to a Gaussian target distribution, as detailed in Bierkens, Fearnhead, Roberts, The Zig-Zag Process and Super-Efficient Sampling for Bayesian Analysis of Big Data, 2016.
//' Assume potential of the form \deqn{U(x) = (x - mu)^T V (x - mu)/2,} i.e. a Gaussian with mean vector \code{mu} and covariance matrix \code{inv(V)}
//'
//' @param V the inverse covariance matrix (or precision matrix) of the Gaussian target distribution; if V is a matrix consisting of a single column, it is interpreted as the diagonal of the precision matrix.//' @param mu mean of the Gaussian target distribution
//' @param mu mean of the Gaussian target distribution
//' @param n_iterations Number of algorithm iterations; will result in the equivalent amount of skeleton points in Gaussian case because no rejections are needed.
//' @param x0 starting point
//' @param finalTime If provided and nonnegative, run the sampler until a trajectory of continuous time length finalTime is obtained (ignoring the value of \code{n_iterations})
//' @param n_samples Number of discrete time samples to extract from the Zig-Zag skeleton.
//' @param n_batches If non-zero, estimate effective sample size through the batch means method, with n_batches number of batches.
//' @param computeCovariance Boolean indicating whether to estimate the covariance matrix.
//' @return Returns a list with the following objects:
//' @return \code{skeletonTimes}: Vector of switching times
//' @return \code{skeletonPoints}: Matrix whose columns are locations of switches. The number of columns is identical to the length of \code{skeletonTimes}. Be aware that the skeleton points themselves are NOT samples from the target distribution.
//' @return \code{skeletonDirections}: Matrix whose columns are directions just after switches. The number of columns is identical to the length of \code{skeletonTimes}.
//' @return \code{samples}: If \code{n_samples > 0}, this is a matrix whose \code{n_samples} columns are samples along the Zig-Zag trajectory.
//' @return \code{mode}: Not used for a Gaussian target.
//' @return \code{batchMeans}: If \code{n_batches > 0}, this is a matrix whose \code{n_batches} columns are the batch means
//' @return \code{means}: If \code{n_batches > 0}, this is a vector containing the means of each coordinate along the Zig-Zag trajectory 
//' @return \code{covariance}: If \code{n_batches > 0} or \code{computeCovariance = TRUE}, this is a matrix containing the sample covariance matrix along the trajectory
//' @return \code{asVarEst}: If \code{n_batches > 0} this is an estimate of the asymptotic variance along each component
//' @return \code{ESS}: If \code{n_batches > 0} this is an estimate of the effective sample size along each component
//' @examples
//' V <- matrix(c(3,1,1,3),nrow=2)
//' mu <- c(2,2)
//' x0 <- c(0,0)
//' result <- ZigZagGaussian(V, mu, 100, x0, n_samples = 10)
//' plot(result$skeletonPoints[1,], result$skeletonPoints[2,],type='l',asp=1)
//' points(result$samples[1,], result$samples[2,], col='magenta')
//' 
//' V <- matrix(rep(1,100),nrow=100)
//' mu <- numeric(100)
//' x0 <- numeric(100)
//' result <- ZigZagGaussian(V, mu, 1000, x0, n_samples = 10)
//' plot(result$skeletonPoints[1,], result$skeletonPoints[2,],type='l',asp=1)
//' points(result$samples[1,], result$samples[2,], col='magenta')
//' @export
// [[Rcpp::export]]
List ZigZagGaussian(const Eigen::MatrixXd V, const Eigen::VectorXd mu, int n_iterations, const Eigen::VectorXd x0, const double finalTime = -1, const int n_samples=0, const int n_batches=0, bool computeCovariance=false) {
  Skeleton skeleton;
  if (finalTime >= 0)
    n_iterations = -1;
  skeleton.GaussianZZ(V, mu, n_iterations, finalTime, x0);
  if (n_samples > 0)
    skeleton.sample(n_samples);
  if (n_batches > 0)
    skeleton.computeBatchMeans(n_batches);
  if (computeCovariance)
    skeleton.computeCovariance();
  return skeleton.toR();
}

//' BPSGaussian
//' 
//' Applies the BPS Sampler to a Gaussian target distribution, as detailed in Bouchard-Côté et al, 2017.
//' Assume potential of the form \deqn{U(x) = (x - mu)^T V (x - mu)/2,} i.e. a Gaussian with mean vector \code{mu} and covariance matrix \code{inv(V)}
//'
//' @param V the inverse covariance matrix (or precision matrix) of the Gaussian target distribution; if V is a matrix consisting of a single column, it is interpreted as the diagonal of the precision matrix.
//' @param mu mean of the Gaussian target distribution
//' @param n_iterations Number of algorithm iterations; will result in the equivalent amount of skeleton points in Gaussian case because no rejections are needed.
//' @param x0 starting point
//' @param finalTime If provided and nonnegative, run the BPS sampler until a trajectory of continuous time length finalTime is obtained (ignoring the value of \code{n_iterations})
//' @param refresh_rate \code{lambda_refresh}
//' @param unit_velocity TRUE indicates velocities uniform on unit sphere, FALSE indicates standard normal velocities
//' @param n_samples Number of discrete time samples to extract from the skeleton.
//' @param n_batches If non-zero, estimate effective sample size through the batch means method, with n_batches number of batches.
//' @param computeCovariance Boolean indicating whether to estimate the covariance matrix.
//' @return Returns a list with the following objects:
//' @return \code{skeletonTimes}: Vector of switching times
//' @return \code{skeletonPoints}: Matrix whose columns are locations of switches. The number of columns is identical to the length of \code{skeletonTimes}. Be aware that the skeleton points themselves are NOT samples from the target distribution.
//' @return \code{skeletonDirections}: Matrix whose columns are directions just after switches. The number of columns is identical to the length of \code{skeletonTimes}.
//' @return \code{samples}: If \code{n_samples > 0}, this is a matrix whose \code{n_samples} columns are samples along the Zig-Zag trajectory.
//' @return \code{mode}: Not used for a Gaussian target.
//' @return \code{batchMeans}: If \code{n_batches > 0}, this is a matrix whose \code{n_batches} columns are the batch means
//' @return \code{means}: If \code{n_batches > 0}, this is a vector containing the means of each coordinate along the Zig-Zag trajectory 
//' @return \code{covariance}: If \code{n_batches > 0} or \code{computeCovariance = TRUE}, this is a matrix containing the sample covariance matrix along the trajectory
//' @return \code{asVarEst}: If \code{n_batches > 0} this is an estimate of the asymptotic variance along each component
//' @return \code{ESS}: If \code{n_batches > 0} this is an estimate of the effective sample size along each component
//' @examples
//' V <- matrix(c(3,1,1,3),nrow=2)
//' mu <- c(2,2)
//' x0 <- c(0,0)
//' result <- BPSGaussian(V, mu, 100, x0, n_samples = 10)
//' plot(result$skeletonPoints[1,], result$skeletonPoints[2,],type='l',asp=1)
//' points(result$samples[1,], result$samples[2,], col='magenta')
//' V <- matrix(rep(1,100),nrow=100)
//' mu <- numeric(100)
//' x0 <- numeric(100)
//' result <- BPSGaussian(V, mu, 1000, x0, n_samples = 10)
//' plot(result$skeletonPoints[1,], result$skeletonPoints[2,],type='l',asp=1)
//' points(result$samples[1,], result$samples[2,], col='magenta')
//' @export
// [[Rcpp::export]]
List BPSGaussian(const Eigen::MatrixXd V, const Eigen::VectorXd mu, int n_iterations, const Eigen::VectorXd x0, const double finalTime = -1.0, const double refresh_rate = 1, const bool unit_velocity = true, const int n_samples=0, const int n_batches=0, bool computeCovariance=false) {

  Skeleton skeleton;
  if (finalTime >= 0)
    n_iterations = -1;
  skeleton.GaussianBPS(V, mu, n_iterations, finalTime, x0, refresh_rate, unit_velocity);
  if (n_samples > 0)
    skeleton.sample(n_samples);
  if (n_batches > 0)
    skeleton.computeBatchMeans(n_batches);
  if (computeCovariance)
    skeleton.computeCovariance();
  return skeleton.toR();
}

double getRandomTime(double a, double b, double u) {
  // simulate T such that P(T>= t) = exp(-at-bt^2/2), using uniform random input u
  // NOTE: Return value -1 indicates +Inf!
  if (b > 0) {
    if (a < 0) 
      return -a/b + getRandomTime(0, b, u);
    else       // a >= 0
      return -a/b + sqrt(a*a/(b * b) - 2 * log(u)/b);
  }
  else if (b == 0) {
    if (a > 0)
      return -log(u)/a;
    else
      return -1; // infinity
  }
  else {
    // b  < 0
    if (a <= 0)
      return -1; // infinity
    else {
      // a > 0
      double t1 = -a/b;
      if (-log(u) <= a * t1 + b * t1 * t1/2)
        return -a/b - sqrt(a*a/(b * b) - 2 * log(u)/b);
      else
        return -1;
    }
  }
}
