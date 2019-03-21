// ZigZag.cpp
//
// Copyright (C) 2017--2019 Joris Bierkens
//
// This file is part of RZigZag.
//
// This is an attempt to bring all zigzag functionality into a pure C++ setting
// with the idea of possible interfacing to both Python and R
//
// ZigZag is free software: you can redistribute it and/or modify it
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
// along with ZigZag.  If not, see <http://www.gnu.org/licenses/>.

#include "ZigZag.h"

Skeleton::Skeleton(const int dim, int initialSize) {
  Positions = MatrixXd(dim, initialSize);
  Velocities = MatrixXd(dim,initialSize);
  dimension = dim;
  Times = VectorXd(initialSize);
  capacity = initialSize;
  size = 0;
}

void Skeleton::Resize(const int factor) {
  capacity *= factor;
  Times.conservativeResize(capacity);
  Positions.conservativeResize(dimension, capacity);
  Velocities.conservativeResize(dimension, capacity);
}

void Skeleton::Push(const State& state, const double finalTime) {
  if (size >= capacity)
    Resize();
  Velocities.col(size) = state.v;
  if (finalTime < 0 || state.t < finalTime) {
    Times[size] = state.t;
    Positions.col(size) = state.x;
  }
  else {
    Times[size] = finalTime;
    double previousTime = Times[size-1];
    VectorXd previousPosition = Positions.col(size-1);
    Positions.col(size) = previousPosition + (finalTime - previousTime) * (state.x - previousPosition) / (state.t - previousTime);
  }
  size++;
}

void Skeleton::ShrinkToCurrentSize() {
  Times.conservativeResize(size);
  Positions.conservativeResize(dimension, size);
  Velocities.conservativeResize(dimension, size);
  capacity = size;
}

MatrixXd Skeleton::sample(const int n_samples) const {
  
  const int n_steps = Times.size();
  if (n_steps < 2)
    throw "Skeleton::sample: skeleton size < 2.";
  const int dim = Positions.rows();
  const double t_max = Times(n_steps-1);
  const double dt = t_max / (n_samples+1);
  
  double t_current = dt;
  double t0 = Times(0);
  double t1;
  VectorXd x0(Positions.col(0));
  VectorXd x1(dim);
  MatrixXd samples(dim, n_samples);
  int n_sampled = 0; // number of samples collected
  
  for (int i = 1; i < n_steps; ++i) {
    x1 = Positions.col(i);
    t1 = Times(i);
    while (t_current < t1 && n_sampled < n_samples) {
      samples.col(n_sampled) = x0 + (x1-x0) * (t_current - t0)/(t1-t0);
      ++n_sampled;
      t_current = t_current + dt;
    }
    x0 = x1;
    t0 = t1;
  }
  return samples;
}

bool RejectionSampler::simulationStep() {
  // returns true if a switch is accepted

  bool accepted = false;
  SizeType proposedIndex(proposeEvent()); // this moves the full sampler state ahead in time
  
  double V = getUniforms(1)(0);
  double bound = getBound(proposedIndex);
  double trueIntensity = getTrueIntensity(proposedIndex);
  if (trueIntensity > bound)
    throw "RejectionSampler::simulateEvent(): switching rate > bound.";
  if (V <= trueIntensity/bound) {
    state.v(proposedIndex) = -state.v(proposedIndex);
    updateBound(proposedIndex, -trueIntensity);
    accepted = true;
  }
  else
    updateBound(proposedIndex, trueIntensity);
  return accepted;
}

SizeType AffineRejectionSampler::proposeEvent() {
  
  VectorXd U(getUniforms(dim));
  SizeType index = - 1;
  double deltaT = -1;
  
  for (int i = 0; i < dim; ++i) {
    double simulatedTime = getTimeAffineBound(a(i), b(i), U(i));
    if (simulatedTime > 0 && (index == -1 || simulatedTime < deltaT)) {
      index = i;
      deltaT = simulatedTime;
    }
  }
  if (deltaT < 0)
  {
    throw "RejectionSampler::simulateEvent(): wandered off to infinity.";
  }
  else {
    a += b * deltaT;
    state.x += deltaT * state.v;
    state.t += deltaT;
    return index;
  }
}

MatrixXd Skeleton::estimateCovariance(const SizeType coordinate) const {

  const SizeType dim = ( coordinate < 0 ? Positions.rows() : 1);
  
  const double t_max = Times[size-1];
  
  double t0 = Times[0];
  VectorXd x0(dim), x1(dim);
  if (coordinate < 0)
    x0 = Positions.col(0);
  else
    x0 = Positions.row(coordinate).col(0);
  
  MatrixXd covarianceMatrix = VectorXd::Zero(dim, dim);
  VectorXd means = VectorXd::Zero(dim);
  
  for (int i = 1; i < size; ++i) {
    double t1 = Times[i];
    if (coordinate < 0)
      x1 = Positions.col(i);
    else
      x1 = Positions.row(coordinate).col(i);
    // the following expression equals \int_{t_0}^{t_1} x(t) (x(t))^T d t
    covarianceMatrix += (t1 - t0) * (2 * x0 * x0.transpose() + x0 * x1.transpose() + x1 * x0.transpose() + 2 * x1 * x1.transpose())/(6 * t_max);
    means += (t1 - t0) * (x1 + x0) /(2 * t_max);
    t0 = t1;
    x0 = x1;
  }
  covarianceMatrix -= means * means.transpose();
  
  return covarianceMatrix;
}

VectorXd Skeleton::estimateAsymptoticVariance(const int n_batches, const SizeType coordinate) const {
  if (n_batches <= 0)
    throw std::range_error("n_batches should be positive.");
  const SizeType dim = (coordinate < 0 ? Positions.rows() : 1);
  const double t_max = Times[size-1];
  const double batch_length = t_max / n_batches;

  double t0 = Times[0];
  VectorXd x0(dim), x1(dim);
  if (coordinate < 0)
    x0 = Positions.col(0);
  else
    x0 = Positions.row(coordinate).col(0);
  
  MatrixXd batchMeans(dim, n_batches);
  
  int batchNr = 0;
  double t_intermediate = batch_length;
  VectorXd currentBatchMean = VectorXd::Zero(dim);
  
  for (int i = 1; i < size; ++i) {
    double t1 = Times[i];
    if (coordinate < 0)
      x1 = Positions.col(i);
    else
      x1 = Positions.row(coordinate).col(i);

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
  VectorXd means(batchMeans.rowwise().sum()/n_batches);
  
  MatrixXd meanZeroBatchMeans = batchMeans.colwise() - means;
  return batch_length * meanZeroBatchMeans.rowwise().squaredNorm()/(n_batches - 1);
  // ESS = (covarianceMatrix.diagonal().array()/asVarEst.array() * t_max).matrix();
}

VectorXd Skeleton::estimateESS(const int n_batches, const SizeType coordinate, VectorXd asVarEst) const {
  MatrixXd covarianceMatrix = estimateCovariance(coordinate);
  if (asVarEst.size() == 0)
    asVarEst = estimateAsymptoticVariance(n_batches, coordinate);
  double t_max = Times[size-1];
  return covarianceMatrix.diagonal().array()/asVarEst.array() * t_max;
}

Skeleton ZigZag(Sampler& sampler, const int n_iter, const double finalTime) {
  
  double currentTime = 0;
  int iteration = 0;
  Skeleton skel(sampler.getDim(), n_iter);
  
  skel.Push(sampler.getState());
  
  while (currentTime < finalTime || iteration < n_iter) {
    iteration++;
    if(sampler.simulationStep()) // i.e. a switch is accepted
      skel.Push(sampler.getState(), finalTime);
  }
  skel.ShrinkToCurrentSize();
//  Rprintf("ZigZag: Fraction of accepted switches: %g\n", double(skel.getSize()-1)/(iteration));
  return skel;
}

double getTimeAffineBound(double a, double b, double u) {
  // simulate T such that P(T>= t) = exp(-at-bt^2/2), using uniform random input u
  // NOTE: Return value -1 indicates +Inf!
  if (b > 0) {
    if (a < 0) 
      return -a/b + getTimeAffineBound(0, b, u);
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

VectorXd newton(VectorXd& x, const FunctionObject& fn, double precision, const int max_iter) {
  VectorXd grad(fn.gradient(x));
  int i = 0;
  for (i = 0; i < max_iter; ++i) {
    if (grad.norm() < precision)
      break;
    MatrixXd H(fn.hessian(x));
    x -= H.ldlt().solve(grad);
    grad = fn.gradient(x);
  }
  if (i == max_iter) {
    messageStream << "newton: warning: Maximum number of iterations reached without convergence." << std::endl;
  }
  else
    messageStream << "newton: Converged to local minimum in " << i << " iterations." << std::endl;
  return grad;
}

