// RZigZag.h : implements Zig-Zag and other PDMP samplers
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

#ifndef __RZIGZAG_H
#define __RZIGZAG_H

// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

using namespace Rcpp;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;

#define DEFAULTSIZE 1e4

double getRandomTime(double a, double b, double u); // simulate T such that P(T>= t) = exp(-at-bt^2/2), using uniform random input u

class ComputationalBound {
public:
  virtual void proposeTimeAndUpdateBound(int& index, double& deltaT) = 0;
  virtual void updateBound(const int index, const double partial_derivative, const VectorXd& x, const VectorXd& v) = 0;
  virtual double getBound(const int index) = 0;
};

class DataObject {

public:
  virtual int getDim() const = 0;
  virtual double getDerivative(const VectorXd& position, const int index) const = 0;
};

class AffineBound : public ComputationalBound {
public:
  AffineBound(const VectorXd& b): b{b} {};
  AffineBound(VectorXd& a, const VectorXd& b): a{a}, b{b} {};
  void proposeTimeAndUpdateBound(int& index, double& deltaT);
  void updateBound(const int index, const double partial_derivative, const VectorXd& x, const VectorXd& v);
  double getBound(const int index);
  
protected:
  VectorXd a;
  const VectorXd b;
};

class ConstantBound : public ComputationalBound {
public:
  ConstantBound(const VectorXd& a): a{a} {};
  void proposeTimeAndUpdateBound(int& index, double& deltaT);
  void updateBound(const int index, const double partial_derivative, const VectorXd& x, const VectorXd& v);
  double getBound(const int index);
  
private:
  const VectorXd a;
};

class CVBound : public AffineBound {
public:
  void updateBound(const int index, const double partial_derivative, const VectorXd& x, const VectorXd& v);
  CVBound(const VectorXd& b, const VectorXd& uniformBound, const VectorXd& x_ref, const VectorXd& grad_ref, const VectorXd& x, const VectorXd& v) : AffineBound(b), uniformBound{uniformBound}, x_ref{x_ref}, grad_ref{grad_ref} { updateBound(0, 0, x, v); };

private:
  const VectorXd uniformBound;  // bound to be used for updates, uniform over observations 
  const VectorXd x_ref;
  const VectorXd grad_ref;
};

class Skeleton {
public:
  void ZigZag(const DataObject& data, ComputationalBound& computationalBound, const VectorXd& x0, const VectorXd& v0, const int n_iter, const double finalTime);
  void LogisticBasicZZ(const MatrixXd& dataX, const VectorXi& dataY, const int n_iter, const double finalTime, const VectorXd& x0, const VectorXd& v0); // logistic regression with zig zag
  void LogisticUpperboundZZ(const MatrixXd& dataX, const VectorXi& dataY, const int n_iter, const double finalTime, const VectorXd& x0, const VectorXd& v0); 
  void LogisticSubsamplingZZ(const MatrixXd& dataX, const VectorXi& dataY, const int n_iter, const double finalTime, const VectorXd& x0, const VectorXd& v0);
  void LogisticCVZZ(const MatrixXd& dataX, const VectorXi& dataY, const int n_iter, const double finalTime, VectorXd& x0, const VectorXd& v0, VectorXd x_ref = VectorXd()); // control variates zigzag
  void GaussianZZ(const MatrixXd& V, const VectorXd& mu, const int n_iter, const double finalTime, const VectorXd& x0); // sample Gaussian with precision matrix V
  void GaussianBPS(const MatrixXd& V, const VectorXd& mu, const int n_iter, const double finalTime, const VectorXd& x0, const double refresh_rate, const bool unit_velocity = true); // sample Gaussian with precision matrix V
  void sample(const int n_samples);
  void computeBatchMeans(const int n_batches);
  void computeCovariance();
  List toR();

private:
  void Initialize(const int dim, int initialSize);
  void Push(const double time, const VectorXd& point, const VectorXd& direction, const double finalTime = -1);
  void ShrinkToCurrentSize(); // shrinks to actual size;
  void Resize(const int factor = 2);
  void ZZStep(ComputationalBound& computationalBound, const DataObject& data, double& currentTime, VectorXd& position, VectorXd& direction, const double intendedFinalTime);
    
  MatrixXd Points;
  MatrixXd Directions;
  VectorXd Times;
  int capacity;
  int currentSize;
  int dimension;

  MatrixXd samples;
  VectorXd mode;
  MatrixXd batchMeans;
  VectorXd means;
  MatrixXd covarianceMatrix;
  VectorXd asVarEst;
  VectorXd ESS;
};


inline double pospart(const double a) {
  if (a > 0)
    return a;
  else
    return 0;
}


#endif