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


#include <Rcpp.h>

using namespace Rcpp;

// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;

#define DEFAULTSIZE 1e4

double getRandomTime(double a, double b, double u); // simulate T such that P(T>= t) = exp(-at-bt^2/2), using uniform random input u

class Skeleton {
public:
  void LogisticBasicZZ(const MatrixXd& dataX, const VectorXi& dataY, const int n_iter, const double finalTime, const VectorXd& x0); // logistic regression with zig zag
  void LogisticUpperboundZZ(const MatrixXd& dataX, const VectorXi& dataY, const int n_iter, const double finalTime, const VectorXd& x0); 
  void LogisticSubsamplingZZ(const MatrixXd& dataX, const VectorXi& dataY, const int n_iter, const double finalTime, const VectorXd& x0);
  void LogisticCVZZ(const MatrixXd& dataX, const VectorXi& dataY, const int n_iter, const double finalTime, const VectorXd& x0); // control variates zigzag
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
