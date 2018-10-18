// LogisticData.h : subroutines for logistic regression
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with RZigZag. If not, see <http://www.gnu.org/licenses/>.

#ifndef __LOGISTICDATA_H
#define __LOGISTICDATA_H

// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include "RZigZag.h"

using namespace Rcpp;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;

// MatrixXd preprocessLogistic(const MatrixXd& dataX); // center data and add a row of ones
MatrixXd domHessianLogistic(const MatrixXd& dataX); // compute dominating Hessian for logistic regression
VectorXd logisticUpperbound(const MatrixXd& dataX);
VectorXd cvBound(const MatrixXd& dataX);

// TODO
// MatrixXd LogisticMALA(const MatrixXd& dataX, const VectorXi& dataY, const int n_epochs, const VectorXd& beta0, const double stepsize);


class LogisticData : public DataObject {
public:
  LogisticData(const MatrixXd& dataX, const VectorXi& dataY);
  double potential(const VectorXd& beta) const;
  VectorXd gradient(const VectorXd& beta) const;
  MatrixXd hessian(const VectorXd& beta) const;
  double getDerivative(const VectorXd& position, const int index) const;
  double subsampledDerivative(const VectorXd& position, const int index) const;
private:
  int dim, n_observations;
  const MatrixXd dataX;
  const VectorXi dataY;
};

double newtonLogistic(const LogisticData& data, VectorXd& beta, double precision, const int max_iter);

#endif