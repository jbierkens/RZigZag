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
  LogisticData(const MatrixXd* dataXptr, const VectorXi* dataYptr) : dataXptr{dataXptr}, dataYptr{dataYptr}, dim{dataXptr->rows()}, n_observations{dataXptr->cols()} {};
  double potential(const VectorXd& position) const;
  VectorXd gradient(const VectorXd& position) const;
  MatrixXd hessian(const VectorXd& position) const;
  double getDerivative(const VectorXd& position, const int index) const;
  AffineBound getAffineBoundObject(const VectorXd& position, const VectorXd& direction) const;
  ConstantBound getConstantBoundObject() const;
  VectorXd newtonLogistic(VectorXd& position, double precision, const int max_iter);
  int getDim() const { return dim;};
  MatrixXd dominatingHessian() const;
  VectorXd computeUniformBound() const;

protected:
  const Eigen::Index dim, n_observations;
  const MatrixXd* dataXptr;
  const VectorXi* dataYptr;
};

class LogisticDataForSubsampling : public LogisticData
{
public:
  LogisticDataForSubsampling(const MatrixXd* dataXptr, const VectorXi* dataYptr) : LogisticData(dataXptr, dataYptr) {};
  double getDerivative(const VectorXd& position, const int index) const;
  CVBound getCVBoundObject(VectorXd& position, const VectorXd& direction, VectorXd& x_ref);
private:
  void setReference(const VectorXd& xr, const VectorXd& gr) { x_ref = xr; grad_ref = gr;};
  VectorXd x_ref, grad_ref;
};



#endif