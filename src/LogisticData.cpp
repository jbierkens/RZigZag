// LogisticData.cpp : implements useful subroutines for logistic regression
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

#include "LogisticData.h"

LogisticData::LogisticData(const MatrixXd* dataXptr, const VectorXi* dataYptr) : dataXptr(dataXptr), dataYptr(dataYptr) {
  dim = dataXptr->rows();
  n_observations = dataXptr->cols();
}

double LogisticData::potential(const VectorXd& beta) const {
  double val = 0;
  for (int j = 0; j < n_observations; ++j) {
    double innerproduct = beta.dot(dataXptr->col(j));
    val += log(1 + exp(innerproduct)) - (*dataYptr)(j) * innerproduct;
  }
  return val;
}

VectorXd LogisticData::gradient(const VectorXd& beta) const {
  VectorXd grad(VectorXd::Zero(dim));
  for (int j = 0; j < n_observations; ++j) {
    double val = exp(dataXptr->col(j).dot(beta));
    grad += dataXptr->col(j) * (val/(1+val) - (*dataYptr)(j));
  }
  return grad;
}

MatrixXd LogisticData::hessian(const VectorXd& beta) const {
  MatrixXd hess(MatrixXd::Zero(dim,dim));
  for (int j = 0; j < n_observations; ++j) {
    double innerproduct = beta.dot(dataXptr->col(j));
    hess += (dataXptr->col(j) * dataXptr->col(j).transpose())* exp(innerproduct)/((1+exp(innerproduct)*(1+exp(innerproduct))));
  }
  return hess;
}


// MatrixXd preprocessLogistic(const MatrixXd& dataX) {
//   const int n_observations = dataX.cols();
//   const int n_components = dataX.rows() + 1; // we will add a row of constant, so for practical purposes +1
//   
//   // CURRENTLY OFF: re-center the data around the origin. TODO: How does this affect the results?
//   // VectorXd meanX = dataX.rowwise().sum()/n_observations;
//   VectorXd meanX(VectorXd::Zero(n_components));
//   MatrixXd dataXpp(n_components, n_observations);
//   dataXpp.topRows(1) = MatrixXd::Constant(1, n_observations,1);
//   for (int i = 0; i < n_observations; ++i)
//     dataXpp.bottomRows(n_components - 1).col(i) = dataX.col(i) - meanX;
//   return dataXpp;
// }

MatrixXd domHessianLogistic(const MatrixXd& dataX) {
  const int n_observations = dataX.cols();
  const int dim = dataX.rows();
  
  MatrixXd domHessian(MatrixXd::Zero(dim,dim));
  for (int j = 0; j < n_observations; ++j) {
    domHessian += 0.25 * (dataX.col(j) * dataX.col(j).transpose());
  }
  return domHessian;
}


VectorXd cvBound(const MatrixXd& dataX) {
  const int dim = dataX.rows();
  const int n_observations = dataX.cols();
  const VectorXd norms (dataX.colwise().norm());
  VectorXd bounds(dim);
  
  for (int k =0; k < dim; ++k) {
    bounds(k) = 0.0;
    for (int l = 0; l < n_observations; ++l) {
      double val = fabs(dataX(k,l) * norms(l));
      if (bounds(k) < val)
        bounds(k) = val;
    }
  }
  return 0.25 * bounds;
}

double derivativeLogistic(const MatrixXd& dataX, const VectorXi& dataY, const VectorXd& beta, int k) {
  // compute dPsi/dbeta_k for logistic regression
  
  const int n_observations = dataX.cols();
  double derivative = 0;
  
  for (int j = 0; j < n_observations; ++j) {
    double val = exp(dataX.col(j).dot(beta));
    derivative += dataX(k,j) * (val/(1+val) - dataY(j));
  }
  return derivative;
}

VectorXd logisticUpperbound(const MatrixXd& dataX) {
  return dataX.array().abs().rowwise().maxCoeff();
}

double newtonLogistic(const LogisticData& data, VectorXd& beta, double precision, const int max_iter) {
  VectorXd grad(data.gradient(beta));
  int i = 0;
  for (i = 0; i < max_iter; ++i) {
    if (grad.norm() < precision)
      break;
    MatrixXd H(data.hessian(beta));
    beta -= H.ldlt().solve(grad);
    grad = data.gradient(beta);
  }
  if (i == max_iter) {
    stop("newtonLogistic: Maximum number of iterations reached without convergence in Newton's method in computing control variate.");
  }
  else
    Rprintf("newtonLogistic: Converged to local minimum in %d iterations.\n", i);
  return data.potential(beta);
}

