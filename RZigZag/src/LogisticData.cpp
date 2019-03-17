// LogisticData.cpp : subroutines for logistic regression
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

#include "LogisticData.h"

double LogisticData::potential(const VectorXd& position) const {
  double val = 0;
  for (int j = 0; j < n_observations; ++j) {
    double innerproduct = position.dot(dataXptr->row(j));
    val += log(1 + exp(innerproduct)) - (*dataYptr)(j) * innerproduct;
  }
  return val;
}

VectorXd LogisticData::gradient(const VectorXd& position) const {
  VectorXd grad(VectorXd::Zero(dim));
  for (int j = 0; j < n_observations; ++j) {
    double val = exp(dataXptr->row(j).dot(position));
    grad += dataXptr->row(j).transpose() * (val/(1+val) - (*dataYptr)(j));
  }
  return grad;
}

MatrixXd LogisticData::hessian(const VectorXd& position) const {
  MatrixXd hess(MatrixXd::Zero(dim,dim));
  for (int j = 0; j < n_observations; ++j) {
    double innerproduct = position.dot(dataXptr->row(j));
    hess += (dataXptr->row(j).transpose() * dataXptr->row(j))* exp(innerproduct)/((1+exp(innerproduct)*(1+exp(innerproduct))));
  }
  return hess;
}

double LogisticData::getDerivative(const VectorXd& position, const int index) const {
  
  double derivative = 0;
  for (int j = 0; j < n_observations; ++j) {
    double val = exp(dataXptr->row(j).dot(position));
    derivative += (*dataXptr)(j,index) * (val/(1+val) - (*dataYptr)(j));
  }
  return derivative;
}

AffineBound LogisticData::getAffineBoundObject(const VectorXd& x, const VectorXd& v) const {
  
  const MatrixXd Q(dominatingHessian());
  const VectorXd b(sqrt(dim) * Q.rowwise().norm());
  VectorXd a(dim);
  for (int k = 0; k < dim; ++k) 
    a(k) = v(k) * getDerivative(x, k);
  return AffineBound(a, b);
}

ConstantBound LogisticData::getConstantBoundObject() const {
  
  return ConstantBound(n_observations * dataXptr->array().abs().colwise().maxCoeff());
}

double LogisticDataForSubsampling::getDerivative(const VectorXd& position, const int index) const {

  int J = floor(n_observations*runif(1)(0)); // randomly select observation
  if (x_ref.rows() == 0)
    return n_observations * (*dataXptr)(J,index) * (1.0/(1.0+exp(-dataXptr->row(J).dot(position))) - (*dataYptr)(J));
  else {
    return grad_ref(index) + n_observations * (*dataXptr)(J,index) * (1.0/(1.0+exp(-dataXptr->row(J).dot(position))) -  1.0/(1.0+exp(-dataXptr->row(J).dot(x_ref))));
  }
}

CVBound LogisticDataForSubsampling::getCVBoundObject(const VectorXd& x, const VectorXd& v, VectorXd& x_ref) {
  
  VectorXd grad_ref;
  if (x_ref.rows() == 0) {
    const double precision = 1e-10;
    const int max_iter = 1e2;
    grad_ref = newtonLogistic(x_ref, precision, max_iter);
  }
  else {
    grad_ref = gradient(x_ref);
  }
  setReference(x_ref, grad_ref);

  const VectorXd C(computeUniformBound());
  const VectorXd b = sqrt(getDim()) * C;
  const VectorXd a_ref = (v.cwiseProduct(grad_ref)).unaryExpr(&pospart);
  VectorXd a(x_ref.rows());

  if (x.rows()==0)
    a = a_ref;
  else 
    a = (x-x_ref).norm() * C + a_ref;

  // Rprintf("a: (%g, %g)\n", a[1], a[2]);
  // Rprintf("b: (%g, %g)\n", b[1], b[2]);
    return CVBound(a, b, C, a_ref, x_ref);
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

MatrixXd LogisticData::dominatingHessian() const {

  MatrixXd domHessian(MatrixXd::Zero(dim,dim));
  for (int j = 0; j < n_observations; ++j) {
    domHessian += 0.25 * (dataXptr->row(j).transpose() * dataXptr->row(j));
  }
  return domHessian;
}


VectorXd LogisticData::computeUniformBound() const {
  const VectorXd norms (dataXptr->rowwise().norm());
  VectorXd bounds(dim);
  
  for (int k = 0; k < dim; ++k) {
    bounds(k) = 0.0;
    for (int l = 0; l < n_observations; ++l) {
      double val = fabs((*dataXptr)(l,k) * norms(l));
      if (bounds(k) < val)
        bounds(k) = val;
    }
  }
  return 0.25 * bounds * n_observations;
}

VectorXd LogisticData::newtonLogistic(VectorXd& x, double precision, const int max_iter) {
  if (x.size() != getDim())
    x = VectorXd::Ones(getDim());
  VectorXd grad(gradient(x));
  int i = 0;
  for (i = 0; i < max_iter; ++i) {
    if (grad.norm() < precision)
      break;
    MatrixXd H(hessian(x));
    x -= H.ldlt().solve(grad);
    grad = gradient(x);
  }
  if (i == max_iter) {
    warning("LogisticData::newtonLogistic: Maximum number of iterations reached without convergence in Newton's method in computing control variate.");
  }
  else
    Rprintf("LogisticData::newtonLogistic: Converged to local minimum in %d iterations.\n", i);
  return grad;
}

