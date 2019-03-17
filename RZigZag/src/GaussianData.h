// GaussianData.h : subroutines for logistic regression
//
// Copyright (C) 2018 Joris Bierkens
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

#ifndef __GAUSSIANDATA_H
#define __GAUSSIANDATA_H

// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include "RZigZag.h"

using namespace Rcpp;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::ArrayXd;

class GaussianData;

class GaussianBound : public AffineBound {
public:
  GaussianBound(VectorXd& a, const VectorXd& b, const ArrayXd& w, const ArrayXd& z, const GaussianData* dataPtr) : AffineBound(a,b), w{w}, z{z}, dataPtr{dataPtr} {};
  void proposeTimeAndUpdateBound(int& index, double& deltaT);
  void updateBound(const int index, const double partial_derivative, const VectorXd& x, const VectorXd& v);
  
private:
  ArrayXd w, z;
  const GaussianData* dataPtr;

};

class GaussianData : public DataObject {
public:
  GaussianData(const MatrixXd* Vptr, const VectorXd mu);
  double getDerivative(const VectorXd& position, const int index) const { return 0; };
  int getDim() const { return dim;};
  GaussianBound getGaussianBoundObject(const VectorXd& x, const VectorXd& v);
  friend void GaussianBound::updateBound(const int index, const double partial_derivative, const VectorXd& x, const VectorXd& v);
  
private:
  const MatrixXd* Vptr;
  const VectorXd mu;
  ArrayXd diagV;
  bool productForm;
  const Eigen::Index dim;
};


#endif