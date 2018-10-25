// GaussianData.cpp : subroutines for logistic regression
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with RZigZag.  If not, see <http://www.gnu.org/licenses/>.

#include "GaussianData.h"


GaussianData::GaussianData(const MatrixXd* Vptr, const VectorXd mu) : Vptr{Vptr}, mu{mu}, dim{Vptr->rows()} {
  if (Vptr->cols() != Vptr->rows()) {
    diagV = Vptr->array();
    productForm = true;
  }
  else {
    productForm = false;
  }
}

GaussianBound GaussianData::getGaussianBoundObject(const VectorXd& x, const VectorXd& v) {
  ArrayXd w, z;
  if (productForm) {
    w = diagV * v.array();
    z = diagV * (x-mu).array();
  }
  else {
    w = ((*Vptr) * v).array();
    z = ((*Vptr) * (x - mu)).array();
  }
  
  VectorXd a(v.array() * z);
  VectorXd b(v.array() * w);
  
  return GaussianBound(a,b,w,z,this);
  
}

void GaussianBound::proposeTimeAndUpdateBound(int& index, double& deltaT) {
  AffineBound::proposeTimeAndUpdateBound(index,deltaT);
  z = z + w * deltaT;
}


void GaussianBound::updateBound(const int index, const double partial_derivative, const VectorXd& x, const VectorXd& v) {
  
  if (dataPtr->productForm) {
    w(index) = w[index] + 2 * v(index) * dataPtr->diagV(index);
    b(index) = v[index]*w[index];
  }
  else {
    w = w + 2 * v(index) * dataPtr->Vptr->col(index).array(); // preserve invariant w = V theta
    b = v.array() * w;
  }
  a = v.array() * z;
}