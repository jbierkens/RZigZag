// LogisticSampler.h
//
// Copyright (C) 2017--2019 Joris Bierkens
//
// This file is part of RZigZag.
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

#ifndef __LOGISTICSAMPLER_H
#define __LOGISTICSAMPLER_H

#include "ZigZag.h"

using Eigen::VectorXi;

class LogisticData : public FunctionObject {
public:
  LogisticData(const MatrixXd& dataX, const VectorXi& dataY): dataX{dataX}, dataY{dataY}, dim{dataX.cols()}, n_observations{dataX.rows()} {};
  double potential(const VectorXd& position) const;
  VectorXd gradient(const VectorXd& position) const;
  MatrixXd hessian(const VectorXd& position) const;
  double getDerivative(const VectorXd& position, const int index) const;
  MatrixXd dominatingHessian() const;
  VectorXd getUniformBound() const;
  double getSubsampledDerivative(const VectorXd& position, const int index, const VectorXd& x_ref = VectorXd(0)) const;

protected:
  const SizeType dim, n_observations;
  const MatrixXd& dataX;
  const VectorXi& dataY;
};

class LogisticZZ : public AffineRejectionSampler {
public:
  LogisticZZ(const MatrixXd& dataX, const VectorXi& dataY): AffineRejectionSampler(dataX.cols()), data{LogisticData(dataX, dataY)} { InitializeBound();};
  LogisticZZ(const MatrixXd& dataX, const VectorXi& dataY, const VectorXd& x0, const VectorXd& v0): AffineRejectionSampler(State(0.0, x0, v0)), data{LogisticData(dataX, dataY)} { InitializeBound();};
  double getTrueIntensity(const SizeType proposedIndex) const;
  void updateBound(const SizeType proposedIndex, double trueIntensity);
  
private:
  void InitializeBound();
  const LogisticData data;
};

class LogisticCVZZ : public AffineRejectionSampler {
public:
  LogisticCVZZ(const MatrixXd& dataX, const VectorXi& dataY): AffineRejectionSampler(dataX.cols()), data{LogisticData(dataX, dataY)} { InitializeBound();};
  LogisticCVZZ(const MatrixXd& dataX, const VectorXi& dataY, const VectorXd& x0, const VectorXd& v0): AffineRejectionSampler(State(0.0, x0, v0)), data{LogisticData(dataX, dataY)}, x_ref{VectorXd(0)} { InitializeBound();};
  LogisticCVZZ(const MatrixXd& dataX, const VectorXi& dataY, const VectorXd& x0, const VectorXd& v0, const VectorXd& x_ref): AffineRejectionSampler(State(0.0, x0, v0)), data{LogisticData(dataX, dataY)}, x_ref{x_ref} { InitializeBound();};
  double getTrueIntensity(const SizeType proposedIndex) const;
  void updateBound(const SizeType proposedIndex, double trueIntensity);
  
private:
  void InitializeBound();
  const LogisticData data;
  VectorXd x_ref, grad_ref;
  ArrayXd a_ref, C;
};

#endif