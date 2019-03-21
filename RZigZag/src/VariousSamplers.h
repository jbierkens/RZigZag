// VariousSamplers.h
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

#ifndef __VARIOUSSAMPLERS_H
#define __VARIOUSSAMPLERS_H

#include "ZigZag.h"

class IID_Sampler : public Sampler {
public:
  IID_Sampler(State initialState, double x0 = 0): Sampler(initialState), x0{x0} {};
  IID_Sampler(SizeType dim, double x0 = 0): Sampler(dim), x0{x0} {};
  bool simulationStep();
  
  virtual double inversePotentialPlus(double) const = 0;
  virtual double inversePotentialMinus(double) const = 0;
  virtual double univariatePotential(double) const = 0;

protected:
  const double x0;
  void InitializeWaitingTimes();
  double sampleWaitingTime(const double x, const double v, const double uniform) const;
  VectorXd waitingTimes;
};

class Symmetric_IID_Sampler : public IID_Sampler {
public:
  Symmetric_IID_Sampler(State initialState): IID_Sampler(initialState) {};
  Symmetric_IID_Sampler(SizeType dim): IID_Sampler(dim) {};
  double inversePotentialMinus(double y) const  { return -inversePotentialPlus(y);};
};

class StudentT_IID_Sampler : public Symmetric_IID_Sampler {
public:
  StudentT_IID_Sampler(State initialState, double dof): dof{dof}, Symmetric_IID_Sampler(initialState) { InitializeWaitingTimes(); };
  StudentT_IID_Sampler(SizeType dim, double dof): dof{dof}, Symmetric_IID_Sampler(dim)  { InitializeWaitingTimes(); };
  
  double univariatePotential(double x) const { return (dof + 1)/2 * log(1 + x*x/dof);}
  double inversePotentialPlus(double y) const { return sqrt(dof * (exp(2 * y/(dof+1)) -1));}
private:
  const double dof;
};


class Gaussian_IID_Sampler : public Symmetric_IID_Sampler {
public:
  Gaussian_IID_Sampler(State initialState, double variance): variance{variance}, Symmetric_IID_Sampler(initialState) { InitializeWaitingTimes(); };
  Gaussian_IID_Sampler(SizeType dim, double variance): variance{variance}, Symmetric_IID_Sampler(dim)  { InitializeWaitingTimes(); };
  
  double univariatePotential(double x) const { return x*x/(2 * variance);}
  double inversePotentialPlus(double y) const { return sqrt(2 * variance * y);}
private:
  const double variance;
};

#endif