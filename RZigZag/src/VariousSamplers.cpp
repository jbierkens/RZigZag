// VariousSamplers.cpp
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

#include "VariousSamplers.h"

void IID_Sampler::InitializeWaitingTimes() {
  VectorXd uniforms = getUniforms(dim);
  waitingTimes = VectorXd(dim);
  for (int i = 0; i < dim; ++i) {
    waitingTimes(i) = sampleWaitingTime(state.x(i),state.v(i), uniforms(i));
  }
}

double IID_Sampler::sampleWaitingTime(const double x, const double v, const double uniform) const {
  
  double U_val = (v *(x-x0) > 0 ? univariatePotential(x) : univariatePotential(x0));
  if (v > 0) {
    return -x/v + inversePotentialPlus(U_val - log(uniform))/v;
  }
  else {
    return -x/v + inversePotentialMinus(U_val - log(uniform))/v;
  }
}

bool IID_Sampler::simulationStep() {
  SizeType index;
  double deltaT = waitingTimes.minCoeff(&index);
  waitingTimes = waitingTimes.array() - deltaT;
  state.x += state.v * deltaT;
  state.t += deltaT;
  state.v(index) = -state.v(index);
  waitingTimes(index) = sampleWaitingTime(state.x(index), state.v(index), getUniforms(1)(0));
  return true;
}
