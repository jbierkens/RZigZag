// GaussianSampler.cpp
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

#include "GaussianSampler.h"

bool GaussianZZ::simulationStep() {
  VectorXd U(getUniforms(dim));
  int index = -1;
  double deltaT = -1;
  
  for (int i = 0; i < dim; ++i) {
    double simulatedTime = getTimeAffineBound(a(i), b(i), U(i));
    if (simulatedTime > 0 && (index == -1 || simulatedTime < deltaT)) {
      index = i;
      deltaT = simulatedTime;
    }
  }
  state.x += deltaT * state.v; // O(d)
  state.v[index] = -state.v[index];
  state.t += deltaT;
  z = z + w * deltaT; // O(d), invariant z = V * x
  w = w + 2 * state.v(index) * V.col(index).array(); // preserve invariant w = V * v, O(d)
  a = state.v.array() * z; // invariant a = v .* z, O(d)
  b = state.v.array() * w; // invariant ab = v .* w, O(d)
  
  return true;
}

