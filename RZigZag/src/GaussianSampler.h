// GaussianSampler.h
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

#ifndef __GAUSSIANSAMPLER_H
#define __GAUSSIANSAMPLER_H
#include "ZigZag.h"

class GaussianZZ : public Sampler {
public:
  GaussianZZ(const MatrixXd& V, const VectorXd& mu = VectorXd(0), VectorXd x = VectorXd(0), VectorXd v = VectorXd(0)): Sampler(State(V.rows())), V{V}, mu{mu}, w{(V * v).array()}, z{(V * (x - mu)).array()}, a{v.array() * z}, b{v.array() * w} {};
  bool simulationStep();

private:
  const MatrixXd& V;
  const VectorXd& mu;
  ArrayXd w, z; // invariants w = V * v, z = V * x
  ArrayXd a, b; // invariants a = v.* z, b = v .* w
};

#endif