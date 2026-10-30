/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with GPUMD.  If not, see <https://www.gnu.org/licenses/>.
*/

#pragma once

#include "action.cuh"
#include "utilities/gpu_vector.cuh"
#include <string>
#include <vector>

class Add_Wall : public Action
{
public:
  Add_Wall(
    const std::vector<std::string>& tokens,
    const std::vector<Group>& groups);

  void pre_run(
    const int number_of_steps,
    const double time_step,
    Integrate& integrate,
    std::vector<Group>& groups,
    Atom& atom,
    Box& box,
    Force& force) override;

  void setup_force(
    const double time_step,
    Integrate& integrate,
    std::vector<Group>& groups,
    Atom& atom,
    Box& box,
    Force& force) override;

  void post_force(
    const int step,
    const double time_step,
    Integrate& integrate,
    std::vector<Group>& groups,
    Atom& atom,
    Box& box,
    Force& force) override;

private:
  struct Wall
  {
    int dimension = 2;
    int side = -1; // -1: lower wall; +1: upper wall
    bool use_edge = false;
    double coordinate = 0.0;
    double epsilon = 0.0;
    double sigma = 0.0;
    double cutoff = 0.0;
    double coeff1 = 0.0;
    double coeff2 = 0.0;
    double coeff3 = 0.0;
    double coeff4 = 0.0;
    double energy_offset = 0.0;
  };

  void apply_walls(
    std::vector<Group>& groups,
    Atom& atom,
    Box& box);

  int grouping_method_ = 0;
  int group_id_ = 0;
  std::vector<Wall> walls_;
  GPU_Vector<int> error_flag_;
};
