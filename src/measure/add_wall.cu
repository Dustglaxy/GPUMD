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

/*----------------------------------------------------------------------------80
LAMMPS-compatible planar Lennard-Jones 9-3 walls.
------------------------------------------------------------------------------*/

#include "add_wall.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "model/group.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace
{
constexpr int BLOCK_SIZE = 128;

static __global__ void gpu_add_lj93_wall(
  const int group_size,
  const int group_size_sum,
  const int* __restrict__ g_group_contents,
  const int number_of_atoms,
  const int dimension,
  const int side,
  const double coordinate,
  const double cutoff,
  const double coeff1,
  const double coeff2,
  const double coeff3,
  const double coeff4,
  const double energy_offset,
  double* __restrict__ g_x,
  double* __restrict__ g_y,
  double* __restrict__ g_z,
  double* __restrict__ g_fx,
  double* __restrict__ g_fy,
  double* __restrict__ g_fz,
  double* __restrict__ g_potential,
  double* __restrict__ g_virial,
  int* __restrict__ g_error_flag)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= group_size) return;

  const int atom_id = g_group_contents[group_size_sum + tid];
  const double position =
    dimension == 0 ? g_x[atom_id] : (dimension == 1 ? g_y[atom_id] : g_z[atom_id]);
  const double delta = side < 0 ? position - coordinate : coordinate - position;

  // Match LAMMPS fix wall behavior: atoms on or behind the wall are invalid.
  if (delta <= 0.0) {
    atomicExch(g_error_flag, 1);
    return;
  }
  if (delta >= cutoff) return;

  const double rinv = 1.0 / delta;
  const double r2inv = rinv * rinv;
  const double r3inv = r2inv * rinv;
  const double r4inv = r2inv * r2inv;
  const double r8inv = r4inv * r4inv;
  const double r9inv = r8inv * rinv;
  const double r10inv = r8inv * r2inv;

  const double force_magnitude = coeff1 * r10inv - coeff2 * r4inv;
  const double energy = coeff3 * r9inv - coeff4 * r3inv - energy_offset;
  const double force_component = -static_cast<double>(side) * force_magnitude;

  if (dimension == 0) {
    g_fx[atom_id] += force_component;
    g_virial[atom_id] += force_magnitude * delta;
  } else if (dimension == 1) {
    g_fy[atom_id] += force_component;
    g_virial[atom_id + number_of_atoms] += force_magnitude * delta;
  } else {
    g_fz[atom_id] += force_component;
    g_virial[atom_id + 2 * number_of_atoms] += force_magnitude * delta;
  }
  g_potential[atom_id] += energy;
}
} // namespace

Add_Wall::Add_Wall(
  const std::vector<std::string>& tokens,
  const std::vector<Group>& groups)
{
  action_name = "add_wall";

  if (tokens.size() < 9 || (tokens.size() - 4) % 5 != 0) {
    PRINT_INPUT_ERROR(
      "add_wall syntax: add_wall lj93 <group_method> <group_id> "
      "<face> <coordinate|EDGE> <epsilon> <sigma> <cutoff> [...].");
  }
  if (tokens[1] != "lj93") {
    PRINT_INPUT_ERROR("Only the lj93 wall style is currently supported.");
  }

  if (!is_valid_int(tokens[2], &grouping_method_) ||
      grouping_method_ < 0 || grouping_method_ >= static_cast<int>(groups.size())) {
    PRINT_INPUT_ERROR("add_wall: invalid grouping method.");
  }
  if (!is_valid_int(tokens[3], &group_id_) ||
      group_id_ < 0 || group_id_ >= groups[grouping_method_].number) {
    PRINT_INPUT_ERROR("add_wall: invalid group id.");
  }

  for (size_t i = 4; i < tokens.size(); i += 5) {
    Wall wall;
    const std::string& face = tokens[i];
    if (face == "xlo") {
      wall.dimension = 0;
      wall.side = -1;
    } else if (face == "xhi") {
      wall.dimension = 0;
      wall.side = 1;
    } else if (face == "ylo") {
      wall.dimension = 1;
      wall.side = -1;
    } else if (face == "yhi") {
      wall.dimension = 1;
      wall.side = 1;
    } else if (face == "zlo") {
      wall.dimension = 2;
      wall.side = -1;
    } else if (face == "zhi") {
      wall.dimension = 2;
      wall.side = 1;
    } else {
      PRINT_INPUT_ERROR("add_wall: face must be xlo, xhi, ylo, yhi, zlo, or zhi.");
    }

    if (tokens[i + 1] == "EDGE") {
      wall.use_edge = true;
    } else if (!is_valid_real(tokens[i + 1], &wall.coordinate)) {
      PRINT_INPUT_ERROR("add_wall: wall coordinate must be a number or EDGE.");
    }

    if (!is_valid_real(tokens[i + 2], &wall.epsilon) || wall.epsilon < 0.0) {
      PRINT_INPUT_ERROR("add_wall: epsilon must be non-negative.");
    }
    if (!is_valid_real(tokens[i + 3], &wall.sigma) || wall.sigma <= 0.0) {
      PRINT_INPUT_ERROR("add_wall: sigma must be positive.");
    }
    if (!is_valid_real(tokens[i + 4], &wall.cutoff) || wall.cutoff <= 0.0) {
      PRINT_INPUT_ERROR("add_wall: cutoff must be positive.");
    }

    const double sigma3 = wall.sigma * wall.sigma * wall.sigma;
    const double sigma9 = sigma3 * sigma3 * sigma3;
    wall.coeff1 = 6.0 / 5.0 * wall.epsilon * sigma9;
    wall.coeff2 = 3.0 * wall.epsilon * sigma3;
    wall.coeff3 = 2.0 / 15.0 * wall.epsilon * sigma9;
    wall.coeff4 = wall.epsilon * sigma3;
    const double cutoff_inv = 1.0 / wall.cutoff;
    const double cutoff2_inv = cutoff_inv * cutoff_inv;
    const double cutoff3_inv = cutoff2_inv * cutoff_inv;
    const double cutoff9_inv = cutoff3_inv * cutoff3_inv * cutoff3_inv;
    wall.energy_offset = wall.coeff3 * cutoff9_inv - wall.coeff4 * cutoff3_inv;

    walls_.push_back(wall);
  }

  error_flag_.resize(1);
  error_flag_.fill(0);

  printf(
    "Add LJ 9-3 wall potential for group %d of grouping method %d (%zu wall(s)).\n",
    group_id_,
    grouping_method_,
    walls_.size());
}

void Add_Wall::pre_run(
  const int,
  const double,
  Integrate&,
  std::vector<Group>&,
  Atom&,
  Box& box,
  Force&)
{
  box.set_is_orthogonal();
  for (const Wall& wall : walls_) {
    const int dimension = wall.dimension;
    const int periodic =
      dimension == 0 ? box.pbc_x : (dimension == 1 ? box.pbc_y : box.pbc_z);
    if (periodic) {
      PRINT_INPUT_ERROR("add_wall cannot be used along a periodic direction.");
    }
    if (wall.use_edge && !box.is_orthogonal) {
      PRINT_INPUT_ERROR("add_wall EDGE positions currently require an orthogonal box.");
    }
  }
}

void Add_Wall::setup_force(
  const double,
  Integrate&,
  std::vector<Group>& groups,
  Atom& atom,
  Box& box,
  Force&)
{
  apply_walls(groups, atom, box);
}

void Add_Wall::post_force(
  const int,
  const double,
  Integrate&,
  std::vector<Group>& groups,
  Atom& atom,
  Box& box,
  Force&)
{
  apply_walls(groups, atom, box);
}

void Add_Wall::apply_walls(
  std::vector<Group>& groups,
  Atom& atom,
  Box& box)
{
  const int group_size = groups[grouping_method_].cpu_size[group_id_];
  if (group_size == 0) return;

  const int group_size_sum = groups[grouping_method_].cpu_size_sum[group_id_];
  const int number_of_atoms = atom.number_of_atoms;
  const int grid_size = (group_size - 1) / BLOCK_SIZE + 1;

  double* g_x = atom.position_per_atom.data();
  double* g_y = atom.position_per_atom.data() + number_of_atoms;
  double* g_z = atom.position_per_atom.data() + 2 * number_of_atoms;
  double* g_fx = atom.force_per_atom.data();
  double* g_fy = atom.force_per_atom.data() + number_of_atoms;
  double* g_fz = atom.force_per_atom.data() + 2 * number_of_atoms;

  error_flag_.fill(0);
  for (const Wall& wall : walls_) {
    double coordinate = wall.coordinate;
    if (wall.use_edge) {
      if (wall.side < 0) {
        coordinate = 0.0;
      } else {
        coordinate = box.cpu_h[wall.dimension * 3 + wall.dimension];
      }
    }

    gpu_add_lj93_wall<<<grid_size, BLOCK_SIZE>>>(
      group_size,
      group_size_sum,
      groups[grouping_method_].contents.data(),
      number_of_atoms,
      wall.dimension,
      wall.side,
      coordinate,
      wall.cutoff,
      wall.coeff1,
      wall.coeff2,
      wall.coeff3,
      wall.coeff4,
      wall.energy_offset,
      g_x,
      g_y,
      g_z,
      g_fx,
      g_fy,
      g_fz,
      atom.potential_per_atom.data(),
      atom.virial_per_atom.data(),
      error_flag_.data());
    GPU_CHECK_KERNEL
  }

  int host_error = 0;
  error_flag_.copy_to_host(&host_error);
  if (host_error != 0) {
    PRINT_INPUT_ERROR("Atom on or inside add_wall surface.");
  }
}
