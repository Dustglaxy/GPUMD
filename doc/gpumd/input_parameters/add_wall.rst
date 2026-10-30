.. _kw_add_wall:
.. index::
   single: add_wall (keyword in run.in)

:attr:`add_wall`
=================

This keyword adds a planar Lennard-Jones 9-3 wall potential to atoms in a selected group. Its force and energy convention follows LAMMPS `fix wall/lj93`.

Syntax
------

::

   add_wall lj93 <group_method> <group_id> <face> <coordinate|EDGE> <epsilon> <sigma> <cutoff> [<face> <coordinate|EDGE> <epsilon> <sigma> <cutoff> ...]

A single command can define one or more walls. The six supported faces are `xlo`, `xhi`, `ylo`, `yhi`, `zlo`, and `zhi`. Each wall has its own coordinate, epsilon, sigma, and cutoff.

* `group_method` and `group_id` select the atoms affected by the walls, following the grouping convention used by :ref:`add_force <kw_add_force>`.
* `coordinate` is the fixed Cartesian coordinate of the wall in Angstrom.
* `EDGE` places a lower wall at coordinate zero or an upper wall at the corresponding box length. EDGE currently requires an orthogonal box.
* `epsilon` is the wall energy scale in eV.
* `sigma` and `cutoff` are in Angstrom.
* A wall cannot be placed along a periodic direction.

Potential
---------

For the perpendicular distance :math:`r` from an atom to a wall, the potential is

.. math::

   U(r)=\epsilon\left[\frac{2}{15}\left(\frac{\sigma}{r}\right)^9-\left(\frac{\sigma}{r}\right)^3\right], \qquad 0<r<r_c.

The energy is shifted by its value at the cutoff, :math:`U_{\rm shift}(r)=U(r)-U(r_c)`. The force is not force-shifted, matching LAMMPS `fix wall/lj93`. At and beyond the cutoff, both energy and force are zero.

Atoms on or beyond the wall surface cause an input/runtime error, matching the LAMMPS wall behavior.

Examples
--------

Apply a lower z wall at :math:`z=0\,\mathrm{\AA}`::

   add_wall lj93 0 0 zlo 0.0 0.1 3.0 10.0

Confine a group between two walls::

   add_wall lj93 0 0 zlo 0.0 0.1 3.0 10.0 zhi 50.0 0.1 3.0 10.0

Note
----

Multiple `add_wall` commands are allowed in one run. Wall forces, shifted wall energies, and diagonal wall virials are added to the per-atom GPUMD properties.
