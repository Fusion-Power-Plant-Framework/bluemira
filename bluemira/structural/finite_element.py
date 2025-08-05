# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
FEM solvers for 2D structural problems under plain strain linear elasticity assumptions.
"""

import numpy as np
from mpi4py import MPI
from ufl import (
    TrialFunction, TestFunction, sym, grad, inner, dot, as_vector, as_matrix,
    FacetNormal, Measure
)
from dolfinx import fem, cpp
from dolfinx.io import VTKFile
from dolfinx_mpc import LinearProblem, MultiPointConstraint
from dolfinx_mpc.utils import create_normal_approximation

from bluemira.mesh.meshing import Mesh


class FEMPlainStrainLEA2D:
    def __init__(self, mesh: Mesh, cell_markers: cpp.mesh.MeshTags_float64 | cpp.mesh.MeshTags_int32 | None = None,
                 facet_markers: cpp.mesh.MeshTags_float64 | cpp.mesh.MeshTags_int32 | None = None, degree: int = 2,
                 repr: str = "vectorial"):

        if repr != "vectorial":
            raise NotImplementedError("Only 'vectorial' representation is currently supported.")

        self._gdim = 2
        self._repr = repr
        self.domain = mesh
        self.cell_markers = cell_markers
        self.facet_markers = facet_markers
        self.degree = degree
        self.comm = MPI.COMM_WORLD

        self._setup_domain()
        self._define_function_spaces()

        self.mpc = None
        self.uh = None
        self.loads = []
        self._mpc_activated = False

        # Material properties as cell-wise functions
        self.E = fem.Function(self.V0)
        self.nu = fem.Function(self.V0)

        self._define_constitutive_laws()

    def _setup_domain(self):
        self.dx = Measure("dx", domain=self.domain, subdomain_data=self.cell_markers)
        self.ds = Measure("ds", domain=self.domain, subdomain_data=self.facet_markers)
        self.dS = Measure("dS", domain=self.domain, subdomain_data=self.facet_markers)

    def _define_function_spaces(self):
        self.V = fem.functionspace(self.domain, ("P", self.degree, (self._gdim,)))
        self.V0 = fem.functionspace(self.domain, ("DG", 0))  # For E and nu
        self.VDG0_v = fem.functionspace(self.domain, ("DG", 0, (3,)))
        self.du = TrialFunction(self.V)
        self.v = TestFunction(self.V)

    def _define_constitutive_laws(self):
        """
        Defines the constitutive (material) laws for strain and stress.
        These can be evaluated for any displacement function later.
        """
        zz = self.E / ((1 + self.nu) * (1 - 2 * self.nu))
        C = as_matrix([
            [zz * (1 - self.nu), zz * self.nu, 0.0],
            [zz * self.nu, zz * (1 - self.nu), 0.0],
            [0.0, 0.0, 0.5 * zz * (1 - 2 * self.nu)]
        ])

        def _strain(u):
            eps = sym(grad(u))
            return as_vector([eps[0, 0], eps[1, 1], 2 * eps[0, 1]])

        def _stress(u):
            return dot(C, _strain(u))

        self._strain = _strain
        self._stress = _stress

    def set_materials(self, materials: dict[int, tuple[float, float]]):
        """
        Assigns materials to mesh subdomains by tag.
        Each tag is associated with (E, nu).
        """
        cell_tags = self.cell_markers.values
        for tag, (E_val, nu_val) in materials.items():
            indices = np.where(cell_tags == tag)[0]
            self.E.x.array[indices] = E_val
            self.nu.x.array[indices] = nu_val

    def set_normal_pressure(self, value: float, tag: int):
        """
        Apply normal pressure on a boundary facet identified by `tag`.
        """
        T = fem.Constant(self.domain, value)
        n = FacetNormal(self.domain)
        pressure_term = dot(T * n, self.v) * self.ds(tag)
        self.loads.append(pressure_term)

    def apply_slip_conditions(self, slip_tags: list):
        """
        Enforces slip boundary conditions on specified tags using MPC.
        """
        if self.mpc is None:
            self.mpc = MultiPointConstraint(self.V)

        for tag in slip_tags:
            normal_vec = create_normal_approximation(self.V, self.facet_markers, tag)
            self.mpc.create_slip_constraint(self.V, (self.facet_markers, tag), normal_vec)
        self.mpc.finalize()
        self._mpc_activated = True

    def _setup_forms(self):
        self.a_form = inner(self._stress(self.du), self._strain(self.v)) * self.dx

        if self.loads:
            self.L_form = sum(self.loads)
        else:
            zero = fem.Constant(self.domain, 0.0)
            self.L_form = dot(zero * FacetNormal(self.domain), self.v) * self.ds

    def solve(self):
        if not self._mpc_activated:
            raise RuntimeError("Slip conditions not applied. Call apply_slip_conditions() first.")

        if np.all(self.E.x.array == 0.0) or np.all(self.nu.x.array == 0.0):
            raise RuntimeError("Material properties E and nu not set. Did you call set_materials()?")

        self.uh = fem.Function(self.mpc.function_space, name="Displacement")
        self._setup_forms()

        self.problem = LinearProblem(self.a_form, self.L_form, self.mpc, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        self.uh = self.problem.solve()
        return self.uh

    def postprocess_stress_strain(self):
        """
        Computes and returns strain and stress fields from the solution.
        """
        strain_expr = self._strain(self.uh)
        stress_expr = self._stress(self.uh)

        strain_fn = fem.Function(self.VDG0_v, name="Strain")
        stress_fn = fem.Function(self.VDG0_v, name="Stress")

        for i in range(3):
            scalar_space = fem.functionspace(self.domain, ("DG", 0))
            strain_i = fem.Function(scalar_space)
            stress_i = fem.Function(scalar_space)

            strain_i.interpolate(fem.Expression(strain_expr[i], scalar_space.element.interpolation_points()))
            stress_i.interpolate(fem.Expression(stress_expr[i], scalar_space.element.interpolation_points()))

            strain_fn.x.array[i::3] = strain_i.x.array
            stress_fn.x.array[i::3] = stress_i.x.array

        return strain_fn, stress_fn

    def export_vtk(self, filename="output.pvd", *functions):
        """
        Export any number of computed functions to a VTK file.
        """
        with VTKFile(self.domain.comm, filename, "w") as vtk:
            for f in functions:
                vtk.write_function(f)
