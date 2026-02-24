# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
FEM solvers for 2D structural problems under plain strain linear elasticity assumptions.
"""

import basix
import numpy as np
from basix.ufl import element as basix_element
from dolfinx import common, cpp, default_real_type, fem
from dolfinx.fem.petsc import LinearProblem as PETScLinearProblem
from dolfinx.fem.petsc import apply_lifting_nest, set_bc_nest
from dolfinx.io import VTKFile
from dolfinx_mpc import LinearProblem as MPCProblem
from dolfinx_mpc import (
    MultiPointConstraint,
    assemble_matrix_nest,
    assemble_vector_nest,
    create_matrix_nest,
    create_vector_nest,
)
from dolfinx_mpc.utils import create_normal_approximation
from mpi4py import MPI
from petsc4py import PETSc
from ufl import (
    FacetNormal,
    Identity,
    Measure,
    TestFunction,
    TrialFunction,
    as_matrix,
    as_tensor,
    as_vector,
    dot,
    grad,
    inner,
    sym,
    tr,
)

from bluemira.mesh.meshing import Mesh


class FEMPlainStrainLEA2D:
    def __init__(
        self,
        mesh: Mesh,
        cell_markers: cpp.mesh.MeshTags_float64 | cpp.mesh.MeshTags_int32 | None = None,
        facet_markers: cpp.mesh.MeshTags_float64 | cpp.mesh.MeshTags_int32 | None = None,
        degree: int = 2,
        repr: str = "vectorial",
    ):
        """
        Initialize the FEM solver for plane strain elasticity.

        Parameters
        ----------
        mesh : Mesh
            The computational mesh.
        cell_markers : MeshTags, optional
            Mesh cell markers for material assignment.
        facet_markers : MeshTags, optional
            Mesh facet markers for boundary conditions.
        degree : int, optional
            Polynomial degree of the finite element basis. Default is 2.
        repr : str, optional
            Representation type for stress/strain ('vectorial' only supported).
        """
        if repr != "vectorial":
            raise NotImplementedError(
                "Only 'vectorial' representation is currently supported."
            )

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
        self.E = fem.Function(self.V0, name="E")
        self.nu = fem.Function(self.V0, name="nu")
        self.E.x.array[:] = 0.0
        self.nu.x.array[:] = 0.0

        self._define_constitutive_laws()

    def _setup_domain(self):
        """
        Set up UFL integration measures for volume (dx) and boundary (ds, dS)
        using provided cell and facet markers.
        """
        self.dx = Measure("dx", domain=self.domain, subdomain_data=self.cell_markers)
        self.ds = Measure("ds", domain=self.domain, subdomain_data=self.facet_markers)
        self.dS = Measure("dS", domain=self.domain, subdomain_data=self.facet_markers)

    def _define_function_spaces(self):
        """
        Define function spaces for:
        - Displacement field (P-deg vector)
        - Material properties (DG(0) scalar)
        - Stress/strain fields (DG(0) vector)
        Also defines trial/test functions used in variational forms.
        """
        self.V = fem.functionspace(self.domain, ("P", self.degree, (self._gdim,)))
        self.V0 = fem.functionspace(self.domain, ("DG", 0))  # For E and nu
        self.VDG0_v = fem.functionspace(self.domain, ("DG", 0, (3,)))
        self.du = TrialFunction(self.V)
        self.v = TestFunction(self.V)

    def _define_constitutive_laws(self):
        """
        Define stress and strain expressions symbolically using material tensors.
        These are stored as lambda functions and evaluated later using displacement.
        """
        zz = self.E / ((1 + self.nu) * (1 - 2 * self.nu))
        C = as_matrix([
            [zz * (1 - self.nu), zz * self.nu, 0.0],
            [zz * self.nu, zz * (1 - self.nu), 0.0],
            [0.0, 0.0, 0.5 * zz * (1 - 2 * self.nu)],
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
        Assigns material properties to regions of the mesh.

        Parameters
        ----------
        materials : dict[int, tuple[float, float]]
            Dictionary mapping cell marker tags to tuples of (E, nu).
        """
        cell_tags = self.cell_markers.values
        for tag, (E_val, nu_val) in materials.items():
            indices = np.where(cell_tags == tag)[0]
            self.E.x.array[indices] = E_val
            self.nu.x.array[indices] = nu_val

    def set_body_load(self, fx: float, fy: float, tags: list[int]):
        """
        Apply a constant body load on a surface identified by its facet tag.

        Parameters
        ----------
        value : float
            force magnitude (positive is outward).
        tag : int
            Facet marker tag identifying the boundary region.
        """
        f = as_vector((PETSc.ScalarType(fx), PETSc.ScalarType(fy)))
        self.loads.append(sum(dot(f, self.v) * self.dx(tag) for tag in tags))

    def set_normal_pressure(self, value: float, tag: int):
        """
        Adds: ∫ dot(p*n, v) ds(tag)
        """
        p = fem.Constant(self.domain, PETSc.ScalarType(value))
        n = FacetNormal(self.domain)
        self.loads.append(dot(p * n, self.v) * self.ds(tag))

    def set_dirichlet_bcs(self, boundary_conditions: list[tuple[int, int, float]]):
        """
        Set Dirichlet boundary conditions on specific component and facet tag.

        Parameters
        ----------
        boundary_conditions : list of (component, facet_tag, value)
            - component: 0 for x, 1 for y
            - facet_tag: boundary marker tag
            - value: value to set
        """
        self.bcs = []
        for component, tag, value in boundary_conditions:
            fdim = self.domain.topology.dim - 1
            self.domain.topology.create_connectivity(fdim, 0)
            facets = self.facet_markers.find(tag)
            dofs = fem.locate_dofs_topological(self.V.sub(component), fdim, facets)
            bc = fem.dirichletbc(PETSc.ScalarType(value), dofs, self.V.sub(component))
            self.bcs.append(bc)

    def apply_slip_conditions(self, slip_tags: list):
        """
        Apply slip boundary conditions using MultiPoint Constraints (MPC)
        on specified facet marker tags.

        Parameters
        ----------
        slip_tags : list[int]
            List of facet marker tags where slip conditions are applied.
        """
        if self.mpc is None:
            self.mpc = MultiPointConstraint(self.V)

        for tag in slip_tags:
            normal_vec = create_normal_approximation(self.V, self.facet_markers, tag)
            self.mpc.create_slip_constraint(
                self.V, (self.facet_markers, tag), normal_vec
            )
        self.mpc.finalize()
        self._mpc_activated = True

    def _setup_forms(self):
        """
        Assemble the bilinear form (stiffness matrix) and linear form (load vector)
        for the variational problem using current boundary conditions and loads.
        """
        self.a_form = inner(self._stress(self.du), self._strain(self.v)) * self.dx

        if self.loads:
            self.L_form = sum(self.loads)
        else:
            zero = fem.Constant(self.domain, PETSc.ScalarType(0.0))
            self.L_form = dot(as_vector((zero, zero)), self.v) * self.dx

    def solve(self):
        """
        Solve the linear system for displacements under applied loads and constraints.

        Returns
        -------
        fem.Function
            The computed displacement field.

        Raises
        ------
        RuntimeError
            If materials are not set or slip conditions are not applied.
        """
        if np.all(self.E.x.array == 0.0) or np.all(self.nu.x.array == 0.0):
            raise RuntimeError(
                "Material properties E and nu not set. Did you call set_materials()?"
            )

        if self._mpc_activated:
            self.uh = fem.Function(self.mpc.function_space, name="Displacement")
            self._setup_forms()
            self.problem = MPCProblem(
                self.a_form,
                self.L_form,
                self.mpc,
                petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
            )
        else:
            self._setup_forms()
            self.problem = PETScLinearProblem(
                self.a_form,
                self.L_form,
                bcs=self.bcs if hasattr(self, "bcs") else None,
                u=self.uh,
                petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
            )

        self.uh = self.problem.solve()
        return self.uh

    def postprocess_stress_strain(self):
        """
        Compute element-wise strain and stress tensors from the solved displacement field.

        Returns
        -------
        strain_fn : fem.Function
            The strain field stored in DG(0, 3) format.
        stress_fn : fem.Function
            The stress field stored in DG(0, 3) format.
        """
        strain_expr = self._strain(self.uh)
        stress_expr = self._stress(self.uh)

        strain_fn = fem.Function(self.VDG0_v, name="Strain")
        stress_fn = fem.Function(self.VDG0_v, name="Stress")

        for i in range(3):
            scalar_space = fem.functionspace(self.domain, ("DG", 0))
            strain_i = fem.Function(scalar_space)
            stress_i = fem.Function(scalar_space)

            strain_i.interpolate(
                fem.Expression(
                    strain_expr[i], scalar_space.element.interpolation_points()
                )
            )
            stress_i.interpolate(
                fem.Expression(
                    stress_expr[i], scalar_space.element.interpolation_points()
                )
            )

            strain_fn.x.array[i::3] = strain_i.x.array
            stress_fn.x.array[i::3] = stress_i.x.array

        return strain_fn, stress_fn

    def export_vtk(self, filename="output.pvd", *functions):
        """
        Export one or more finite element functions to a VTK file for visualization.

        Parameters
        ----------
        filename : str
            The name of the VTK file to write to.
        functions : fem.Function
            One or more functions to include in the output.
        """
        with VTKFile(self.domain.comm, filename, "w") as vtk:
            for f in functions:
                vtk.write_function(f)


class FEMGeneralizedPlainStrainLEA2D:
    def __init__(
        self,
        mesh,
        cell_markers: cpp.mesh.MeshTags_float64 | cpp.mesh.MeshTags_int32 | None = None,
        facet_markers: cpp.mesh.MeshTags_float64 | cpp.mesh.MeshTags_int32 | None = None,
        degree: int = 2,
        ezz_degree: int = 1,
    ):
        self.domain = mesh
        self.cell_markers = cell_markers
        self.facet_markers = facet_markers
        self.degree = degree
        self.ezz_degree = ezz_degree
        self.comm = MPI.COMM_WORLD

        self._setup_measures()
        self._define_function_spaces()

        # MPCs (u and ezz)
        self.mpc_u: MultiPointConstraint | None = None
        self.mpc_ezz: MultiPointConstraint | None = None
        self._mpc_u_active = False

        # Loads (block RHS)
        self.loads_u = []  # forms that live in u-equation
        self.loads_ezz = []  # forms that live in ezz-equation

        # Dirichlet BCs (only used in assembly helpers; kept as list)
        self.bcs = []

        # DG0 material fields
        self.E = fem.Function(self.R0, name="E")
        self.nu = fem.Function(self.R0, name="nu")
        self.E.x.array[:] = 0.0
        self.nu.x.array[:] = 0.0

        # Solution functions
        self.uh = None
        self.ezzh = None

    # ------------------------- setup -------------------------
    def _setup_measures(self):
        self.dx = Measure("dx", domain=self.domain, subdomain_data=self.cell_markers)
        self.ds = Measure("ds", domain=self.domain, subdomain_data=self.facet_markers)
        self.dS = Measure("dS", domain=self.domain, subdomain_data=self.facet_markers)

    def _define_function_spaces(self):
        cellname = self.domain.ufl_cell().cellname()

        # V: vector P(degree)
        Ve = basix.ufl.element(
            basix.ElementFamily.P,
            cellname,
            self.degree,
            shape=(self.domain.geometry.dim,),
            dtype=default_real_type,
        )
        self.V = fem.functionspace(self.domain, Ve)

        # Q: scalar P(ezz_degree)
        Qe = basix.ufl.element(
            basix.ElementFamily.P,
            cellname,
            self.ezz_degree,
            dtype=default_real_type,
        )
        self.Q = fem.functionspace(self.domain, Qe)

        # DG0 for E, nu
        self.R0 = fem.functionspace(self.domain, ("DG", 0))

        # DG0 tensor output (3x3)
        self.TensorDG0 = fem.functionspace(self.domain, ("DG", 0, (3, 3)))

        # Trial/Test
        self.u = TrialFunction(self.V)
        self.v = TestFunction(self.V)
        self.ezz = TrialFunction(self.Q)
        self.ezz_test = TestFunction(self.Q)

    # ------------------------- materials -------------------------
    def set_materials(self, materials: dict[int, tuple[float, float]]):
        """
        materials: {cell_tag: (E, nu)}
        """
        if self.cell_markers is None:
            raise RuntimeError("cell_markers is None; cannot assign materials by tag.")

        cell_tags = self.cell_markers.values
        for tag, (E_val, nu_val) in materials.items():
            idx = np.where(cell_tags == tag)[0]
            self.E.x.array[idx] = E_val
            self.nu.x.array[idx] = nu_val

    # ------------------------- loads -------------------------
    def set_body_load(self, fx: float, fy: float, tags: list[int]):
        """
        Volume body force on u-equation over given cell tags:
            sum_tag ∫ dot(f, v) dx(tag)
        """
        f = as_vector((PETSc.ScalarType(fx), PETSc.ScalarType(fy)))
        self.loads_u.append(sum(dot(f, self.v) * self.dx(tag) for tag in tags))

    def set_normal_pressure(self, value: float, tag: int):
        """
        Boundary pressure traction on u-equation:
            ∫ dot(p*n, v) ds(tag)
        """
        p = fem.Constant(self.domain, PETSc.ScalarType(value))
        n = FacetNormal(self.domain)
        self.loads_u.append(dot(p * n, self.v) * self.ds(tag))

    # ------------------------- BCs / constraints -------------------------
    def set_dirichlet_bcs(self, boundary_conditions: list[tuple[int, int, float]]):
        """
        boundary_conditions: list of (component, facet_tag, value)
            component: 0->x, 1->y
        """
        if self.facet_markers is None:
            raise RuntimeError(
                "facet_markers is None; cannot apply Dirichlet BCs by tag."
            )

        self.bcs = []
        fdim = self.domain.topology.dim - 1
        self.domain.topology.create_connectivity(fdim, 0)

        for component, tag, value in boundary_conditions:
            facets = self.facet_markers.find(tag)
            dofs = fem.locate_dofs_topological(self.V.sub(component), fdim, facets)
            self.bcs.append(
                fem.dirichletbc(PETSc.ScalarType(value), dofs, self.V.sub(component))
            )

    def apply_slip_conditions(self, slip_tags: list[int]):
        """
        Slip MPC on displacement space V only.
        """
        if self.facet_markers is None:
            raise RuntimeError("facet_markers is None; cannot create slip constraints.")

        if self.mpc_u is None:
            self.mpc_u = MultiPointConstraint(self.V)

        for tag in slip_tags:
            nvec = create_normal_approximation(self.V, self.facet_markers, tag)
            # pass bcs=[] here to avoid dolfinx_mpc trying to interpret user bcs in constraint
            self.mpc_u.create_slip_constraint(
                self.V, (self.facet_markers, tag), nvec, bcs=[]
            )

        self.mpc_u.finalize()
        self._mpc_u_active = True

    # ------------------------- constitutive (3D) -------------------------
    def _lambda_mu(self):
        # DG0 lambda/mu from DG0 E,nu
        lmbda = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        mu = self.E / (2 * (1 + self.nu))
        return lmbda, mu

    @staticmethod
    def _eps_3d(u, ezz):
        return sym(
            as_tensor([
                [u[0].dx(0), u[0].dx(1), 0],
                [u[1].dx(0), u[1].dx(1), 0],
                [0, 0, ezz],
            ])
        )

    def _sigma_3d(self, u, ezz):
        lmbda, mu = self._lambda_mu()
        eps = self._eps_3d(u, ezz)
        return lmbda * tr(eps) * Identity(3) + 2 * mu * eps

    # ------------------------- assemble/solve -------------------------
    def _setup_block_forms(self):
        """
        Build 2x2 block bilinear forms a_ij and block RHS L_i.
        This follows your working script pattern closely.
        """
        # helpers
        zero_u = fem.Constant(self.domain, PETSc.ScalarType((0.0, 0.0)))
        zero_ezz = fem.Constant(self.domain, PETSc.ScalarType(0.0))

        # Bilinear blocks
        a00 = fem.form(
            inner(self._sigma_3d(self.u, 0), self._eps_3d(self.v, 0)) * self.dx
        )
        a01 = fem.form(
            inner(self._sigma_3d(zero_u, self.ezz), self._eps_3d(self.v, 0)) * self.dx
        )
        a10 = fem.form(
            inner(self._sigma_3d(self.u, 0), self._eps_3d(zero_u, self.ezz_test))
            * self.dx
        )
        a11 = fem.form(
            inner(self._sigma_3d(zero_u, self.ezz), self._eps_3d(zero_u, self.ezz_test))
            * self.dx
        )

        self.a_blocks = [[a00, a01], [a10, a11]]

        # RHS blocks
        if self.loads_u:
            L0 = sum(self.loads_u)
        else:
            # zero RHS
            f0 = as_vector((PETSc.ScalarType(0.0), PETSc.ScalarType(0.0)))
            L0 = dot(f0, self.v) * self.dx

        if self.loads_ezz:
            L1 = sum(self.loads_ezz)
        else:
            L1 = zero_ezz * self.ezz_test * self.dx

        self.L_blocks = [fem.form(L0), fem.form(L1)]

    def solve(
        self,
        petsc_options: dict | None = None,
        ksp_setup: str = "fieldsplit_gamg",
        rtol: float = 1e-8,
    ):
        """
        Solve the GPS system with nested matrix and fieldsplit.

        ksp_setup:
          - "fieldsplit_gamg" : MINRES + (u: GAMG, ezz: Jacobi)
          - "lu"              : direct LU on full nested (can be huge)
        """
        if np.allclose(self.E.x.array, 0.0) or np.allclose(self.nu.x.array, 0.0):
            raise RuntimeError(
                "Material properties E and nu not set. Call set_materials()."
            )

        self._setup_block_forms()

        # constraints: u has MPC; ezz generally unconstrained but needs a finalized MPC object
        if self.mpc_u is None:
            self.mpc_u = MultiPointConstraint(self.V)
            self.mpc_u.finalize()
        if self.mpc_ezz is None:
            self.mpc_ezz = MultiPointConstraint(self.Q)
            self.mpc_ezz.finalize()

        constraints = [self.mpc_u, self.mpc_ezz]

        # Assemble nested system
        with common.Timer("~Assemble GPS LHS/RHS"):
            A = create_matrix_nest(self.a_blocks, constraints)
            assemble_matrix_nest(A, self.a_blocks, constraints, self.bcs)
            A.assemble()

            b = create_vector_nest(self.L_blocks, constraints)
            assemble_vector_nest(b, self.L_blocks, constraints)

            apply_lifting_nest(b, self.a_blocks, self.bcs)
            for b_sub in b.getNestSubVecs():
                b_sub.ghostUpdate(
                    addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
                )

            set_bc_nest(
                b, fem.bcs_by_block(fem.extract_function_spaces(self.L_blocks), self.bcs)
            )

        # Preconditioner for ezz block
        P11 = fem.petsc.assemble_matrix(fem.form(self.ezz * self.ezz_test * self.dx))
        P11.assemble()

        P = PETSc.Mat().createNest([[A.getNestSubMatrix(0, 0), None], [None, P11]])
        P.assemble()

        # Solve
        ksp = PETSc.KSP().create(self.domain.comm)
        ksp.setOperators(A, P)
        ksp.setTolerances(rtol=rtol)

        if petsc_options is not None:
            # allow user override entirely
            ksp.setFromOptions()
        elif ksp_setup == "lu":
            ksp.setType("preonly")
            ksp.getPC().setType("lu")
        else:
            # default: robust fieldsplit
            ksp.setType("minres")
            pc = ksp.getPC()
            pc.setType("fieldsplit")
            pc.setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)

            nested_IS = P.getNestISs()
            pc.setFieldSplitIS(("u", nested_IS[0][0]), ("ezz", nested_IS[0][1]))

            ksp_u, ksp_ezz = pc.getFieldSplitSubKSP()
            ksp_u.setType("preonly")
            ksp_u.getPC().setType("gamg")
            ksp_ezz.setType("preonly")
            ksp_ezz.getPC().setType("jacobi")

        # Solve into nest vector
        Uh = b.copy()
        ksp.solve(b, Uh)

        for Uh_sub in Uh.getNestSubVecs():
            Uh_sub.ghostUpdate(
                addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
            )

        # Put into Functions
        self.uh = fem.Function(self.mpc_u.function_space, name="Displacement")
        self.ezzh = fem.Function(self.mpc_ezz.function_space, name="ezz")

        self.uh.x.petsc_vec.setArray(Uh.getNestSubVecs()[0].array)
        self.ezzh.x.petsc_vec.setArray(Uh.getNestSubVecs()[1].array)
        self.uh.x.scatter_forward()
        self.ezzh.x.scatter_forward()

        # Backsubstitute to update slave dofs
        self.mpc_u.backsubstitution(self.uh)
        self.mpc_ezz.backsubstitution(self.ezzh)

        return self.uh, self.ezzh

    # ------------------------- postprocess -------------------------
    def postprocess_strain_stress(self):
        """
        Returns DG0 (3x3) strain and stress tensors in full 3D form.
        """
        if self.uh is None or self.ezzh is None:
            raise RuntimeError("No solution. Call solve() first.")

        strain = fem.Function(self.TensorDG0, name="strain")
        stress = fem.Function(self.TensorDG0, name="stress")

        strain_expr = self._eps_3d(self.uh, self.ezzh)
        stress_expr = self._sigma_3d(self.uh, self.ezzh)

        strain.interpolate(
            fem.Expression(strain_expr, self.TensorDG0.element.interpolation_points())
        )
        stress.interpolate(
            fem.Expression(stress_expr, self.TensorDG0.element.interpolation_points())
        )
        return strain, stress

    def export_vtk(self, filename: str, *functions: fem.Function):
        with VTKFile(self.domain.comm, filename, "w") as vtk:
            for f in functions:
                vtk.write_function(f)

    def export_solution_vtk(
        self,
        filename: str = "gps_output.pvd",
        interpolate_u: bool = True,
    ):
        """
        Convenience exporter: writes u (optionally interpolated), ezz, strain, stress.
        """
        if self.uh is None or self.ezzh is None:
            raise RuntimeError("No solution. Call solve() first.")

        # Optionally interpolate u into an unconstrained space for nicer output
        if interpolate_u:
            cellname = self.domain.ufl_cell().cellname()
            V0_elem = basix_element("P", cellname, self.degree, shape=(2,))
            V0 = fem.functionspace(self.domain, V0_elem)
            uh_out = fem.Function(V0, name="displacement")
            uh_out.interpolate(self.uh)
        else:
            uh_out = self.uh

        strain, stress = self.postprocess_strain_stress()

        self.export_vtk(filename, uh_out, self.ezzh, strain, stress)
        return filename
