# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
#                    D. Short
#
# bluemira is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# bluemira is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with bluemira; if not, see <https://www.gnu.org/licenses/>.

"""
An attempt at including structural constraints in TF coil optimisations..
"""
# flake8: noqa - UNDER DEVELOPMENT

import numpy as np

from bluemira.geometry._deprecated_tools import get_intersect
from bluemira.geometry.constants import VERY_BIG
from BLUEPRINT.beams.crosssection import (
    AnalyticalShellComposite,
    CircularHollowBeam,
    CompositeCrossSection,
    MultiCrossSection,
    RectangularBeam,
)
from BLUEPRINT.beams.material import CastEC1, ForgedJJ1, ForgedSS316LN, Material
from BLUEPRINT.beams.model import FiniteElementModel
from BLUEPRINT.beams.node import get_midpoint
from BLUEPRINT.beams.transformation import cyclic_pattern
from BLUEPRINT.geometry.geomtools import (
    circle_seg,
    get_angle_between_vectors,
    qrotate,
    rotate_matrix,
)
from BLUEPRINT.geometry.loop import Loop
from BLUEPRINT.geometry.shell import Shell

FORGED_JJ1 = ForgedJJ1()
FORGED_SS316_LN = ForgedSS316LN()
CAST_EC1 = CastEC1()


class StructuralSolver:
    """
    Structural solver for coil cage structure optimisations.

    Parameters
    ----------
    architect: Type[CoilArchitect]
        The CoilArchitect object used to build the solver model
    coilcage: Type[CoilCage]
        The CoilCage used to calculate the TF forces
    equilibria: List[Union[Type[Equilibrium], Type[Breakdown]]
        The list of equilibria objects used to calculate the PF forces
    material_cache: MaterialCache
        The cache of material definitions.

    Notes
    -----
    This structural model of the TF coil uses cyclic symmetry by default, with
    the TF coil on the y = 0 plane, which is not aligned with the Reactor geometry
    convention.
    """

    def __init__(self, architect, coilcage, equilibria, material_cache):
        self.architect = architect
        self.coilcage = coilcage
        self.equilibria = equilibria
        self.tf = architect.tf
        self.materials = {}
        self.results = None

        self._tf_node_ids = []  # Storage for easier load application
        self._tf_elem_ids = []  # Storage for easier load application
        self._gs_node_ids = []  # Storage for tweaking GS position
        self._set_nose()
        self.model = None
        self.define_materials(material_cache)
        self.define_geometry()

    def _set_nose(self):
        # Simple, for now
        x_min = np.min(self.tf.loops["cl"]["x"])
        self.x_nose = 1.1 * x_min

    def define_materials(self, material_cache):
        """
        Define the materials to be used in the StructuralSolver problem.
        """
        # Load up some materials objects
        nb3_sn_wp = material_cache.get_material("Toroidal_Field_Coil_2015")
        # No properties below 300 K ...
        mat_dict = nb3_sn_wp.make_mat_dict(300)
        wp_material = Material(*mat_dict.values())
        self.materials["Nb3SnWP"] = wp_material

    def define_geometry(self):
        """
        Set up a FiniteElementModel and define the geometry.
        """
        self.model = FiniteElementModel()
        self.add_tf_coils()
        # self.add_pf_coils()
        self.add_intercoil_structures()
        self.add_gravity_supports()

    def add_tf_coils(self):
        """
        Adds the TF coil geometry to the FE model
        """
        self._tf_elem_ids = []  # Reset
        loop = self.architect.tfl["cl_fe"].copy()
        loop.close()

        shell = self.architect.get_tf_xsection(min(loop["x"]), 0, 1)

        nose_xs = AnalyticalShellComposite(
            shell, [FORGED_JJ1, self.materials["Nb3SnWP"]]
        )

        x_min = np.min(self.architect.tfl["cl_fe"].x)
        nose_nodes = np.where(np.isclose(loop.x, x_min))[0]
        arg = np.where(np.diff(nose_nodes) != 1)[0][0]
        last_down_nose = nose_nodes[arg]

        prev_id = None  # Need this to catch closed loops

        for i, (x, z) in enumerate(zip(loop["x"], loop["z"])):
            next_id = self.model.add_node(x, 0, z)
            if i == 0:
                # Just adding a node, no element
                node2 = self.model.geometry.nodes[next_id]
                first = next_id
            else:
                # Add an element

                # Get element mid-point
                node1 = self.model.geometry.nodes[prev_id]
                node2 = self.model.geometry.nodes[next_id]
                mid = get_midpoint(node1, node2)

                # Get the direction of the element
                if node1.z >= 0:
                    dz = -1
                else:
                    dz = 1

                if i in nose_nodes:
                    x_section = nose_xs.copy()
                    if i > last_down_nose and i != nose_nodes[-1] + 1:
                        x_section.geometry[0].rotate(180, p1=[0, 0, 0], p2=[dz, 0, 0])

                else:
                    shell = self.architect.get_tf_xsection(mid[0], mid[2], dz)
                    x_section = AnalyticalShellComposite(
                        shell,
                        [CAST_EC1, self.materials["Nb3SnWP"]],
                    )

                # Add the element to the model
                elem_id = self.model.add_element(prev_id, next_id, x_section)

                # Store element id, so we can apply loads correctly later
                self._tf_elem_ids.append(elem_id)

            if x <= self.x_nose:
                # Support nose nodes (free displacement in Z)
                self.model.add_support(
                    node2.id_number, True, True, False, True, True, True
                )

            # Increment node ID
            prev_id = next_id
        last = prev_id
        self._tf_node_ids = list(range(first, last))

    def add_pf_coils(self):
        """
        Adds the PF coil geometry to the FE model
        """
        # NOTE: for the moment, we don't explicitly add PF coils
        pf_coils = self.architect.pf.get_PF_names()

        for name in pf_coils:
            # Get the coil
            coil = self.architect.pf.coils[name]
            # Make a loop
            x, y = circle_seg(coil.x, npoints=2 * int(self.coilcage.n_TF))
            loop = Loop(x, y, coil.z)

            # Make a cross-section (y-z centred at 0, 0)
            winding_pack = Loop(y=coil.x, z=coil.z)
            winding_pack.close()
            centroid = np.array(winding_pack._point_23d(winding_pack.centroid))
            winding_pack.translate(-centroid)

            # TODO: Hook up casing thickness parameter
            shell = Shell(winding_pack.offset(-0.1), winding_pack)
            # TODO: NbTi material properties
            section = CompositeCrossSection(
                [shell, shell.inner],
                [self.materials["SS316"], self.materials["Nb3SnWP"]],
            )
            # Add elements
            self.model.add_loop(loop, section)

    def add_gravity_supports(self, n_gs=7):
        """
        Add the gravity supports to the FE model.
        """
        # Track the node numbers added
        n_before = self.model.geometry.n_nodes

        g_support = self.architect.geom["feed 3D CAD"]["Gsupport"]

        if g_support["gs_type"] == "JT60SA":
            self._add_JT60SA_gravity_support(g_support, n_gs)
        elif g_support["gs_type"] == "ITER":
            self._add_ITER_gravity_support(g_support, n_gs)

        # Label GS Nodes
        # Connect the GS to the TF loop
        gs_top = [g_support["Xo"], 0, g_support["zbase"]]
        gs_id = self.model.geometry.find_node(*gs_top)

        # Find the closest Node to where the GS should be connected
        top_id = self._connect_support(gs_top[0])
        # Move that Node to be directly above the GS
        vertical_line = Loop(
            x=[g_support["Xo"], g_support["Xo"]],
            z=[g_support["zbase"], g_support["zbase"] + VERY_BIG],
        )
        x_inter, z_inter = get_intersect(self.architect.tfl["cl_fe"], vertical_line)
        arg_inter = np.argmin(z_inter)
        x_inter = x_inter[arg_inter]
        z_inter = z_inter[arg_inter]
        dx = x_inter - self.model.geometry.nodes[top_id].x
        dz = z_inter - self.model.geometry.nodes[top_id].z

        self.model.geometry.move_node(top_id, dx=dx, dy=0, dz=dz)

        xsection = RectangularBeam(
            self.tf.params.tf_wp_depth, self.tf.params.tf_wp_depth
        )
        self.model.add_element(gs_id, top_id, xsection, FORGED_SS316_LN)

        n_after = self.model.geometry.n_nodes
        self._gs_node_ids = list(range(n_before, n_after))

    def _add_JT60SA_gravity_support(self, g_support, n_gs):
        """
        Add a JT60SA-like gravity support structure to the FiniteElementModel.
        """
        for sign in [1, -1]:
            y_gs = np.linspace(0, sign * g_support["yfloor"], n_gs)
            z_gs = np.linspace(g_support["zbase"], g_support["zfloor"], n_gs)

            gs = Loop(x=g_support["Xo"], y=y_gs, z=z_gs)
            if sign < 0:
                # This is because the Loop will force the straight line to be
                # ccw
                gs.reverse()

            cross_section = CircularHollowBeam(
                g_support["rtube"] - 2 * g_support["ttube"], g_support["rtube"]
            )
            self.model.add_loop(gs, cross_section, FORGED_SS316_LN)

            gs_node_id = self.model.geometry.nodes[-1].id_number
            self.model.add_support(gs_node_id, True, True, True, True, False, True)

    def _add_ITER_gravity_support(self, g_support, n_gs):
        """
        Add an ITER-like gravity support structure to the FiniteElementModel.
        """
        x_gs = np.ones(n_gs) * g_support["Xo"]
        y_gs = np.ones(n_gs) * g_support["yfloor"]
        z_gs = np.linspace(g_support["zbase"], g_support["zfloor"], n_gs)
        gs = Loop(x_gs, y_gs, z_gs)
        # Make the ITER GS cross-section
        plates = []
        n_plates = g_support["width"] / (2 * g_support["plate_width"])
        delta = (n_plates - int(n_plates)) * 2 * g_support["plate_width"]
        x_left = -g_support["width"] / 2 + delta
        for i in range(int(n_plates)):
            plate = RectangularBeam(g_support["plate_width"], g_support["width"])
            plate.translate([0, x_left + i * 2 * g_support["plate_width"], 0])
            plates.append(plate)

        cross_section = MultiCrossSection(plates, centroid=[0, 0])
        cross_section.rotate(90)
        self.model.add_loop(gs, cross_section, FORGED_SS316_LN)

        gs_node_id = self.model.geometry.nodes[-1].id_number
        self.model.add_support(gs_node_id, True, True, True, True, True, True)

    def _connect_support(self, x_gs):
        """
        Find the closest TF node to the top of the GS.
        """
        lower_args = np.where(self.architect.tfl["cl_fe"].z < 0)[0]
        x_vals = self.architect.tfl["cl_fe"].x[lower_args]
        arg = int(np.argmin(abs(x_vals - x_gs)))
        arg = int(lower_args[arg])
        point = self.architect.tfl["cl_fe"][arg]
        return self.model.geometry.find_node(*point)

    def _connect_to_tf(self, x, y, z):
        """
        Finds the closest node in the tf to a location, and adds an element
        """
        arg = self.architect.tfl["cl_fe"].argmin([x, z])
        point = self.architect.tfl["cl_fe"][int(arg)]
        return self.model.geometry.find_node(*point)

    def add_intercoil_structures(self, n_ois=3):
        """
        Add the inter-coil structures to the FiniteElementModel. Include a
        cyclic symmetry boundary condition.

        Parameters
        ----------
        n_ois: int (default = 3)
            The number of nodes along the OIS beam length.
        """
        structures = [
            v.copy() for k, v in self.architect.geom.items() if k.startswith("OIS")
        ]

        theta = 2 * np.pi / self.tf.params.n_TF

        for ois in structures:
            # Find closest node on TF centreline (approximation)
            tfpoint = [ois.centroid[0], 0, ois.centroid[1]]
            tf_id = self._connect_to_tf(*tfpoint)
            tfpoint = self.model.geometry.node_xyz[tf_id]
            # Get OIS length
            min_length = ois.get_min_length()
            max_length = (ois.length - 2 * min_length) / 2
            xsection = RectangularBeam(min_length, max_length)

            # Get rotation angle and rotate cross-section about its centroid
            v1 = np.array([0, 0, 1])
            tf1 = np.array(self.model.geometry.node_xyz[tf_id - 1])
            tf2 = np.array(self.model.geometry.node_xyz[tf_id + 1])
            v2 = tf2 - tf1
            angle = get_angle_between_vectors(v1, v2, signed=False)
            if tfpoint[2] > 0:
                angle *= -1

            xsection.rotate(angle)

            sym_nodes = []
            for side in [-1, 1]:
                # on both sides of the TF coil
                # first rotate by one sector
                oppoint = qrotate(
                    tfpoint, theta=side * theta, p1=[0, 0, 0], p2=[0, 0, 1]
                )[0]
                # then find midpoint between the two points
                mid_x = (oppoint[0] + tfpoint[0]) / 2
                mid_y = (oppoint[1] + tfpoint[1]) / 2
                x = np.linspace(tfpoint[0], mid_x, n_ois)
                y = np.linspace(0, mid_y, n_ois)
                z = tfpoint[2]

                loop = Loop(x, y, z)

                self.model.add_loop(loop, xsection, FORGED_SS316_LN)

                mid_node = self.model.geometry.find_node(mid_x, mid_y, z)
                sym_nodes.append(mid_node)

            # Add a cyclical symmetry boundary condition
            self.model.apply_cyclic_symmetry(*sym_nodes, p1=[0, 0, 0], p2=[0, 0, 1])

    def apply_loads(self, eq):
        """
        Apply all the loads to the model for a given equilibrium.

        Parameters
        ----------
        eq: Equilibrium
            The equilibrium for which to add the J x B and vertical forces
        """
        self.model.add_gravity_loads()
        self.add_pf_loads(eq)
        self.add_tf_loads(eq)

    def add_tf_loads(self, eq):
        """
        Add the magnetic loads onto the TF coil.

        Parameters
        ----------
        eq: Equilibrium
            The equilibrium for which to add the J x B forces
        """
        # Rotate the point to be on the cage to calculate TF forces
        theta = np.pi / self.tf.params.n_TF
        rotation = rotate_matrix(theta, axis="z")
        for elem_id in self._tf_elem_ids:
            element = self.model.geometry.elements[elem_id]
            mid_point = element.mid_point
            mid_point = np.dot(mid_point, rotation)

            i_tf = self.tf.params.I_tf
            d_l = element.space_vector / element.length

            F = self.coilcage.tf_forces(mid_point, i_tf * d_l, eq.Bx, eq.Bz)
            for j, c in enumerate(["Fx", "Fy", "Fz"]):
                self.model.add_distributed_load(elem_id, F[j] / element.length, c)

    def add_pf_loads(self, eq):
        """
        Add the vertical forces from the PF coils onto the closest node in the
        TF loop.

        Parameters
        ----------
        eq: Equilibrium
            The equilibrium for which to add the vertical PF forces
        """
        # Get PF forces from equilibrium / divide by n_TF
        f_z = eq.get_forces().T[1, eq.coilset.n_CS :] / self.tf.params.n_TF
        pf_names = eq.coilset.get_PF_names()
        for i, coil_name in enumerate(pf_names):
            coil = eq.coilset.coils[coil_name]
            coil_point = [coil.x, 0, coil.z]

            # Find closest node on the TF loop
            node_id = self._connect_to_tf(*coil_point)

            # Add the vertical force to the closest point on the TF
            self.model.load_case.add_node_load(node_id, f_z[i], "Fz")

    def pattern(self):
        """
        Pattern the FE model, destroying the specified cyclic symmetry conditions.
        """
        # apply loads so they get patterned too
        self.model._apply_load_case(self.model.load_case)
        self.model.clear_load_case()  # To avoid duplicate loads on the first sector
        self.model.geometry = cyclic_pattern(
            self.model.geometry,
            np.array([0, 0, 1]),
            360 / self.tf.params.n_TF,
            self.tf.params.n_TF,
        )
        # Deconstruct cyclic_symmetry
        self.model.cycle_sym = None
        self.model.cycle_sym_ids = []

    def solve(self, **kwargs):
        """
        Solve the strucutural problem for each of the input equilibria.

        Returns
        -------
        results: List[Type[Result]]
            The list of FE Result objects
        """
        results = []
        for equilibrium in self.equilibria:
            self.model.clear_loads()
            self.apply_loads(equilibrium)
            result = self.model.solve(**kwargs)
            results.append(result)

        self.results = results  # Stash / over-write results
        return results

    def plot(self, ax=None):
        """
        Plot the coil cage structural model.
        """
        return self.model.plot(ax=ax)
