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
Builder for the PF coils
"""

from typing import List, Optional

from bluemira.base.components import PhysicalComponent, Component
from bluemira.base.builder import Builder, BuildConfig
from bluemira.base.error import BuilderError, ComponentError
from bluemira.base.parameter import ParameterFrame
from bluemira.geometry.tools import revolve_shape, make_circle, offset_wire
from bluemira.geometry.wire import BluemiraWire
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.parameterisations import PictureFrame
from bluemira.equilibria.coils import CoilSet
from bluemira.display.palettes import BLUE_PALETTE
import bluemira.utilities.plot_tools as bm_plot_tools


class PFCoilsComponent(Component):
    """
    Poloidal field coils component, with a solver for the magnetic field from all of the
    PF coils.

    Parameters
    ----------
    name: str
        Name of the component
    parent: Optional[Component] = None
        Parent component
    children: Optional[List[Component]] = None
        List of child components
    field_solver: Optional[CurrentSource]
        Magnetic field solver
    """

    def __init__(
        self,
        name: str,
        parent: Optional[Component] = None,
        children: Optional[List[Component]] = None,
        field_solver=None,
    ):
        super().__init__(name, parent=parent, children=children)
        self._field_solver = field_solver

    def field(self, x, y, z):
        """
        Calculate the magnetic field due to the TF coils at a set of points.
        Parameters
        ----------
        x: Union[float, np.array]
            The x coordinate(s) of the points at which to calculate the field
        y: Union[float, np.array]
            The y coordinate(s) of the points at which to calculate the field
        z: Union[float, np.array]
            The z coordinate(s) of the points at which to calculate the field
        Returns
        -------
        field: np.array
            The magnetic field vector {Bx, By, Bz} in [T]
        """
        # TODO: Implement PF rotation to 3-D Cartesian coordinates
        return self._field_solver.field(x, y, z)


class PFCoilBuilder:
    """
    Builder for a single PF coil
    """

    def __init__(self, coil, r_corner, tk_insulation, tk_casing, ctype):
        self.coil = coil
        self.r_corner = r_corner
        self.tk_insulation = tk_insulation
        self.tk_casing = tk_casing
        self.ctype = ctype

    def build_xy(self):
        """
        Build the x-y representation of a PF coil.
        """
        r_in = self.coil.x - self.coil.dx
        r_out = self.coil.x + self.coil.dx
        c1 = make_circle(r_out)
        c2 = make_circle(r_in)

        wp = PhysicalComponent("winding pack", BluemiraFace([c1, c2]))
        idx = 0 if self.ctype == "CS" else 1
        wp.plot_options.face_options["color"] = BLUE_PALETTE["PF"][idx]

        r_in -= self.tk_insulation
        c3 = make_circle(r_in)
        inner_ins = PhysicalComponent("inner", BluemiraFace([c2, c3]))
        inner_ins.plot_options.face_options["color"] = BLUE_PALETTE["PF"][3]

        r_out += self.tk_insulation
        c4 = make_circle(r_out)
        outer_ins = PhysicalComponent("outer_ins", BluemiraFace([c4, c1]))
        outer_ins.plot_options.face_options["color"] = BLUE_PALETTE["PF"][3]

        ins = Component(
            name="ground insulation",
            children=[inner_ins, outer_ins],
        )

        r_in -= self.tk_casing
        c5 = make_circle(r_in)
        inner_cas = PhysicalComponent("inner", BluemiraFace([c3, c5]))
        inner_cas.plot_options.face_options["color"] = BLUE_PALETTE["PF"][2]

        r_out += self.tk_casing
        c6 = make_circle(r_out)
        outer_cas = PhysicalComponent("outer", BluemiraFace([c6, c4]))
        outer_cas.plot_options.face_options["color"] = BLUE_PALETTE["PF"][2]
        casing = Component(
            "casing",
            children=[inner_cas, outer_cas],
        )

        component = Component(self.coil.name, children=[wp, ins, casing])
        return component

    def build_xz(self):
        """
        Build the x-z representation of a PF coil.
        """
        x_in = self.coil.x - self.coil.dx
        x_out = self.coil.x + self.coil.dx
        z_up = self.coil.z + self.coil.dz
        z_down = self.coil.z - self.coil.dz

        shape = PictureFrame(
            {
                "x1": {"value": x_in, "fixed": True},
                "x2": {"value": x_out, "fixed": True},
                "z1": {"value": z_up, "fixed": True},
                "z2": {"value": z_down, "fixed": True},
                "ri": {"value": self.r_corner, "fixed": True},
                "ro": {"value": self.r_corner, "fixed": True},
            }
        ).create_shape()
        wp = PhysicalComponent("winding pack", BluemiraFace(shape))
        idx = 0 if self.ctype == "CS" else 1
        wp.plot_options.face_options["color"] = BLUE_PALETTE["PF"][idx]

        ins_shape = offset_wire(shape, self.tk_insulation)

        ins = PhysicalComponent("ground insulation", BluemiraFace([ins_shape, shape]))
        ins.plot_options.face_options["color"] = BLUE_PALETTE["PF"][3]
        cas_shape = offset_wire(ins_shape, self.tk_casing)

        casing = PhysicalComponent("casing", BluemiraFace([cas_shape, ins_shape]))
        casing.plot_options.face_options["color"] = BLUE_PALETTE["PF"][2]
        return Component(self.coil.name, children=[wp, ins, casing])

    def build_xyz(self):
        """
        Build the x-y-z representation of a PF coil.
        """
        # I doubt this is floating-point safe to collisions...
        c_xz = self.build_xz()
        components = []
        for c in c_xz.children:
            shape = revolve_shape(c.shape, degree=360)
            c_xyz = PhysicalComponent(c.name, shape)
            c_xyz.display_cad_options.color = c.plot_options.face_options["color"]
            components.append(c_xyz)

        return Component(self.coil.name, children=components)


class PFCoilsBuilder(Builder):
    """
    Builder for the PF Coils.
    """

    _required_params: List[str] = [
        "tk_pf_insulation",
        "tk_pf_casing",
        "tk_cs_insulation",
        "tk_cs_casing",
        "r_pf_corner",
        "r_cs_corner",
    ]
    _required_config: List[str] = []
    _params: ParameterFrame

    def _extract_config(self, build_config: BuildConfig):
        super()._extract_config(build_config)

        if self._runmode.name.lower() == "read":
            if build_config.get("eqdsk_path") is None:
                raise BuilderError(
                    "Must supply eqdsk_path in build_config when using 'read' mode."
                )
            self._eqdsk_path = build_config["eqdsk_path"]

    def reinitialise(self, params, **kwargs) -> None:
        """
        Initialise the state of this builder ready for a new run.
        Parameters
        ----------
        params: Dict[str, Any]
            The parameterisation containing at least the required params for this
            Builder.
        """
        super().reinitialise(params, **kwargs)

        self._reset_params(params)
        self._coilset = None

    def run(self, *args):
        """
        Build PF coils from a design optimisation problem.
        """
        pass

    def read(self, **kwargs):
        """
        Build PF coils from a equilibrium file.
        """
        self._coilset = CoilSet.from_eqdsk(self._eqdsk_path)

    def mock(self, coilset):
        """
        Build PF coils from a CoilSet.
        """
        self._coilset = coilset

    def build(self, label: str = "PF Coils", **kwargs) -> PFCoilsComponent:
        """
        Build the PF Coils component.
        Returns
        -------
        component: PFCoilsComponent
            The Component built by this builder.
        """
        super().build(**kwargs)

        self.sub_components = []
        for coil in self._coilset.coils.values():
            if coil.ctype == "PF":
                r_corner = self.params.r_pf_corner
                tk_ins = self.params.tk_pf_insulation
                tk_cas = self.params.tk_pf_casing
            elif coil.ctype == "CS":
                r_corner = self.params.r_cs_corner
                tk_ins = self.params.tk_cs_insulation
                tk_cas = self.params.tk_cs_casing
            else:
                raise BuilderError(f"Unrecognised coil type {coil.ctype}.")

            sub_comp = PFCoilBuilder(coil, r_corner, tk_ins, tk_cas, coil.ctype)
            self.sub_components.append(sub_comp)

        field_solver = self._make_field_solver()
        component = PFCoilsComponent(self.name, field_solver=field_solver)

        component.add_child(self.build_xz())
        component.add_child(self.build_xy())
        component.add_child(self.build_xyz())
        return component

    def build_xy(self):
        """
        Build the x-y components of the PF coils.
        """
        xy_comps = []
        for comp in self.sub_components:
            xy_comps.append(comp.build_xy())
        component = Component("xy", children=xy_comps)
        bm_plot_tools.set_component_plane(component, "xy")
        return component

    def build_xz(self):
        """
        Build the x-z components of the PF coils.
        """
        xz_comps = []
        for comp in self.sub_components:
            xz_comps.append(comp.build_xz())
        component = Component("xz", children=xz_comps)
        bm_plot_tools.set_component_plane(component, "xz")
        return component

    def build_xyz(self):
        """
        Build the x-y-z components of the PF coils.
        """
        xyz_comps = []
        for comp in self.sub_components:
            xyz_comps.append(comp.build_xyz())
        component = Component("xyz", children=xyz_comps)
        return component

    def _make_field_solver(self):
        return None


if __name__ == "__main__":
    from bluemira.equilibria.coils import Coil, CoilSet

    coil = Coil(4, 4, 10e6, dx=0.5, dz=0.8, name="PF_1")
    coil2 = Coil(10, 10, 1e5, dx=0.1, dz=0.5, name="PF_2")
    coilset = CoilSet([coil, coil2])
    builder = PFCoilBuilder(coil, 0.05, 0.05, 0.1)

    c_xy = builder.build_xy()
    c_xz = builder.build_xz()
    c_xyz = builder.build_xyz()

    c_xy.plot_2d()
    c_xz.plot_2d()
    c_xyz.show_cad()

    builder = PFCoilsBuilder(coilset, 0.05, 0.05, 0.1)
    c_xy = builder.build_xy()
    c_xz = builder.build_xz()
    c_xyz = builder.build_xyz()

    c_xy.plot_2d()
    c_xz.plot_2d()
    c_xyz.show_cad()
