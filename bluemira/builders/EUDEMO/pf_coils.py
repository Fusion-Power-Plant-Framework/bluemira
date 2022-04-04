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
EU-DEMO specific builder for PF coils
"""

from typing import List, Optional

import numpy as np

import bluemira.utilities.plot_tools as bm_plot_tools
from bluemira.base.builder import BuildConfig, Builder
from bluemira.base.components import Component
from bluemira.base.config import Configuration
from bluemira.base.error import BuilderError
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.builders.pf_coils import PFCoilBuilder
from bluemira.equilibria.coils import CoilSet
from bluemira.geometry.constants import VERY_BIG
from bluemira.geometry.tools import boolean_cut, distance_to, split_wire
from bluemira.magnetostatics.baseclass import SourceGroup
from bluemira.magnetostatics.circular_arc import CircularArcCurrentSource
from bluemira.utilities.positioning import PathInterpolator, PositionMapper


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
        return self._field_solver.field(x, y, z)


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
    _params: Configuration

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

        Returns
        -------
        component: Component
            The component grouping the results in the xy plane.
        """
        xy_comps = []
        comp: PFCoilBuilder
        for comp in self.sub_components:
            xy_comps.append(comp.build_xy())
        component = Component("xy", children=xy_comps)
        bm_plot_tools.set_component_view(component, "xy")
        return component

    def build_xz(self):
        """
        Build the x-z components of the PF coils.

        Returns
        -------
        component: Component
            The component grouping the results in the xz plane.
        """
        xz_comps = []
        comp: PFCoilBuilder
        for comp in self.sub_components:
            xz_comps.append(comp.build_xz())
        component = Component("xz", children=xz_comps)
        bm_plot_tools.set_component_view(component, "xz")
        return component

    def build_xyz(self, degree: float = 360.0):
        """
        Build the x-y-z components of the PF coils.

        Parameters
        ----------
        degree: float
            The angle [°] around which to build the components, by default 360.0.

        Returns
        -------
        component: Component
            The component grouping the results in 3D (xyz).
        """
        xyz_comps = []
        comp: PFCoilBuilder
        for comp in self.sub_components:
            xyz_comps.append(comp.build_xyz(degree=degree))
        component = Component("xyz", children=xyz_comps)
        return component

    def _make_field_solver(self):
        """
        Make a magnetostatics solver for the field from the PF coils.
        """
        sources = []
        for coil in self._coilset.coils.values():
            sources.append(
                CircularArcCurrentSource(
                    [0, 0, coil.z],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    coil.dx,
                    coil.dz,
                    coil.x,
                    2 * np.pi,
                    coil.current,
                )
            )

        field_solver = SourceGroup(sources)
        return field_solver


def make_coil_mapper(track, exclusion_zones, coils):
    """
    Break a track down into individual interpolator segments, incorporating exclusion
    zones and mapping them to coils.

    Parameters
    ----------
    track: BluemiraWire
        Full length interpolator track for PF coils
    exclusion_zones: List[BluemiraFace]
        List of exclusion zones
    coils: List[Coil]
        List of coils

    Returns
    -------
    mapper: PositionMapper
        Position mapper for coil position interpolation
    """
    # Break down the track into subsegments
    segments = boolean_cut(track, exclusion_zones)

    # Sort the coils into the segments
    coil_bins = [[] for _ in range(len(segments))]
    for i, coil in enumerate(coils):
        distances = [distance_to([coil.x, 0, coil.z], seg)[0] for seg in segments]
        coil_bins[np.argmin(distances)].append(i)

    # Check if multiple coils are on the same segment and split the segments
    new_segments = []
    for segment, bin in zip(segments, coil_bins):
        if len(bin) < 1:
            bluemira_warn("There is a segment of the track which has no coils on it.")
        elif len(bin) == 1:
            new_segments.append(PathInterpolator(segment))
        else:
            coils = [coils[i] for i in bin]
            l_values = [
                segment.parameter_at([c.x, 0, c.z], tolerance=VERY_BIG) for c in coils
            ]
            split_values = l_values[:-1] + 0.5 * np.diff(l_values)

            sub_segs = []
            for i, split in enumerate(split_values):
                sub_seg, segment = split_wire(segment, segment.value_at(alpha=split))
                if sub_seg:
                    sub_segs.append(PathInterpolator(sub_seg))

                if i == len(split_values) - 1:
                    if segment:
                        sub_segs.append(PathInterpolator(segment))

            new_segments.extend(sub_segs)

    return PositionMapper(new_segments)
