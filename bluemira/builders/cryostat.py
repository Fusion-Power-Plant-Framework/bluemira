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
Cryostat builder
"""


class Cryostat(ComponentManager):
    pass


class CryostatDesignerParams(ParameterFrame):
    x_g_support: Parameter[float]
    x_gs_kink_diff: Parameter[float]  # TODO add to Parameter default = 2
    g_cr_ts: Parameter[float]
    tk_cr_vv: Parameter[float]
    well_depth: Parameter[float]  # TODO add to Parameter default = 5 chickens
    z_gs: Parameter[
        float
    ]  # TODO add to Parameter default (z gravity support) = -15 chickens


class CryostatBuilderParams(ParameterFrame):
    n_TF: Parameter[int]


class CryostatDesigner(Designer[BluemiraWire]):

    param_cls: Type[CryostatDesignerParams] = CyrostatDesignerParams

    def __init__(
        self,
        params: Union[ParameterFrame, Dict],
        geom_parameterisation: str,
        variable_map: Dict[str, str],
        optimisation_config: Dict,
    ):
        super().__init__(params)
        self.geom_parameterisation = geom_parameterisation
        self.variable_map = variable_map
        self.optimisation_config = optimisation_config
        self.lcfs = lcfs
        self.min_dist_to_lcfs = min_dist_to_lcfs

    def run_xz(self):
        # Cryostat VV
        x_in = 0
        x_out, z_top = self._get_extrema()

        x_gs_kink = (
            self.params.x_g_support - self.params.x_gs_kink_diff
        )  # TODO: Get from a parameter
        z_mid = self.params.z_gs - self.params.g_cr_ts
        z_bot = z_mid - self.params.well_depth
        tk = self.params.tk_cr_vv.value

        x_inner = [x_in, x_out, x_out, x_gs_kink, x_gs_kink, x_in]
        z_inner = [z_top, z_top, z_mid, z_mid, z_bot, z_bot]
        x_outer = [x_in, x_gs_kink + tk, x_gs_kink + tk, x_out + tk, x_out + tk, x_in]
        z_outer = [
            z_bot - tk,
            z_bot - tk,
            z_mid - tk,
            z_mid - tk,
            z_top + tk,
            z_top + tk,
        ]
        x = np.concatenate([x_inner, x_outer])
        z = np.concatenate([z_inner, z_outer])

        shape = BluemiraFace(make_polygon({"x": x, "y": 0, "z": z}, closed=True))

    def run_xy(self):
        r_in, _ = self._get_extrema()
        r_out = r_in + self.params.tk_cr_vv
        inner = make_circle(radius=r_in)
        outer = make_circle(radius=r_out)

        return BluemiraFace([outer, inner])

    def _get_extrema(self):
        bound_box = self._cts_xz.bounding_box
        z_max = bound_box.z_max
        x_max = bound_box.x_max
        x_out = x_max + self.params.g_cr_ts
        z_top = z_max + self.params.g_cr_ts
        return x_out, z_top


class CryostatBuilder(Builder):
    CRYO = "Cryostat VV"
    param_cls: Type[CryostatBuilderParams] = CyrostatBuilderParams

    def build(self) -> Cyrostat:
        component = super().build()

        self._xz_cross_section = self.designer.run_xz()

        component.add_child(Component("xz", children=[self.build_xz()]))
        component.add_child(Component("xy", children=[self.build_xy()]))
        component.add_child(Component("xyz", children=self.build_xyz()))
        return Cyrostat(component)

    def build_xz(self):
        cryostat_vv = PhysicalComponent(self.CRYO, self._xz_cross_section)
        cryostat_vv.plot_options.face_options["color"] = BLUE_PALETTE["CR"][0]
        return cryostat_vv

    def build_xy(self):
        shape = self.designer.run_xy()

        cryostat_vv = PhysicalComponent(self.CRYO, shape)
        cryostat_vv.plot_options.face_options["color"] = BLUE_PALETTE["CR"][0]
        return cryostat_vv

    def build_xyz(self, degree=360) -> List[PhysicalComponent]:
        sector_degree = 360 / self.params.n_TF.value
        n_sectors = max(1, int(degree // int(sector_degree)))

        shape = revolve_shape(
            self._xz_cross_section,
            base=(0, 0, 0),
            direction=(0, 0, 1),
            degree=sector_degree,
        )

        cryostat_vv = PhysicalComponent(self.CRYO, shape)
        cryostat_vv.display_cad_options.color = BLUE_PALETTE["CR"][0]
        return circular_pattern_component(
            cryostat_vv, n_sectors, degree=sector_degree * n_sectors
        )
