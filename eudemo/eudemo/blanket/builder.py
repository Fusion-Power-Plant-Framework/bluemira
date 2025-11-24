# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
EUDEMO builder for blanket
"""

from dataclasses import dataclass

from bluemira.base.builder import Builder, Component
from bluemira.base.components import PhysicalComponent
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.builders.tools import (
    apply_component_display_options,
    circular_pattern_component,
    get_n_sectors,
    pattern_revolved_silhouette,
)
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.placement import BluemiraPlacement
from bluemira.geometry.tools import (
    boolean_cut,
    boolean_fragments,
    make_polygon,
    offset_wire,
    slice_shape,
)


@dataclass
class BlanketBuilderParams(ParameterFrame):
    """
    Blanket builder parameters
    """

    n_TF: Parameter[int]
    n_bb_inboard: Parameter[int]
    n_bb_outboard: Parameter[int]
    c_rm: Parameter[float]
    tk_bb_fw_ib: Parameter[float]
    tk_bb_bz_ib: Parameter[float]
    tk_bb_fw_ob: Parameter[float]
    tk_bb_bz_ob: Parameter[float]


class BlanketBuilder(Builder):
    """
    Blanket builder

    Parameters
    ----------
    params:
        the parameter frame
    build_config:
        the build config
    blanket_silhouette:
        breeding blanket silhouette
    """

    BB = "BB"
    IBS = "IBS"
    OBS = "OBS"
    FW = "FW"
    BZ = "BZ"
    MANIFOLD = "MANIFOLD"
    param_cls: type[BlanketBuilderParams] = BlanketBuilderParams
    params: BlanketBuilderParams

    def __init__(
        self,
        params: BlanketBuilderParams | dict,
        build_config: dict,
        ib_silhouette: BluemiraFace,
        ob_silhouette: BluemiraFace,
        panel_points: Coordinates,
    ):
        super().__init__(params, build_config)
        self.ib_silhouette = ib_silhouette
        self.ob_silhouette = ob_silhouette
        self.panel_points = panel_points

    def build(self) -> Component:
        """
        Build the blanket component.

        Returns
        -------
        :
            The component tree
        """
        ib_fw, ib_bz, ib_manifold = self._subdivide_poloidally(
            self.ib_silhouette,
            self.params.tk_bb_fw_ib.value,
            self.params.tk_bb_bz_ib.value,
        )
        ob_fw, ob_bz, ob_manifold = self._subdivide_poloidally(
            self.ob_silhouette,
            self.params.tk_bb_fw_ob.value,
            self.params.tk_bb_bz_ob.value,
        )

        segments = (
            self.get_segments(ib_fw, self.FW, inboard=True, color_index=0)
            + self.get_segments(ib_bz, self.BZ, inboard=True, color_index=1)
            + self.get_segments(ib_manifold, self.MANIFOLD, inboard=True, color_index=2)
            + self.get_segments(ob_fw, self.FW, inboard=False, color_index=0)
            + self.get_segments(ob_bz, self.BZ, inboard=False, color_index=1)
            + self.get_segments(ob_manifold, self.MANIFOLD, inboard=False, color_index=2)
        )

        return self.component_tree(
            xz=[self.build_xz(ib_fw, ib_bz, ib_manifold, ob_fw, ob_bz, ob_manifold)],
            xy=self.build_xy(segments),
            xyz=self.build_xyz(segments, degree=0),
        )

    def build_xz(self, ib_fw, ib_bz, ib_manifold, ob_fw, ob_bz, ob_manifold):
        """
        Build the x-z components of the blanket.

        Returns
        -------
        :
            The xz component
        """
        ib_fw = PhysicalComponent(
            f"{self.IBS}_{self.FW}", ib_fw, material=self.get_material(self.FW)
        )
        ib_bz = PhysicalComponent(
            f"{self.IBS}_{self.BZ}", ib_bz, material=self.get_material(self.BZ)
        )
        ib_manifold = PhysicalComponent(
            f"{self.IBS}_{self.MANIFOLD}",
            ib_manifold,
            material=self.get_material(self.MANIFOLD),
        )
        apply_component_display_options(ib_fw, color=BLUE_PALETTE[self.BB][0])
        apply_component_display_options(ib_bz, color=BLUE_PALETTE[self.BB][1])
        apply_component_display_options(ib_manifold, color=BLUE_PALETTE[self.BB][2])

        ob_fw = PhysicalComponent(
            f"{self.OBS}_{self.FW}", ob_fw, material=self.get_material(self.FW)
        )
        ob_bz = PhysicalComponent(
            f"{self.OBS}_{self.BZ}", ob_bz, material=self.get_material(self.BZ)
        )
        ob_manifold = PhysicalComponent(
            f"{self.OBS}_{self.MANIFOLD}",
            ob_manifold,
            material=self.get_material(self.MANIFOLD),
        )
        apply_component_display_options(ob_fw, color=BLUE_PALETTE[self.BB][0])
        apply_component_display_options(ob_bz, color=BLUE_PALETTE[self.BB][1])
        apply_component_display_options(ob_manifold, color=BLUE_PALETTE[self.BB][2])

        return Component(
            self.BB, children=[ib_fw, ib_bz, ib_manifold, ob_fw, ob_bz, ob_manifold]
        )

    def build_xy(self, segments: list[PhysicalComponent]):
        """
        Build the x-y components of the blanket.

        Returns
        -------
        :
            The xy components
        """
        xy_plane = BluemiraPlacement.from_3_points([0, 0, 0], [1, 0, 0], [0, 1, 0])

        slices = []
        for segment in segments:
            single_slice = PhysicalComponent(
                segment.name,
                BluemiraFace(slice_shape(segment.shape, xy_plane)[0]),
                material=segment.material,
            )
            apply_component_display_options(
                single_slice, color=segment.display_cad_options.color
            )
            slices.append(single_slice)

        return circular_pattern_component(
            Component(self.BB, children=slices), self.params.n_TF.value
        )

    def build_xyz(self, segments: list[PhysicalComponent], degree: float = 360.0):
        """
        Build the x-y-z components of the blanket.

        Returns
        -------
        :
            The xyz components
        """
        sector_degree, n_sectors = get_n_sectors(self.params.n_TF.value, degree)

        return circular_pattern_component(
            Component(self.BB, children=segments),
            n_sectors,
            degree=sector_degree * n_sectors,
        )

    @staticmethod
    def _find_union_face(silhouette: BluemiraFace, cut: BluemiraFace) -> BluemiraFace:
        # This is perhaps not entirely robust, but works for now
        fragments = boolean_fragments([silhouette, cut])[1][0]
        return fragments[0]

    def _subdivide_poloidally(
        self, silhouette: BluemiraFace, fw_thickness: float, bz_thickness: float
    ):
        base_wire = make_polygon(self.panel_points.T)
        base_wire.close()
        fw_cut_wire = offset_wire(base_wire, fw_thickness)
        bz_cut_wire = offset_wire(fw_cut_wire, bz_thickness)
        fw_cut = BluemiraFace(fw_cut_wire)
        bz_cut = BluemiraFace(bz_cut_wire)
        fw = self._find_union_face(silhouette, fw_cut)
        bz = self._find_union_face(silhouette, bz_cut)
        manifold = boolean_cut(silhouette, bz_cut)[0]

 
        dodgy_wire = offset_wire(bz.wires[0], -0.002)  # avoid coincident faces
        dodgy_wire = offset_wire(dodgy_wire, 0.001)  # Superstition
        bz = BluemiraFace(dodgy_wire)

        return fw, bz, manifold

    def get_segments(
        self, silhouette: BluemiraFace, sub_name: str, *, inboard: bool, color_index: int
    ):
        """
        Create the sub-layer-segments of the blanket from a silhouette of
        a sub-layer.
        """  # noqa: DOC201
        if inboard:
            n_seg_per_sector = self.params.n_bb_inboard.value
            name = self.IBS
        else:
            n_seg_per_sector = self.params.n_bb_outboard.value
            name = self.OBS

        shapes = pattern_revolved_silhouette(
            silhouette,
            n_seg_per_sector,
            self.params.n_TF.value,
            self.params.c_rm.value,
        )

        segments = []
        for no, shape in enumerate(shapes):
            segment = PhysicalComponent(
                f"{name}_{sub_name}_{no}",
                shape,
                material=self.get_material(sub_name),
            )
            apply_component_display_options(
                segment, color=BLUE_PALETTE[self.BB][color_index]
            )
            segments.append(segment)

        return segments
