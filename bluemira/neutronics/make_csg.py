# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Create csg geometry from bluemira wires."""

from dataclasses import dataclass

import numpy as np

from bluemira.base.constants import EPS
from bluemira.geometry.coordinates import get_normal_vector
from bluemira.geometry.error import GeometryError
from bluemira.neutronics.params import TokamakGeometry


@dataclass
class WallThicknessFraction:
    """List of thickness of various sections of the blanket as fractions"""

    first_wall: float  # [m]
    breeder_zone: float  # [m]
    manifold: float  # [m]
    vacuum_vessel: float  # [m]

    def __post_init__(self):
        """Check fractions are between 0 and 1 and sums to unity."""
        for section in ("first_wall", "manifold", "vacuum_vessel"):
            if getattr(self, section) <= 0:
                raise GeometryError(f"Thickness fraction of {section} must be non-zero")
        if self.breeder_zone < 0:  # can be zero, but not negative.
            raise GeometryError("Thickness fraction of breeder_zone must be nonnegative")
        if not np.isclose(
            sum(self.first_wall, self.manifold, self.breeder_zone, self.vacuum_vessel),
            1.0,
            rtol=0,
            atol=EPS,
        ):
            raise GeometryError(
                "Thickness fractions of all four sections " "must add up to unity!"
            )


class InboardThicknessFraction(WallThicknessFraction):
    """Thickness fraction list of the inboard wall of the blanket"""

    pass


class OutboardThicknessFraction(WallThicknessFraction):
    """Thickness fraction list of the outboard wall of the blanket"""

    pass


@dataclass
class ThicknessFractions:
    """
    A dataclass containing info. on both
    inboard and outboard blanket thicknesses as fractions.
    """

    inboard: InboardThicknessFraction
    outboard: OutboardThicknessFraction

    @classmethod
    def from_TokamakGeometry(cls, tokamak_geometry: TokamakGeometry):
        """
        Create this dataclass by
        translating from our existing tokamak_geometry dataclass.
        """
        inb_sum = (
            sum([
                tokamak_geometry.inb_fw,
                tokamak_geometry.inb_bz,
                tokamak_geometry.inb_mnfld,
                tokamak_geometry.inb_vv,
            ]),
        )
        inb = InboardThicknessFraction(
            tokamak_geometry.inb_fw,
            tokamak_geometry.inb_bz,
            tokamak_geometry.inb_mnfld,
            tokamak_geometry.inb_vv,
        )
        outb_sum = sum([
            tokamak_geometry.outb_fw,
            tokamak_geometry.outb_bz,
            tokamak_geometry.outb_mnfld,
            tokamak_geometry.outb_vv,
        ])
        outb = OutboardThicknessFraction(
            tokamak_geometry.outb_fw,
            tokamak_geometry.outb_bz,
            tokamak_geometry.outb_mnfld,
            tokamak_geometry.outb_vv,
        )
        return cls(inb, outb)


class Cell2D:
    pass


def determine_panel_outward_normal(panel_in, panel_out):
    """Given two wires (representing the inside and outside face of panel),
    Determine the normal vector of the inside face.
    """
    norm_vec = get_normal_vector(...)
    alignment = np.dot(mean(panel_out) - mean(panel_in), norm_vec)
    if np.sign(alignment) == -1:
        return -norm_vec
    if np.sign(alignment) == 1:
        return norm_vec
    raise GeometryError(
        "normal vector is somehow perpendicular to the 'inboard->outboard' vector??"
    )


def get_bisection_vector(vec_1, vec_2):
    return


def pre_cell_division_lines(blanket_wire, start_division_line, end_division_line):
    """Cut it down into pre-cells"""
    panel_list = ...
    dividing_panel_normals = []
    for panel, next_panel in zip(panel_list[:-1], panel_list[1:]):
        out_norm_1 = determine_panel_outward_normal(panel)
        out_norm_2 = determine_panel_outward_normal(next_panel)
        dividing_panel_normals.append(get_bisection_vector(out_norm_1, out_norm_2))
    pre_cell_division_lines = [start_division_line]
    for norm_i, norm_vec in enumerate(dividing_panel_normals):
        _draw_line(start=panel.end, direction=norm_vec, end=intersection_point)
    pre_cell_division_lines.append(end_division_line)
    return pre_cell_division_lines


@dataclass
class BlanketPreCells:
    pass


def split_fw():
    """Split the first wall into pre-cells."""
    pass
