# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Make the entire tokamak from scratch using user-provided variables.
All units used in this file are either [cm] or dimensionless.
"""

import itertools
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import openmc
from numpy import pi

import bluemira.neutronics.result_presentation as present
from bluemira.base.constants import raw_uc
from bluemira.geometry.wire import BluemiraWire
from bluemira.neutronics.params import (
    PlasmaSourceParameters,
    PlasmaSourceParametersPPS,
    TokamakGeometry,
)


def check_geometry(
    source_parameters: PlasmaSourceParameters, tokamak_geometry: TokamakGeometry
) -> None:
    """Some basic geometry checks.

    Parameters
    ----------
    source_parameters:
        dataclass with attribute minor_r, major_r, elong

    tokamak_geometry:
        dataclass with various thickness attributes. (See TokamakGeometry for details.)

    It doesn't matter whether these two parameters are provided in metric or cgs,
    As long as the units match the check can proceed.
    """
    if source_parameters.elongation < 1.0:  # noqa: PLR2004
        raise ValueError("Elongation must be at least 1.0")

    inboard_build = (
        source_parameters.minor_r
        + tokamak_geometry.inb_fw_thick
        + tokamak_geometry.inb_bz_thick
        + tokamak_geometry.inb_mnfld_thick
        + tokamak_geometry.inb_vv_thick
        + tokamak_geometry.tf_thick
        + tokamak_geometry.inb_gap
    )

    if inboard_build > source_parameters.major_r:
        raise ValueError(
            "The inboard build does not fit within the major radius."
            " Increase the major radius."
        )


def normalize_vec(bis_x, bis_y):
    """Normalises a vector"""
    length = (bis_x**2 + bis_y**2) ** 0.5

    return bis_x / length, bis_y / length


def shift_points(points, shift_cm):
    """Moves all radii of points outwards by shift_cm"""
    points[:, 0] += shift_cm

    return points


def elongate(points, adjust_elong):
    """Adjusts the elongation of the points"""
    points[:, 2] *= adjust_elong

    return points


def stretch_r(
    points, source_parameters: PlasmaSourceParametersPPS, stretch_r_val
) -> np.ndarray:
    """Moves the points in the r dimension away from the major radius by extra_r_cm

    Parameters
    ----------
    points: np.array of 2D or 3D points
    source_parameters: PlasmaSourceParametersPPS
        The values that we want is actually in source_parameters.plasma_physics_units.
    stretch_r_val: in cm
    """
    points[:, 0] = (
        points[:, 0] - source_parameters.plasma_physics_units.major_r
    ) * stretch_r_val + source_parameters.plasma_physics_units.major_r

    return points


def get_min_r_of_points(points):
    """Adjusts the elongation of the points"""
    return np.amin(points, axis=0)[0]


def get_min_max_z_r_of_points(points):
    """Adjusts the elongation of the points"""
    min_z = np.amin(points, axis=0)[2]
    max_z = np.amax(points, axis=0)[2]
    max_r = np.amax(points, axis=0)[0]

    return min_z, max_z, max_r


def _join_lists(lists):
    return list(itertools.chain(*[a if isinstance(a, list) else [a] for a in lists]))


@dataclass
class Surface:
    """Suface CSG sections"""

    cones: List = field(default_factory=list)
    planes: List = field(default_factory=list)


@dataclass
class DivertorSurface:
    """Divertor suface CSG model"""

    top: openmc.ZCone
    chop: openmc.ZPlane
    outer_cone: openmc.ZCone
    bottom: Optional[openmc.ZPlane] = None
    inner_r: Optional[openmc.ZCylinder] = None
    r_chop_in: Optional[openmc.ZCylinder] = None
    r_chop_out: Optional[openmc.ZCylinder] = None
    outer_r: Optional[openmc.ZCylinder] = None
    fw: Optional[List[openmc.ZCone]] = None
    # Divertor surface
    fw_back: Optional[List[openmc.ZCone]] = None
    # Div first wall back
    scoring: Optional[List[openmc.ZCone]] = None
    fw_back_mid_z: Optional[openmc.ZPlane] = None


@dataclass
class GraveyardSurface:
    """Graveyard suface CSG sections"""

    top: openmc.ZPlane
    bottom: openmc.ZPlane
    cylinder: openmc.ZCylinder


@dataclass
class SurfaceZones:
    """Section of CSG model"""

    end: Surface
    vv: Optional[Surface] = None
    mani: Optional[Surface] = None
    bz: Optional[Surface] = None
    fw: Optional[Surface] = None
    sf: Optional[Surface] = None


@dataclass
class MeetingPoint:
    """CSG model for meeting point between inboard and outboard"""

    r_cyl: openmc.ZCylinder
    cone: openmc.ZCone


@dataclass
class LayerPoints:
    """Points for each layer of CSG model"""

    bz: np.ndarray
    mani: np.ndarray
    vv: np.ndarray
    end: np.ndarray


@dataclass
class Surfaces:
    """CGS surfaces"""

    bore: Optional[openmc.ZCylinder] = None
    div: Optional[DivertorSurface] = None
    graveyard: Optional[GraveyardSurface] = None
    inb: Optional[SurfaceZones] = None
    inb_bot: Optional[openmc.ZPlane] = None
    inb_top: Optional[openmc.ZPlane] = None
    meeting: Optional[MeetingPoint] = None
    outb: Optional[SurfaceZones] = None
    outer_surface_cyl: Optional[openmc.ZCylinder] = None
    tf_coil: Optional[openmc.ZCylinder] = None


@dataclass
class PlasmaCells:
    """Plasma CSG cells"""

    inner1: openmc.Cell
    outer1: openmc.Cell
    inner2: openmc.Cell
    outer2: openmc.Cell

    def get_cells(self):
        """Get cells from dataclass"""
        return self.inner1, self.outer1, self.inner2, self.outer2


@dataclass
class DivertorCells:
    """Divertor CSG cells"""

    regions: List[openmc.Cell]
    fw: openmc.Cell
    fw_sf: openmc.Cell
    inner1: openmc.Cell
    inner2: openmc.Cell

    def get_cells(self):
        """Get cells from dataclass"""
        return self.regions, self.fw, self.fw_sf, self.inner1, self.inner2


@dataclass
class LayerCells:
    """Reactor layer CSG cells"""

    bz: Optional[List[openmc.Cell]] = None
    fw: Optional[List[openmc.Cell]] = None
    mani: Optional[List[openmc.Cell]] = None
    sf: Optional[List[openmc.Cell]] = None
    vv: Optional[List[openmc.Cell]] = None

    def get_cells(self):
        """Get cells from dataclass"""
        return _join_lists(a for a in (self.bz, self.fw, self.mani, self.sf, self.vv))


@dataclass
class Cells:
    """Universe CSG cells"""

    bore: openmc.Cell
    tf_coil: openmc.Cell
    outer_vessel: openmc.Cell
    outer_container: openmc.Cell
    divertor: DivertorCells
    plasma: PlasmaCells
    inboard: LayerCells
    outboard: LayerCells

    def get_cells(self):
        """Get cells from dataclass"""
        return [
            self.bore,
            self.tf_coil,
            self.outer_vessel,
            self.outer_container,
            *self.divertor.regions,
            *self.divertor.get_cells()[1:],
            *self.plasma.get_cells(),
            *self.inboard.get_cells(),
            *self.outboard.get_cells(),
        ]


def load_fw_points(
    source_parameters: PlasmaSourceParametersPPS,
    blanket_wire: BluemiraWire,
    divertor_wire: BluemiraWire,
    new_major_radius: float,
    new_aspect_ratio: float,
    new_elong: float,
    save_plots: bool = True,
) -> Tuple[npt.NDArray, npt.NDArray, int]:
    """
    Load given first wall points,
    scale them according to the given major and minor radii,
    then downsample them so that a simplified geometry can be made.

    Currently these are full of magic numbers (e.g. manually chosen integer indices)
    because it was tailored for one specific model.

    Parameters
    ----------
    source_parameters: PlasmaSourceParametersPPS
        a dataclass containing plasma geometry parameters:
        major_r, minor_r, elong, triang.

    blanket_wire: BluemiraWire
        The blanket wire

    divertor_wire: BluemiraWire
        The divertor wire

    new_major_radius: [cm]
        major radius of the actual device.

    new_aspect_ratio: [dimensionless]
        aspect ratio of the reactor

    new_elong: [dimensionless]
        elongation parameter of the actual device.

    save_plot: bool
        if True, the following plots would be saved to current directory:
        - blanket_face.svg
        - all_points_before_after.svg
        - selected_pts_inner_blanket_face.svg
        - selected_pts_divertor_face.svg

    Returns
    -------
    new_downsampled_fw:
        points belonging to the first wall

    new_downsampled_div:
        points belonging to the divertor

    num_inboard_points:
        Number of points in new_downsampled_fw that belongs to the inboard section.

    Notes
    -----
    All units are in cgs
    """
    full_blanket_2d_outline = raw_uc(blanket_wire.discretize(100).T, "m", "cm")
    divertor_2d_outline = raw_uc(divertor_wire.discretize(100).T, "m", "cm")
    ex_pts_maj_r = new_major_radius
    ex_pts_min_r = new_major_radius / new_aspect_ratio
    ex_pts_elong = new_elong
    # Specifying the number of the selected points that define the inboard.
    num_inboard_points = 6
    # sample points indices
    selected_fw_samples = [0, 4, 8, 11, 14, 17, 21, 25, 28, 33, 39, 42, -1]
    # also going to use first and last points from first wall
    selected_div_samples = [72, 77, 86]
    num_points_belongong_to_divertor = len(selected_div_samples)

    def _fix_downsampled_ibf(ds_ibf):
        """Move the point that is too close to plasma
        by moving it closer to the central column instead.
        """
        ds_ibf[-5][0] = ds_ibf[-5][0] - 25.0  # [cm]
        return ds_ibf

    # select the part of the outline facing the plasma
    inner_blanket_face = full_blanket_2d_outline[52:-2]

    # (down)sample existing data
    # blanket
    downsampled_ibf = inner_blanket_face[selected_fw_samples]
    downsampled_ibf = _fix_downsampled_ibf(downsampled_ibf)
    # divertor
    downsampled_divf = divertor_2d_outline[selected_div_samples]

    # Create the full plasma-facing outline by concatenating existing var
    old_points = np.concatenate((downsampled_ibf, downsampled_divf), axis=0)

    # rescale data to fit new geometry.
    # Expand point outwards according to new major radius
    shift_cm = source_parameters.plasma_physics_units.major_r - ex_pts_maj_r
    new_points = shift_points(old_points, shift_cm)

    # Adjusting points for elongation and minor radius
    # This elongation also include an allowance for the minor radius
    elong_w_minor_r = (
        source_parameters.plasma_physics_units.minor_r
        / ex_pts_min_r
        * source_parameters.plasma_physics_units.elongation
    )
    stretch_r_val = source_parameters.plasma_physics_units.minor_r / ex_pts_min_r
    new_points = elongate(new_points, elong_w_minor_r / ex_pts_elong)
    new_points = stretch_r(new_points, source_parameters, stretch_r_val)

    # split 'new_points' into new_downsampled_* variables
    new_downsampled_fw = new_points[:-num_points_belongong_to_divertor]
    new_downsampled_div = np.concatenate(
        (new_points[-(num_points_belongong_to_divertor + 1) :], new_points[:1]), axis=0
    )

    # It seems wrong to have the plotting happening within the make_geometry stage?
    # Just something to consider when refactoring.
    # plotting.
    if save_plots:
        # create parametric variables for plotting smoother lines
        u = source_parameters.plasma_physics_units.major_r  # x-position of the center
        v = 0.0  # y-position of the center
        a = source_parameters.plasma_physics_units.minor_r  # radius on the x-axis
        b = (
            source_parameters.plasma_physics_units.elongation
            * source_parameters.plasma_physics_units.minor_r
        )  # radius on the y-axis
        tri = source_parameters.plasma_physics_units.triang  # triangularity
        t = np.linspace(0, 2 * pi, 100)

        with present.PoloidalXSPlot("blanket_face.svg", "Blanket Face") as ax:
            ax.scatter_cm(full_blanket_2d_outline[:, 0], full_blanket_2d_outline[:, 2])

        with present.PoloidalXSPlot(
            "all_points_before_after.svg", "Points sampled for making the OpenMC model"
        ) as ax:
            ax.plot_cm(old_points[:, 0], old_points[:, 2], label="Initial fw points")
            ax.plot_cm(new_points[:, 0], new_points[:, 2], label="Adjusted fw points")
            ax.plot_cm(
                u + a * np.cos(t + tri * np.sin(t)),
                v + b * np.sin(t),
                label="Plasma envelope",
            )  # source envelope
            ax.legend(loc="upper right")

        with present.PoloidalXSPlot(
            "selected_pts_inner_blanket_face.svg", "Selected points on the inner blanket"
        ) as ax:
            ax.scatter_cm(new_downsampled_fw[:, 0], new_downsampled_fw[:, 2])

        with present.PoloidalXSPlot(
            "selected_pts_divertor_face.svg", "Selected points on the divertor face"
        ) as ax:
            ax.scatter_cm(new_downsampled_div[:, 0], new_downsampled_div[:, 2])

    return new_downsampled_fw, new_downsampled_div, num_inboard_points
