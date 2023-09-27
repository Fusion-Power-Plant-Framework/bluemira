# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
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
"""Make the entire tokamak from scratch using user-provided variables.
All units used in this file are either [cm] or dimensionless.
"""
import copy
import itertools
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import openmc
from numpy import pi

import bluemira.neutronics.make_materials as mm
import bluemira.neutronics.result_presentation as present
import bluemira.neutronics.volume_functions as vf
from bluemira.base.constants import raw_uc
from bluemira.base.look_and_feel import bluemira_debug, bluemira_print
from bluemira.geometry.wire import BluemiraWire
from bluemira.neutronics.params import TokamakGeometry


def check_geometry(tokamak_geometry: TokamakGeometry) -> None:
    """Some basic geometry checks"""
    if tokamak_geometry.cgs.elong < 1.0:  # noqa: PLR2004
        raise ValueError("Elongation must be at least 1.0")

    inboard_build = (
        tokamak_geometry.cgs.minor_r
        + tokamak_geometry.cgs.inb_fw_thick
        + tokamak_geometry.cgs.inb_bz_thick
        + tokamak_geometry.cgs.inb_mnfld_thick
        + tokamak_geometry.cgs.inb_vv_thick
        + tokamak_geometry.cgs.tf_thick
        + tokamak_geometry.cgs.inb_gap
    )

    if inboard_build > tokamak_geometry.cgs.major_r:
        raise ValueError(
            "The inboard build does not fit within the major radius. Increase the major radius."
        )


def normalize_vec(bis_x, bis_y):
    """Normalises a vector"""
    length = (bis_x**2 + bis_y**2) ** 0.5

    return bis_x / length, bis_y / length


def get_cone_eqn_from_two_points(p1, p2):
    """Gets the equation of the OpenMC cone surface from two points
    Assumes x0 = 0 and y0 = 0
    """
    if p2[2] > p1[2]:
        r1 = p1[0]
        z1 = p1[2]
        r2 = p2[0]
        z2 = p2[2]
    else:
        r1 = p2[0]
        z1 = p2[2]
        r2 = p1[0]
        z2 = p1[2]

    a = r1**2 - r2**2
    b = 2 * (z1 * r2**2 - z2 * r1**2)
    c = r1**2 * z2**2 - r2**2 * z1**2

    z0 = (-b + (b**2 - 4 * a * c) ** 0.5) / (2 * a)
    r02 = r1**2 / (z1 - z0) ** 2

    return z0, r02


def make_offset_poly(old_x, old_y, offset, outer_ccw):
    """Makes a larger polygon with the same angle from a specified offset
    Not mathematically robust for all polygons
    """
    num_points = len(old_x)

    new_x = []
    new_y = []

    for curr in range(num_points):
        prev = (curr + num_points - 1) % num_points
        nxt = (curr + 1) % num_points

        vnn_x, vnn_y = normalize_vec(old_x[nxt] - old_x[curr], old_y[nxt] - old_y[curr])
        nnn_x = vnn_y
        nnn_y = -vnn_x

        vpn_x, vpn_y = normalize_vec(
            old_x[curr] - old_x[prev], old_y[curr] - old_y[prev]
        )
        npn_x = vpn_y * outer_ccw
        npn_y = -vpn_x * outer_ccw

        bisn_x, bisn_y = normalize_vec(
            (nnn_x + npn_x) * outer_ccw, (nnn_y + npn_y) * outer_ccw
        )
        bislen = offset / np.sqrt((1 + nnn_x * npn_x + nnn_y * npn_y) / 2)

        new_x.append(old_x[curr] + bislen * bisn_x)
        new_y.append(old_y[curr] + bislen * bisn_y)

    return new_x, new_y


def offset_points(points, offset_cm):
    """Calls make_offset_poly with points in the format it expects
    to get the points of an offset polygon
    """
    old_rs = []
    old_zs = []

    for point in points:
        old_rs.append(point[0])
        old_zs.append(point[2])

    new_rs, new_zs = make_offset_poly(old_rs, old_zs, offset_cm, 1)

    return np.array([(new_rs[i], 0.0, new_zs[i]) for i in range(len(points))])


def shift_points(points, shift_cm):
    """Moves all radii of points outwards by shift_cm"""
    points[:, 0] += shift_cm

    return points


def elongate(points, adjust_elong):
    """Adjusts the elongation of the points"""
    points[:, 2] *= adjust_elong

    return points


def stretch_r(points, tokamak_geometry: TokamakGeometry, stretch_r_val) -> np.ndarray:
    """Moves the points in the r dimension away from the major radius by extra_r_cm

    Parameters
    ----------
    points: np.array of 2D or 3D points
    stretch_r_val: in cm
    """
    points[:, 0] = (
        points[:, 0] - tokamak_geometry.cgs.major_r
    ) * stretch_r_val + tokamak_geometry.cgs.major_r

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


def _create_z_cone(point1, point2):
    return openmc.ZCone(0.0, 0.0, *get_cone_eqn_from_two_points(point1, point2))


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


def create_layer(
    short_name: str,
    short_name_behind: str,
    layer_points: np.ndarray,
    num_points: int,
    layer_name: str,
    material_lib: mm.MaterialsLibrary,
    surfaces: Surfaces,
    inboard: bool = True,
):
    """Creates a layer of cells for scoring"""
    prefix = "inb" if inboard else "outb"
    cells = []

    # Generating bottom plane, runs bottom to top
    surf = Surface(planes=[openmc.ZPlane(z0=layer_points[-1 if inboard else 0][2])])
    surface = getattr(surfaces, prefix)
    b_surf = getattr(surface, short_name_behind)
    setattr(surface, short_name, surf)
    sign = -1 if inboard else 1
    mid_section = 3

    for i in range(1 if inboard else 0, num_points):
        # Making surfaces

        surf.cones.append(
            _create_z_cone(layer_points[sign * i], layer_points[sign * (i + 1)])
        )
        surf.planes.append(openmc.ZPlane(z0=layer_points[sign * (i + 1)][2]))

        cell = openmc.Cell(
            name=f"{layer_name} {i!s}",
            fill=getattr(material_lib, f"{prefix}_{short_name}_mat"),
            region=-surf.cones[-1] if inboard else +surf.cones[-1],
        )

        # Different top surface for top region
        # if top segment else Recently appended top surface
        cell.region &= (
            (-surfaces.meeting.cone if inboard else +surfaces.meeting.cone)
            if i == num_points - 1
            else -surf.planes[-1]
        )

        # Different bottom surface for bottom region
        if inboard:
            if i == 1:  # if bottom segment
                cell.region &= +surfaces.div.chop
            elif i == 2:  # noqa: PLR2004
                cell.region &= +surf.planes[-2] & +surfaces.div.chop
            else:
                cell.region &= +surf.planes[-2]
        else:
            cell.region &= +surfaces.div.outer_cone if i == 0 else +surf.planes[-2]

        # Adding outside breeder zone surfaces
        if inboard:
            if i == mid_section:  # Middle section
                cell.region &= (
                    +b_surf.cones[i - 2] & +b_surf.cones[i - 1] & +b_surf.cones[i]
                )
            elif i < mid_section:
                cell.region &= +b_surf.cones[i - 1] & +b_surf.cones[i]
            elif i > mid_section:
                cell.region &= +b_surf.cones[i - 2] & +b_surf.cones[i - 1]
        else:
            cell.region &= (
                -b_surf.cones[i] & -b_surf.cones[i + (1 if i < mid_section else -1)]
            )

        # Calculating volume for first wall - not perfect but very close as wall is thin
        if short_name in ("fw", "sf"):
            if inboard:
                inner_cone = b_surf.cones[i - 1]
                outer_cone = surf.cones[-1]
            else:
                inner_cone = surf.cones[-1]
                outer_cone = b_surf.cones[i]
            cell.volume = vf.get_fw_vol(
                outer_cone,
                inner_cone,
                surf.planes[-1],  # top
                surf.planes[-2],  # bottom
            )

        cells.append(cell)

    bluemira_debug(f"Created {layer_name}")
    return cells, surfaces


def create_divertor(
    div_points: np.ndarray,
    outer_points: np.ndarray,
    inner_points: np.ndarray,
    material_lib: mm.MaterialsLibrary,
    surfaces,
    div_clearance=49.0,  # [cm]
    div_fw_thick=2.5,  # [cm]
    div_sf_thick=0.01,  # [cm]
):
    """Creates the divertors cells
    outer_points gives the bottom of the VV
    """
    # Want whichever is lower - offset below the divertor fw or the bottom of the vv
    bot_z = min(
        get_min_max_z_r_of_points(div_points)[0] - div_clearance, outer_points[0][2]
    )
    ds = surfaces.div
    # Want to finish at bottom of outboard vv
    ds.bottom = openmc.ZPlane(z0=bot_z)
    ds.inner_r = openmc.ZCylinder(r=inner_points[-2][0])
    ds.r_chop_in = openmc.ZCylinder(r=div_points[1][0])
    ds.r_chop_out = openmc.ZCylinder(r=div_points[3][0])
    ds.outer_r = openmc.ZCylinder(r=outer_points[0][0])

    for name, points in (
        ("fw", div_points),
        ("fw_back", offset_points(div_points, div_fw_thick)),
        ("scoring", offset_points(div_points, -div_sf_thick)),
    ):
        setattr(
            ds,
            name,
            [_create_z_cone(points[x], points[x + 1]) for x in range(len(points) - 1)],
        )

    ds.fw_back_mid_z = openmc.ZPlane(z0=ds.fw_back[1].z0, boundary_type="vacuum")

    # Creating divertor regions
    inner_region = -ds.top & -ds.chop & -ds.r_chop_in
    mid_region = +ds.r_chop_in & -ds.r_chop_out
    outer_region = -ds.top & +ds.r_chop_out & -ds.outer_cone

    divertor_region_inner = (
        inner_region & (-ds.fw_back[0] | +ds.fw_back[1]) & +ds.inner_r & +ds.bottom
    )
    div_fw_region_inner = (
        inner_region & -ds.fw[0] & +ds.fw_back[0] & -ds.fw_back[1] & +surfaces.tf_coil
    )
    div_sf_region_inner = (
        inner_region & -ds.scoring[0] & +ds.fw[0] & -ds.fw[1] & +surfaces.tf_coil
    )

    divertor_region_mid = (mid_region & +ds.fw_back[1] & -ds.fw_back[2] & +ds.bottom) | (
        -ds.fw_back[1] & -ds.fw_back_mid_z & +ds.r_chop_in & -ds.r_chop_out & +ds.bottom
    )

    div_fw_region_mid = (
        mid_region & +ds.fw[1] & -ds.fw[2] & (-ds.fw_back[1] | +ds.fw_back[2])
    )
    div_sf_region_mid = (
        mid_region & +ds.scoring[1] & -ds.scoring[2] & (-ds.fw[1] | +ds.fw[2])
    )

    divertor_region_outer = outer_region & +ds.fw_back[3] & +ds.bottom & -ds.outer_r
    div_fw_region_outer = outer_region & +ds.fw[3] & -ds.fw_back[3] & +ds.fw_back[2]
    div_sf_region_outer = outer_region & +ds.scoring[3] & -ds.fw[3] & +ds.fw[2]

    cells = DivertorCells(
        # Making divertor cells
        [
            openmc.Cell(
                name=f"Divertor {name}",
                fill=material_lib.divertor_mat,
                region=region,
            )
            for name, region in (
                ("Inner", divertor_region_inner),
                ("Mid", divertor_region_mid),
                ("Outer", divertor_region_outer),
            )
        ],
        # Making divertor first wall cells
        openmc.Cell(
            name="Divertor PFC",
            fill=material_lib.div_fw_mat,
            region=div_fw_region_inner | div_fw_region_mid | div_fw_region_outer,
        ),
        # Making divertor first wall surface cells
        openmc.Cell(
            name="Divertor PFC Surface",
            fill=material_lib.div_sf_mat,
            region=div_sf_region_inner | div_sf_region_mid | div_sf_region_outer,
        ),
        # Region inside the divertor first wall, i.e. part of the plasma chamber
        openmc.Cell(
            name="Divertor Inner 1",
            region=(
                -ds.top
                & +ds.scoring[0]
                & -ds.scoring[1]
                & -ds.scoring[3]
                & -ds.chop
                & +ds.bottom
            ),
        ),
        openmc.Cell(
            name="Divertor Inner 2",
            region=-ds.top & +ds.scoring[1] & +ds.scoring[2] & -ds.scoring[3],
        ),
    )
    cells.fw.volume = vf.get_div_fw_vol(
        [ds.fw[0], ds.fw_back[1], ds.fw[2], ds.fw_back[3]],  # outer_cones
        [ds.fw_back[0], ds.fw[1], ds.fw_back[2], ds.fw[3]],  # inner_cones
        list(div_points[:, 0]),  # radii
    )

    cells.fw_sf.volume = vf.get_div_fw_vol(
        [ds.scoring[0], ds.fw[1], ds.scoring[2], ds.fw[3]],  # outer_cones
        [ds.fw[0], ds.scoring[1], ds.fw[2], ds.scoring[3]],  # inner_cones
        list(div_points[:, 0]),  # radii
    )

    return cells, surfaces


def create_plasma_chamber(surfaces: Surfaces):
    """Creating the cells that live inside the first wall"""
    cells = PlasmaCells(
        openmc.Cell(
            name="Plasma inner 1",
            region=-surfaces.meeting.r_cyl & +surfaces.div.top,
        ),
        openmc.Cell(
            name="Plasma outer 1",
            region=+surfaces.meeting.r_cyl & +surfaces.div.top,
        ),
        openmc.Cell(
            name="Plasma inner 2",
            region=-surfaces.meeting.r_cyl & -surfaces.div.top & +surfaces.div.chop,
        ),
        openmc.Cell(
            name="Plasma outer 2",
            region=+surfaces.meeting.r_cyl & -surfaces.div.top & +surfaces.div.chop,
        ),
    )

    for inb_sf_surf in surfaces.inb.sf.cones:
        cells.inner1.region &= +inb_sf_surf
        cells.inner2.region &= +inb_sf_surf

    for outb_sf_surf in surfaces.outb.sf.cones:
        cells.outer1.region &= -outb_sf_surf
        cells.outer2.region &= -outb_sf_surf
    return cells


def make_geometry(
    tokamak_geometry: TokamakGeometry,
    fw_points: np.ndarray,
    div_points: np.ndarray,
    num_inboard_points: int,
    material_lib: mm.MaterialsLibrary,
    *,
    fw_surf_score_depth: float = 0.0001,
    div_clearance: float = 49.0,  # [cm]
    clearance_r: float = 50.0,  # [cm]
    container_steel_thick: float = 200.0,  # [cm]
    clear_div_to_shell=5.0,  # [cm]
    div_fw_thick=2.5,  # [cm] # Divertor first wall thickness
    div_sf_thick=0.01,  # [cm]
) -> Tuple[Cells, openmc.Universe]:
    """
    Creates an OpenMC CSG geometry for a reactor

    Parameters
    ----------
    tokamak_geometry:
        data all either floats in cm, or dimensionless.
    fw_points:
        coordinates of sample points representing the blanket, where
        blanket = first wall MINUS divertor
        (Hence I think this variable is poorly named)
    div_points:
        coordinates of sample points representing the divertor
    num_inboard_points:
        number of points in fw points that represents the number of inboard points.
    material_lib:
        dictionary of materials {name:openmc.Material} used to create cells.
    fw_surf_score_depth:
        This is a thin geometry layer to score peak surface values
    """
    surfaces = Surfaces()

    inb_fw_thick = tokamak_geometry.cgs.inb_fw_thick
    inb_bz_thick = tokamak_geometry.cgs.inb_bz_thick
    inb_mnfld_thick = tokamak_geometry.cgs.inb_mnfld_thick
    inb_vv_thick = tokamak_geometry.cgs.inb_vv_thick
    tf_thick = tokamak_geometry.cgs.tf_thick

    outb_fw_thick = tokamak_geometry.cgs.outb_fw_thick
    outb_bz_thick = tokamak_geometry.cgs.outb_bz_thick
    outb_mnfld_thick = tokamak_geometry.cgs.outb_mnfld_thick
    outb_vv_thick = tokamak_geometry.cgs.outb_vv_thick

    # Of the points in fw_points, this specifies the number that define the outboard
    num_outboard_points = len(fw_points) - num_inboard_points

    bluemira_print(
        f"OpenMC geometry\nNumber of inboard points {num_inboard_points}\n"
        f"Number of outboard points {num_outboard_points}"
    )

    # Inboard surfaces behind breeder zone
    # Getting layer points
    outb_points = LayerPoints(
        bz=offset_points(fw_points, outb_fw_thick),
        mani=offset_points(fw_points, outb_fw_thick + outb_bz_thick),
        vv=offset_points(fw_points, outb_fw_thick + outb_bz_thick + outb_mnfld_thick),
        end=offset_points(
            fw_points, outb_fw_thick + outb_bz_thick + outb_mnfld_thick + outb_vv_thick
        ),
    )
    inb_points = LayerPoints(
        bz=offset_points(fw_points, inb_fw_thick),
        mani=offset_points(fw_points, inb_fw_thick + inb_bz_thick),
        vv=offset_points(fw_points, inb_fw_thick + inb_bz_thick + inb_mnfld_thick),
        end=offset_points(
            fw_points, inb_fw_thick + inb_bz_thick + inb_mnfld_thick + inb_vv_thick
        ),
    )

    # Getting surface scoring points
    sf_points = offset_points(fw_points, -raw_uc(fw_surf_score_depth, "m", "cm"))

    # Getting tf coil r surfaces
    back_of_inb_vv_r = get_min_r_of_points(inb_points.end)
    gap_vv_tf = tokamak_geometry.cgs.inb_gap

    surfaces.bore = openmc.ZCylinder(r=back_of_inb_vv_r - gap_vv_tf - tf_thick)
    surfaces.tf_coil = openmc.ZCylinder(r=back_of_inb_vv_r - gap_vv_tf)

    # Getting tf coil top and bottom surfaces
    div_points_w_clearance = copy.deepcopy(div_points)
    div_points_w_clearance[:, 2] -= div_clearance

    _, max_z, max_r = get_min_max_z_r_of_points(
        np.concatenate((outb_points.end, inb_points.end[-num_inboard_points:]), axis=0)
    )
    min_z, _, _ = get_min_max_z_r_of_points(
        np.concatenate((outb_points.end, div_points_w_clearance), axis=0)
    )

    # Setting clearance between the top of the divertor and the container shell
    surfaces.inb_top = openmc.ZPlane(
        z0=max_z + clear_div_to_shell, boundary_type="vacuum"
    )
    surfaces.inb_bot = openmc.ZPlane(
        z0=min_z - clear_div_to_shell, boundary_type="vacuum"
    )

    # Making rough divertor surfaces
    surfaces.div = DivertorSurface(
        top=_create_z_cone(fw_points[0], fw_points[-1]),
        chop=openmc.ZPlane(z0=max(fw_points[0][2], fw_points[-1][2])),
        outer_cone=_create_z_cone(outb_points.end[0], fw_points[0]),
    )

    # Making vv, manifold, breeder zone, and first wall
    surfaces.meeting = MeetingPoint(
        openmc.ZCylinder(r=fw_points[-num_inboard_points][0]),
        _create_z_cone(
            fw_points[-num_inboard_points], inb_points.vv[-num_inboard_points]
        ),
    )

    # Generating inboard cone surfaces (back of vv), runs bottom to top
    cell_layers = {}
    for (sign, start), num_points, (i_o, i_o_short, flag), points in zip(
        ((-1, 1), (1, 0)),
        (num_inboard_points, num_outboard_points),
        (("Inboard", "inb", True), ("Outboard", "outb", False)),
        (inb_points, outb_points),
    ):
        setattr(
            surfaces,
            i_o_short,
            SurfaceZones(
                Surface(
                    cones=[
                        _create_z_cone(points.end[sign * i], points.end[sign * (i + 1)])
                        for i in range(start, num_points)
                    ]
                )
            ),
        )
        cell_layers[i_o.lower()] = LayerCells()
        for name, lyr, lyr_b, pnts in (
            ("VV", "vv", "end", points.vv),
            ("Manifold", "mani", "vv", points.mani),
            ("BZ", "bz", "mani", points.bz),
            ("FW", "fw", "bz", fw_points),
            ("FW Surface", "sf", "fw", sf_points),
        ):
            layer_cell, surfaces = create_layer(
                lyr,
                lyr_b,
                pnts,
                num_points,
                f"{i_o} {name}",
                material_lib,
                surfaces,
                inboard=flag,
            )
            setattr(cell_layers[i_o.lower()], lyr, layer_cell)

    # Outboard surfaces
    # Currently it is not possible to tally on boundary_type='vacuum' surfaces
    surfaces.outer_surface_cyl = openmc.ZCylinder(r=max_r + clearance_r)
    surfaces.graveyard = GraveyardSurface(
        openmc.ZPlane(z0=max_z + container_steel_thick),
        openmc.ZPlane(z0=min_z - container_steel_thick),
        openmc.ZCylinder(
            r=max_r + clearance_r + container_steel_thick, boundary_type="vacuum"
        ),
    )

    # Inboard cells
    bore = openmc.Cell(
        name="Inner bore",
        region=-surfaces.bore & -surfaces.inb_top & +surfaces.inb_bot,
    )
    tf_coil = openmc.Cell(
        name="TF Coils",
        fill=material_lib.tf_coil_mat,
        region=-surfaces.tf_coil
        & +surfaces.bore
        & -surfaces.inb_top
        & +surfaces.inb_bot,
    )

    # Divertor cells
    divertor, surfaces = create_divertor(
        div_points,
        outb_points.end,
        inb_points.end,
        material_lib,
        surfaces,
        div_clearance,
        div_fw_thick,
        div_sf_thick,
    )

    # Plasma chamber
    plasma = create_plasma_chamber(surfaces)

    # Container cells
    outer_vessel = openmc.Cell(
        name="Outer VV Container",
        region=-surfaces.outer_surface_cyl
        & -surfaces.inb_top
        & +surfaces.inb_bot
        & +surfaces.tf_coil,
    )

    for cell in (
        *plasma.get_cells(),
        *divertor.get_cells()[1:],
        *cell_layers["inboard"].get_cells(),
        *cell_layers["outboard"].get_cells(),
        *divertor.regions,
    ):
        outer_vessel.region &= ~cell.region

    outer_container = openmc.Cell(
        name="Container steel",
        fill=material_lib.container_mat,
        region=~(-surfaces.outer_surface_cyl & -surfaces.inb_top & +surfaces.inb_bot)
        & -surfaces.graveyard.cylinder
        & -surfaces.graveyard.top
        & +surfaces.graveyard.bottom,
    )
    cells = Cells(
        bore=bore,
        tf_coil=tf_coil,
        outer_vessel=outer_vessel,
        outer_container=outer_container,
        divertor=divertor,
        plasma=plasma,
        inboard=cell_layers["inboard"],
        outboard=cell_layers["outboard"],
    )

    # Create universe
    universe = openmc.Universe(cells=cells.get_cells())
    geometry = openmc.Geometry(universe)
    geometry.export_to_xml()

    return cells, universe


def load_fw_points(
    tokamak_geometry: TokamakGeometry,
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
    tokamak_geometry:
        tokamak geometry parameters
    blanket_wire:
        The blanket wire
    divertor_wire:
        The divertor wire
    major_radius:
        major radius of the actual device.
        The geometry variables specified by
        tokamak_geometry will then be modified by major_radius
    aspect_ratio:
        aspect ratio of the reactor

    Returns
    -------
    new_downsampled_fw:
        points belonging to the first wall
    new_downsampled_div:
        points belonging to the divertor
    num_inboard_points:
        Number of points in new_downsampled_fw
        that belongs to the inboard section.

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
    shift_cm = tokamak_geometry.cgs.major_r - ex_pts_maj_r
    new_points = shift_points(old_points, shift_cm)

    # Adjusting points for elongation and minor radius
    # This elongation also include an allowance for the minor radius
    elong_w_minor_r = (
        tokamak_geometry.cgs.minor_r / ex_pts_min_r * tokamak_geometry.cgs.elong
    )
    stretch_r_val = tokamak_geometry.cgs.minor_r / ex_pts_min_r
    new_points = elongate(new_points, elong_w_minor_r / ex_pts_elong)
    new_points = stretch_r(new_points, tokamak_geometry, stretch_r_val)

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
        u = tokamak_geometry.cgs.major_r  # x-position of the center
        v = 0.0  # y-position of the center
        a = tokamak_geometry.cgs.minor_r  # radius on the x-axis
        b = (
            tokamak_geometry.cgs.elong * tokamak_geometry.cgs.minor_r
        )  # radius on the y-axis
        tri = tokamak_geometry.cgs.triang  # triangularity
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
