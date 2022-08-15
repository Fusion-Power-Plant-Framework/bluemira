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
import numpy as np

from bluemira.base.error import BuilderError
from bluemira.equilibria.coils import Coil, CoilSet
from bluemira.equilibria.grid import Grid
from bluemira.geometry.constants import VERY_BIG
from bluemira.geometry.tools import distance_to, make_polygon, offset_wire


def make_solenoid(r_cs, tk_cs, z_min, z_max, g_cs, tk_cs_ins, tk_cs_cas, n_CS):
    """
    Make a set of solenoid coils in an EU-DEMO fashion. If n_CS is odd, the central
    module is twice the size of the others. If n_CS is even, all the modules are the
    same size.

    Parameters
    ----------
    r_cs: float
        Radius of the solenoid
    tk_cs: float
        Half-thickness of the solenoid in the radial direction (including insulation and
        casing)
    z_min: float
        Minimum vertical position of the solenoid
    z_max: float
        Maximum vertical position of the solenoid
    g_cs: float
        Gap between modules
    tk_cs_ins: float
        Insulation thickness around modules
    tk_cs_cas: float
        Casing thickness around modules
    n_CS: int
        Number of modules in the solenoid

    Returns
    -------
    coils: List[Coil]
        List of solenoid coil(s)
    """

    def make_CS_coil(z_coil, dz_coil, i):
        return Coil(
            r_cs,
            z_coil,
            current=0,
            dx=tk_cs - tk_inscas,
            dz=dz_coil,
            control=True,
            ctype="CS",
            name=f"CS_{i+1}",
            flag_sizefix=True,
        )

    if z_max < z_min:
        z_min, z_max = z_max, z_min
    if np.isclose(z_max, z_min):
        raise BuilderError(f"Cannot make a solenoid with z_min==z_max=={z_min}")

    total_height = z_max - z_min
    tk_inscas = tk_cs_ins + tk_cs_cas
    total_gaps = (n_CS - 1) * g_cs + n_CS * 2 * tk_inscas
    if total_gaps >= total_height:
        raise BuilderError(
            "Cannot make a solenoid where the gaps and insulation + casing are larger than the height available."
        )

    coils = []
    if n_CS == 1:
        # Single CS module solenoid (no gaps)
        module_height = total_height - 2 * tk_inscas
        coil = make_CS_coil(0.5 * total_height, 0.5 * module_height, 0)
        coils.append(coil)

    elif n_CS % 2 == 0:
        # Equally-spaced CS modules for even numbers of CS coils
        module_height = (total_height - total_gaps) / n_CS
        dz_coil = 0.5 * module_height
        z_iter = z_max
        for i in range(n_CS):
            z_coil = z_iter - tk_inscas - dz_coil
            coil = make_CS_coil(z_coil, dz_coil, i)
            coils.append(coil)
            z_iter = z_coil - dz_coil - tk_inscas - g_cs

    else:
        # Odd numbers of modules -> Make a central module that is twice the size of the
        # others.
        module_height = (total_height - total_gaps) / (n_CS + 1)
        z_iter = z_max
        for i in range(n_CS):
            if i == n_CS // 2:
                # Central module
                dz_coil = module_height
                z_coil = z_iter - tk_inscas - dz_coil

            else:
                # All other modules
                dz_coil = 0.5 * module_height
                z_coil = z_iter - tk_inscas - dz_coil

            coil = make_CS_coil(z_coil, dz_coil, i)
            coils.append(coil)
            z_iter = z_coil - dz_coil - tk_inscas - g_cs

    return coils


def make_PF_coil_positions(tf_boundary, n_PF, R_0, kappa, delta):
    """
    Make a set of PF coil positions crudely with respect to the intended plasma
    shape.
    """
    # Project plasma centroid through plasma upper and lower extrema
    angle_upper = np.arctan2(kappa, -delta)
    angle_lower = np.arctan2(-kappa, -delta)
    scale = 1.1

    angles = np.linspace(scale * angle_upper, scale * angle_lower, n_PF)

    x_c, z_c = np.zeros(n_PF), np.zeros(n_PF)
    for i, angle in enumerate(angles):
        line = make_polygon(
            [
                [R_0, R_0 + VERY_BIG * np.cos(angle)],
                [0, 0],
                [0, VERY_BIG * np.sin(angle)],
            ]
        )
        _, intersection = distance_to(tf_boundary, line)
        x_c[i], _, z_c[i] = intersection[0][0]
    return x_c, z_c


def make_coilset(
    tf_boundary,
    R_0,
    kappa,
    delta,
    r_cs,
    tk_cs,
    g_cs,
    tk_cs_ins,
    tk_cs_cas,
    n_CS,
    n_PF,
    CS_jmax,
    CS_bmax,
    PF_jmax,
    PF_bmax,
):
    """
    Make an initial EU-DEMO-like coilset.
    """
    bb = tf_boundary.bounding_box
    z_min = bb.z_min
    z_max = bb.z_max
    solenoid = make_solenoid(r_cs, tk_cs, z_min, z_max, g_cs, tk_cs_ins, tk_cs_cas, n_CS)

    tf_track = offset_wire(tf_boundary, 1, join="arc")
    x_c, z_c = make_PF_coil_positions(
        tf_track,
        n_PF,
        R_0,
        kappa,
        delta,
    )
    pf_coils = []
    for i, (x, z) in enumerate(zip(x_c, z_c)):
        coil = Coil(
            x,
            z,
            current=0,
            ctype="PF",
            control=True,
            name=f"PF_{i+1}",
            flag_sizefix=False,
            j_max=PF_jmax,
            b_max=PF_bmax,
        )
        pf_coils.append(coil)
    coilset = CoilSet(pf_coils + solenoid)
    coilset.assign_coil_materials("PF", j_max=PF_jmax, b_max=PF_bmax)
    coilset.assign_coil_materials("CS", j_max=CS_jmax, b_max=CS_bmax)
    return coilset


def make_grid(R_0, A, kappa, scale_x=1.6, scale_z=1.7, nx=65, nz=65):
    """
    Make a finite difference Grid for an Equilibrium.
    Parameters
    ----------
    R_0: float
        Major radius
    A: float
        Aspect ratio
    kappa: float
        Elongation
    scale_x: float
        Scaling factor to "grow" the grid away from the plasma in the x direction
    scale_z: float
        Scaling factor to "grow" the grid away from the plasma in the z direction
    nx: int
        Grid discretisation in the x direction
    nz: int
        Grid discretisation in the z direction
    Returns
    -------
    grid: Grid
        Finite difference grid for an Equilibrium
    """
    x_min, x_max = R_0 - scale_x * (R_0 / A), R_0 + scale_x * (R_0 / A)
    z_min, z_max = -scale_z * (kappa * R_0 / A), scale_z * (kappa * R_0 / A)
    return Grid(x_min, x_max, z_min, z_max, nx, nz)
