# %%
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

"""
File saving for fixed boundary equilibrium
"""
from typing import Dict, Optional

import numpy as np
from dolfin import BoundaryMesh
from scipy.integrate import quad, quadrature
from scipy.spatial import ConvexHull

from bluemira.equilibria.fem_fixed_boundary.fem_magnetostatic_2D import (
    FemGradShafranovFixedBoundary,
)
from bluemira.equilibria.fem_fixed_boundary.utilities import find_magnetic_axis
from bluemira.equilibria.file import EQDSKInterface
from bluemira.equilibria.grid import Grid
from bluemira.geometry.coordinates import Coordinates


def _pressure_profile(pprime, psi_norm, psi_mag):
    pressure = np.zeros(len(psi_norm))
    for i in range(len(psi_norm)):
        pressure[i] = quad(pprime, psi_norm[i], 1.0, limit=500)[0] * psi_mag
    return pressure


def _fpol_profile(ffprime, psi_norm, psi_mag, fvac):
    fpol = np.zeros(len(psi_norm))
    for i in range(len(psi_norm)):
        fpol[i] = np.sqrt(
            2
            * quadrature(ffprime, psi_norm[i], 1.0, maxiter=500, rtol=1e-6, tol=1e-6)[0]
            * psi_mag
            + fvac**2
        )
    return fpol


def save_fixed_boundary_to_file(
    file_path: str,
    file_name: str,
    gs_solver: FemGradShafranovFixedBoundary,
    nx: int,
    nz: int,
    formatt: str = "json",
    json_kwargs: Optional[Dict] = None,
):
    """
    Save a fixed boundary equilibrium to a file.

    Parameters
    ----------
    gs_solver: FemGradShafranovFixedBoundary
    nx: int
        Number of radial points to use in the psi map
    nz: int
        Number of vertical points to use in the psi map
    """
    # Recover the FE boundary from the mesh
    mesh = gs_solver.mesh
    boundary = BoundaryMesh(mesh, "exterior", False).coordinates()
    # It's not ordered by connectivity, so we take the convex hull (same
    # number of points, because the shape should be convex)
    hull = ConvexHull(boundary)
    xbdry, zbdry = hull.points[hull.vertices].T
    xbdry = np.append(xbdry, xbdry[0])
    zbdry = np.append(zbdry, zbdry[0])
    nbdry = len(xbdry)

    x_mag, z_mag = find_magnetic_axis(gs_solver.psi, gs_solver.mesh)
    psi_mag = gs_solver.psi(x_mag, z_mag)

    # Make a minimum grid
    x_coords, z_coords = gs_solver.mesh.coordinates().T
    x_min = np.min(x_coords)
    x_max = np.max(x_coords)
    z_min = np.min(z_coords)
    z_max = np.max(z_coords)
    grid = Grid(x_min, x_max, z_min, z_max, nx=nx, nz=nz)

    psi = np.zeros((nx, nz))
    for i, xi in enumerate(grid.x_1d):
        for j, zj in enumerate(grid.z_1d):
            psi[i, j] = gs_solver.psi([xi, zj])

    psi_norm = np.linspace(0, 1, 50)
    pprime = gs_solver._pprime
    ffprime = gs_solver._ffprime
    if callable(pprime):
        pprime_values = pprime(psi_norm)
    if callable(ffprime):
        ffprime_values = ffprime(psi_norm)
    else:
        psi_norm = np.linspace(0, 1, len(ffprime))

    fvac = grid.x_mid * gs_solver._B_0
    psi_vector = psi_norm * psi_mag
    pressure = _pressure_profile(pprime, psi_vector, psi_mag)
    fpol = _fpol_profile(ffprime, psi_norm, psi_mag, fvac)

    data = EQDSKInterface(
        bcentre=gs_solver._B_0,
        cplasma=gs_solver._curr_target,
        dxc=np.array([]),
        dzc=np.array([]),
        ffprime=ffprime_values,
        fpol=fpol,
        Ic=np.array([]),
        name="test",
        nbdry=nbdry,
        ncoil=0,
        nlim=0,
        nx=nx,
        nz=nz,
        pressure=pressure,
        pprime=pprime_values,
        psi=psi,
        psibdry=np.zeros(nbdry),
        psimag=psi_mag,
        xbdry=xbdry,
        xc=np.array([]),
        xcentre=grid.x_mid,
        xdim=grid.x_size,
        xgrid1=grid.x_min,
        xlim=np.array([]),
        xmag=x_mag,
        zbdry=zbdry,
        zc=np.array([]),
        zdim=grid.z_size,
        zlim=np.array([]),
        zmag=z_mag,
        zmid=grid.z_mid,
        x=grid.x_1d,
        z=grid.z_1d,
        psinorm=psi_norm,
        qpsi=np.array([]),
    )
    data.write(file_path, format=formatt, json_kwargs=json_kwargs)
    return data
