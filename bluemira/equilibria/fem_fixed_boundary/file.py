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
from dolfin import BoundaryMesh, Vertex
from scipy.integrate import quad, quadrature

from bluemira.equilibria.fem_fixed_boundary.fem_magnetostatic_2D import (
    FemGradShafranovFixedBoundary,
)
from bluemira.equilibria.fem_fixed_boundary.utilities import find_magnetic_axis
from bluemira.equilibria.file import EQDSKInterface
from bluemira.equilibria.grid import Grid


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


def _get_mesh_boundary(mesh):
    """
    Retrieve the boundary of the mesh, as an ordered set of coordinates.

    Parameters
    ----------
    mesh: dolfin.Mesh
        Mesh for which to retrieve the exterior boundary

    Returns
    -------
    xbdry: np.ndarray
        x coordinates of the boundary
    zbdry: np.ndarray
        z coordinates of the boundary
    """
    boundary = BoundaryMesh(mesh, "exterior")
    edges = boundary.cells()
    check_edge = np.ones(boundary.num_edges())

    index = 0
    temp_edge = edges[index]
    sorted_v = []
    sorted_v.append(temp_edge[0])

    for i in range(len(edges) - 1):
        temp_v = [v for v in temp_edge if v not in sorted_v][0]
        sorted_v.append(temp_v)
        check_edge[index] = 0
        connected = np.where(edges == temp_v)
        index = [e for e in connected[0] if check_edge[e] == 1][0]
        temp_edge = edges[index]

    points_sorted = []
    for v in sorted_v:
        points_sorted.append(Vertex(boundary, v).point().array())
    points_sorted = np.array(points_sorted)
    return points_sorted[:, 0], points_sorted[:, 1]


def save_fixed_boundary_to_file(
    file_path: str,
    file_header_name: str,
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
    xbdry, zbdry = _get_mesh_boundary(gs_solver.mesh)
    xbdry = np.append(xbdry, xbdry[0])
    zbdry = np.append(zbdry, zbdry[0])
    nbdry = len(xbdry)

    x_mag, z_mag = find_magnetic_axis(gs_solver.psi, gs_solver.mesh)
    psi_mag = gs_solver.psi(x_mag, z_mag)

    # Make a minimum grid
    x_min = np.min(xbdry)
    x_max = np.max(xbdry)
    z_min = np.min(zbdry)
    z_max = np.max(zbdry)
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
        name=file_header_name,
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
