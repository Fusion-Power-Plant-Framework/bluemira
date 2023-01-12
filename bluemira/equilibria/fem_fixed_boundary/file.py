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

from bluemira.equilibria.fem_fixed_boundary.fem_magnetostatic_2D import (
    FemGradShafranovFixedBoundary,
)
from bluemira.equilibria.fem_fixed_boundary.utilities import find_magnetic_axis
from bluemira.equilibria.file import EQDSKInterface
from bluemira.equilibria.grid import Grid
from bluemira.geometry.coordinates import Coordinates


def save_fixed_boundary_to_file(
    file_path,
    file_name,
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
    mesh = gs_solver.mesh
    xb, zb = BoundaryMesh(mesh, "exterior").coordinates().T

    boundary = Coordinates({"x": xb, "y": 0, "z": zb})
    boundary.set_ccw([0, 1, 0])
    xbdry, zbdry = boundary.x, boundary.z
    nbdry = len(xbdry)

    x_coords, z_coords = gs_solver.mesh.coordinates().T
    x_min = np.min(x_coords)
    x_max = np.max(x_coords)
    z_min = np.min(z_coords)
    z_max = np.max(z_coords)
    grid = Grid(x_min, x_max, z_min, z_max, nx=nx, nz=nz)
    psi = gs_solver.psi(grid.x, grid.z)

    psi_norm = np.linspace(0, 1, 50)
    pprime = gs_solver._pprime
    ffprime = gs_solver._ffprime
    if callable(pprime):
        pprime = pprime(psi_norm)
    if callable(ffprime):
        ffprime = ffprime(psi_norm)
    else:
        psi_norm = np.linspace(0, 1, len(ffprime))

    x_mag, z_mag = find_magnetic_axis(gs_solver.psi, gs_solver.mesh)
    psi_mag = gs_solver.psi(x_mag, z_mag)
    data = EQDSKInterface(
        bcentre=99.9,  # TODO
        cplasma=gs_solver._curr_target,
        dxc=np.array([]),
        dzc=np.array([]),
        ffprime=ffprime,
        fpol=ffprime,  # TODO
        Ic=np.array([]),
        name="test",
        nbdry=nbdry,
        ncoil=0,
        nlim=0,
        nx=nx,
        nz=nz,
        pressure=np.array([]),
        pprime=pprime,
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
    data.write(file_path, format=format, json_kwargs=json_kwargs)
