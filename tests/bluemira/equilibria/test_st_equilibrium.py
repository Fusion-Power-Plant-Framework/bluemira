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
BLUEPRINT -> bluemira ST equilibrium recursion test
"""

import pytest
import os
from bluemira.base.file import get_bluemira_root
from bluemira.equilibria import (
    Equilibrium,
    CustomProfile,
    Grid,
    CoilSet,
    MagneticConstraintSet,
    IsofluxConstraint,
    Norm2Tikhonov,
    Coil,
    SymmetricCircuit,
    PicardDeltaIterator,
)


class TestSTEquilibrium:
    @classmethod
    def setup_class(cls):
        root = get_bluemira_root()

    def test_equilibrium(self):
        build_tweaks = {
            "plot_fbe_evol": True,
            "plot_fbe": True,
            "sol_isoflux": True,
            "process_midplane_iso": True,
            "tikhonov_gamma": 1e-8,
            "fbe_convergence": "Dudson",
            "fbe_convergence_crit": 1.0e-6,
            "nx_number_x": 7,
            "nz_number_z": 8,
        }

        R_0 = 3.639
        A = 1.667
        I_p = 20975205.2  # (EQDSK)

        xc = [1.5, 1.5, 8.259059936102478, 8.259059936102478, 10.635505223274231]
        zc = [8.78, 11.3, 11.8, 6.8, 1.7]
        dxc = [0.175, 0.25, 0.25, 0.25, 0.35]
        dzc = [0.5, 0.4, 0.4, 0.4, 0.5]

        inboard_iso = [R_0 * (1.0 - 1 / A), 0.0]
        outboard_iso = [R_0 * (1.0 + 1 / A), 0.0]

        upper_iso = [x[np.argmax(z)], np.max(z)]
        lower_iso = [x[np.argmin(z)], np.min(z)]

        x_core = np.array([inboard_iso[0], upper_iso[0], outboard_iso[0], lower_iso[0]])
        z_core = np.array([inboard_iso[1], upper_iso[1], outboard_iso[1], lower_iso[1]])

        # Points chosen to replicate divertor legs in AH's FIESTA demo
        x_hfs = np.array(
            [
                1.42031,
                1.057303,
                0.814844,
                0.669531,
                0.621094,
                0.621094,
                0.645312,
                0.596875,
            ]
        )
        z_hfs = np.array(
            [4.79844, 5.0875, 5.37656, 5.72344, 6.0125, 6.6484, 6.82188, 7.34219]
        )
        x_lfs = np.array(
            [1.85625, 2.24375, 2.53438, 2.89766, 3.43047, 4.27813, 5.80391, 6.7]
        )
        z_lfs = np.array(
            [4.79844, 5.37656, 5.83906, 6.24375, 6.59063, 6.76406, 6.70625, 6.70625]
        )

        x_div = np.concatenate([x_lfs, x_lfs, x_hfs, x_hfs])
        z_div = np.concatenate([z_lfs, -z_lfs, z_hfs, -z_hfs])

        # Scale up Agnieszka isoflux constraints
        size_scaling = R_0 / 2.5
        x_div = size_scaling * x_div
        z_div = size_scaling * z_div

        xx = np.concatenate([x_core, x_div])
        zz = np.concatenate([z_core, z_div])

        constraint_set = MagneticConstraintSet(
            [IsofluxConstraint(xx, zz, ref_x=inboard_iso[0], ref_z=inboard_iso[1])]
        )

        coils = []
        for i, (x, z, dx, dz) in enumerate(zip(xc, zc, dxc, dzc)):
            coil = SymmetricCircuit(
                Coil(x=x, z=z, dx=dx, dz=dz, name=f"PF_{i+1}", ctype="PF")
            )
            coils.append(coil)
        coilset = CoilSet(coils)
        grid = Grid.from_eqdict(eq_dict)
