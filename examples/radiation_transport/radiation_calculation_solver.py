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
Example core and scraper-off layer radiation source calculation
"""

import os

# %%
from cgi import test

import matplotlib.pyplot as plt

import bluemira.codes.process as process
from bluemira.base.config import Configuration
from bluemira.base.file import get_bluemira_path
from bluemira.base.parameter import ParameterFrame
from bluemira.equilibria import Equilibrium
from bluemira.geometry._deprecated_loop import Loop
from bluemira.radiation_transport.radiation_profile import RadiationSolver
from bluemira.radiation_transport.temp_fs_maker import TempFsSolver

# %%[markdown]
# Read equilibrium

# %%
read_path = get_bluemira_path("equilibria", subfolder="data")
eq_name = "DN-DEMO_eqref.json"
eq_name = os.sep.join([read_path, eq_name])
eq = Equilibrium.from_eqdsk(eq_name, load_large_file=True)

# %%[markdown]
# Get first wall shape

# %%
read_path = get_bluemira_path("radiation_transport/test_data", subfolder="tests")
fw_name = "DN_fw_shape.json"
fw_name = os.sep.join([read_path, fw_name])
fw_shape = Loop.from_file(fw_name)

# %%[markdown]
# Run particle solver

# %%
p_solver_params = ParameterFrame()
solver = TempFsSolver(p_solver_params, eq, dx_mp=0.001)
solver.analyse(first_wall=fw_shape)

# %%[markdown]
# Run PROCESS solver

# %%
PROCESS_PATH = ""
binary = f"{PROCESS_PATH}process"

new_params = {
    "kappa": 1.6969830041844367,
}
params = Configuration(new_params)

build_config = {
    "mode": "run",
    "binary": binary,
}

process_solver = process.Solver(
    params=params,
    build_config=build_config,
)
process_solver.run()

# %%[markdown]
# Get impurity fractions

# %%
impurity_content = {
    "H": process_solver.get_species_fraction("H"),
    "He": process_solver.get_species_fraction("He"),
    "Xe": process_solver.get_species_fraction("Xe"),
    "W": process_solver.get_species_fraction("W"),
}

# %%[markdown]
# Get impurity data: temperature reference
# and radiative loss function reference

# %%
impurity_data = {
    "H": {
        "T_ref": process_solver.get_species_data("H")[0],
        "L_ref": process_solver.get_species_data("H")[1],
    },
    "He": {
        "T_ref": process_solver.get_species_data("He")[0],
        "L_ref": process_solver.get_species_data("He")[1],
    },
    "Xe": {
        "T_ref": process_solver.get_species_data("Xe")[0],
        "L_ref": process_solver.get_species_data("Xe")[1],
    },
    "W": {
        "T_ref": process_solver.get_species_data("W")[0],
        "L_ref": process_solver.get_species_data("W")[1],
    },
}

# %%[markdown]
# Eventually customise plasma parameters to deviate from default values

# %%
plasma_params = ParameterFrame(
    # fmt: off
    [
        ["kappa", "Elongation", 3, "dimensionless", None, "Input"],
    ]
    # fmt: on
)

# %%[markdown]
# Run core radiation source calculation for Spherical Tokamak

# %%
rad_solver = RadiationSolver(eq, solver, plasma_params, impurity_content, impurity_data)
x, z, rad = rad_solver.analyse(fw_shape)

core_rad_by_psi = rad_solver.rad_core_by_psi_n(0.9)
print(core_rad_by_psi)

core_rad_by_point = rad_solver.rad_core_by_points(12, -0.5)
print(core_rad_by_point)

rdm_rad_by_psi_1 = rad_solver.rad_by_psi_n(0.9)
print(rdm_rad_by_psi_1)

rdm_rad_by_psi_2 = rad_solver.rad_by_psi_n(1.05)
print(rdm_rad_by_psi_2)

rdm_rad_by_point_1 = rad_solver.rad_by_points(9.5, 5.35)
print(rdm_rad_by_point_1)

rdm_rad_by_point_2 = rad_solver.rad_by_points(12, -0.5)
print(rdm_rad_by_point_2)


rad_solver.plot()

plt.show()
