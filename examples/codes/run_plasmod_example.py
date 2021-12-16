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
Test for plasmod run
"""
from bluemira.base.config import Configuration
import bluemira.codes.plasmod as plasmod
import matplotlib.pyplot as plt

PLASMOD_PATH = "../plasmod_bluemira"

new_params = {
    "A": 3.1,
    "B_0": 5.3,
    "R_0": 8.93,
    "q_95": 3.23,
}
build_config = {
    "problem settings": {
        "Pfus_req": 2000,
        "i_modeltype": 111,
    },
    "mode": "run",
    "binary": f"{PLASMOD_PATH}/plasmod.o",
}

plasmod_solver = plasmod.Solver(
    params=Configuration(new_params),
    build_config=build_config,
)
# plasmod_solver._set_runmode("run")
plasmod_solver.run()

ffprime = plasmod_solver.get_profile("ffprime")
Te = plasmod_solver.get_profile("Te")
x = plasmod_solver.get_profile("x")
fig, ax = plt.subplots()
ax.plot(x, Te)
ax.set(xlabel="x (-)", ylabel="T_e (keV)")
ax.grid()
plt.show()

print(f"Plasma current [MA]: {plasmod_solver.params.I_p.value}")
print(f"Fusion power [MW]: {plasmod_solver.params.P_fus.value/1E6}")
print(f"Additional heating power [MW]: {plasmod_solver.get_scalar('Padd')/1E6}")
print(f"Radiation power [MW]: {plasmod_solver.get_scalar('Prad')/1E6}")
print(f"Transport power across separatrix [MW]: {plasmod_solver.get_scalar('Psep')/1E6}")
