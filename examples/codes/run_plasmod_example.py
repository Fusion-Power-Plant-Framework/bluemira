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
import matplotlib.pyplot as plt

import bluemira.codes.plasmod as plasmod
from bluemira.base.config import Configuration
# from bluemira.base.logs import set_log_level
# set_log_level('DEBUG')



def run_plasmod_test(params, build):
    plasmod_solver = plasmod.Solver(
        params=Configuration(params),
        build_config=build,
    )

    plasmod_solver._set_runmode("run")
    plasmod_solver.run()

    return plasmod_solver


def print_outputs(plasmod_solver):
    print(f"Fusion power [MW]: {plasmod_solver.get_scalar('Pfus')/ 1E6}")
    print(f"Additional heating power [MW]: {plasmod_solver.get_scalar('Paux') / 1E6}")
    print(f"Radiation power [MW]: {plasmod_solver.get_scalar('Prad') / 1E6}")
    print(f"Transport power across separatrix [MW]: {plasmod_solver.get_scalar('Psep') / 1E6}")
    print(f"Safety factor 95 % [-]: {plasmod_solver.get_scalar('q95')}")
    print(f"Plasma current [MA]: {plasmod_solver.get_scalar('Ip')}")
    print(f"Plasma internal inductance [-]: {plasmod_solver.get_scalar('rli')}")
    print(f"Loop voltage [V]: {plasmod_solver.get_scalar('v_loop')}")
    print(f"Effective charge [-]: {plasmod_solver.get_scalar('Zeff')}")
    print(f"H-factor [-]: {plasmod_solver.get_scalar('Hfact')}")
    print(f"Divertor challenging criterion (P_sep * Bt /(q95 * R0 * A)) [-]: {plasmod_solver.get_scalar('psepb_q95AR')}")
    print(f"H-mode operating regime f_LH = P_sep/P_LH [-]: {plasmod_solver.get_scalar('Psep')/plasmod_solver.get_scalar('PLH')}")
    print(f"Energy confinement time [-]: {plasmod_solver.get_scalar('taueff')}")
    print(f"Protium fraction [-]: {plasmod_solver.get_scalar('cprotium')}")
    print(f"Helium fraction [-]: {plasmod_solver.get_scalar('che')}")
    print(f"Xenon fraction [-]: {plasmod_solver.get_scalar('cxe')}")
    print(f"Argon fraction [-]: {plasmod_solver.get_scalar('car')}")


def plot_profile(plasmod_solver, var_name, var_unit):

    prof = plasmod_solver.get_profile(var_name)
    x = plasmod_solver.get_profile("x")
    fig, ax = plt.subplots()
    ax.plot(x, prof)
    ax.set(xlabel="x (-)", ylabel= var_name + " (" + var_unit + ")")
    ax.grid()
    plt.show()


# %% running plasmod for demoH.i reference configuration
# H-factor is set as input

PLASMOD_PATH = "/home/fabrizio/bwSS/plasmod/bin"

new_params = {
    "A": 3.1,
    "R_0": 9.002,
    "I_p": 17.75,
    "B_0": 5.855,
    "V_p": -2500,
    "v_burn": -1.0e6,
    "kappa_95": 1.652,
    "delta_95": 0.333,
    "delta": 0.38491934960310104,
    "kappa": 1.6969830041844367
}

build_config = {
    "problem_settings": {
        "pfus_req": 0.0,
        "pheat_max": 130.0,
        "q_control": 130.0,
        "Hfact": 1.1,
        "i_modeltype": "GYROBOHM_1",
        "i_equiltype": "Ip_sawtooth",
        "i_pedestal": "SAARELMA"
    },
    "mode": "run",
    "binary": f"{PLASMOD_PATH}/plasmod",
}

plasmod_sol = run_plasmod_test(new_params, build_config)
print_outputs(plasmod_sol)
#plot_profile(plasmod_sol, "Te", "keV")


# %% changing transport model, with H factor calculated
build_config['problem_settings']['i_modeltype'] = 'GYROBOHM_2'
plasmod_sol = run_plasmod_test(new_params, build_config)
print_outputs(plasmod_sol)


# %% fixing fusion power to 2000 MW and safety factor q_95 to 3.5.
# plasmod calculates the additional heating power and the plasma current

build_config['problem_settings']['pfus_req'] = 2000.
build_config['problem_settings']['i_equiltype'] = 'q95_sawtooth'
build_config['problem_settings']['q_control'] = 50.
new_params['q_95'] = 3.5
plasmod_sol = run_plasmod_test(new_params, build_config)
print_outputs(plasmod_sol)

# %% setting heat flux on divertor target to 10 MW/m²
# plasmod calculates the argon concentration to fulfill the constraint

build_config['problem_settings']['qdivt_sup'] = 10.
plasmod_sol = run_plasmod_test(new_params, build_config)
print_outputs(plasmod_sol)
