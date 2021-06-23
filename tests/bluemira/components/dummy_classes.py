# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I. Maione, S. McIntosh, J. Morris,
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

from bluemira.components.base import PhysicalComponent


class DummyDivertorProfile(PhysicalComponent):
    """
    Builds the divertor profile based on the desired reference equilibria
    Needs a Nova StreamFlow object
    """

    # fmt: off
    default_params = [
        ["fw_psi_n", "Normalised psi boundary to fit FW to", 1.07, "N/A", None, "Input"],
        ["fw_dx", "Minimum distance of FW to separatrix", 0.225, "m", None, "Input"],
        ["div_L2D_ib", "Inboard divertor leg length", 1.1, "m", None, "Input"],
        ["div_L2D_ob", "Outboard divertor leg length", 1.3, "m", None, "Input"],
        ["div_graze_angle", "Divertor SOL grazing angle", 1.5, "°", None, "Input"],
        ["div_psi_o", "Divertor flux offset", 0.5, "m", None, "Input"],
        ["div_Ltarg", "Divertor target length", 0.5, "m", None, "Input"],
        ["tk_div", "Divertor thickness", 0.5, "m", None, "Input"],
        ["dx_div", "Don't know", 0, "N/A", None, "Input"],
        ["bb_gap", "Gap to breeding blanket", 0.05, "m", None, "Input"],
        ["Xfw", "Don't know", 0, "N/A", None, "Input"],
        ["Zfw", "Don't know", 0, "N/A", None, "Input"],
        ["psi_fw", "Don't know", 0, "N/A", None, "Input"],
        ["c_rm", "Remote maintenance clearance", 0.05, "m", "Distance between IVCs", "Input"],
    ]
    # fmt: on


class DummyBreedingBlanket(PhysicalComponent):
    # fmt: off
    default_params = [
        ['n_TF', 'Number of TF coils', 16, 'N/A', None, 'Input'],
        ['plasma_type', 'Type of plasma', 'SN', 'N/A', None, 'Input'],
        ['A', 'Plasma aspect ratio', 3.1, 'N/A', None, 'Input'],
        ['R_0', 'Major radius', 9, 'm', None, 'Input'],
        ['blanket_type', 'Blanket type', 'HCPB', 'N/A', None, 'Input'],
        ['g_vv_bb', 'Gap between VV and BB', 0.02, 'm', None, 'Input'],
        ['c_rm', 'Remote maintenance clearance', 0.02, 'm', 'Distance between IVCs', None],
        ["bb_e_mult", "Energy multiplication factor", 1.35, "N/A", None, "HCPB classic"],
        ['bb_min_angle', 'Mininum BB module angle', 70, '°', 'Sharpest cut of a module possible', 'Lorenzo Boccaccini said this in a meeting in 2015, Garching, Germany'],
        ['fw_dL_min', 'Minimum FW module length', 0.75, 'm', None, 'Input'],
        ['fw_dL_max', 'Maximum FW module length', 3, 'm', 'Cost+manufacturing implications', 'Input'],
        ['fw_a_max', 'Maximum angle between FW modules', 20, '°', None, 'Input'],
        ['rho', 'Blanket density', 3000, 'kg/m^3', 'Homogenous', None],
        ['tk_bb_ib', 'Inboard blanket thickness', 0.8, 'm', None, 'Input'],
        ['tk_bb_ob', 'Outboard blanket thickness', 1.1, 'm', None, 'Input'],
        ['tk_bb_fw', 'First wall thickness', 0.025, 'm', None, 'Input'],
        ['tk_bb_arm', 'Tungsten armour thickness', 0.002, 'm', None, 'Input'],
        ["tk_r_ib_bz", "Thickness ratio of the inboard blanket breeding zone", 0.309, "N/A", None, "HCPB 2015 design description document"],
        ["tk_r_ib_manifold", "Thickness ratio of the inboard blanket manifold", 0.114, "N/A", None, "HCPB 2015 design description document"],
        ["tk_r_ib_bss", "Thickness ratio of the inboard blanket back supporting structure", 0.577, "N/A", None, "HCPB 2015 design description document"],
        ["tk_r_ob_bz", "Thickness ratio of the outboard blanket breeding zone", 0.431, "N/A", None, "HCPB 2015 design description document"],
        ["tk_r_ob_manifold", "Thickness ratio of the outboard blanket manifold", 0.071, "N/A", None, "HCPB 2015 design description document"],
        ["tk_r_ob_bss", "Thickness ratio of the outboard blanket back supporting structure", 0.498, "N/A", None, "HCPB 2015 design description document"],
    ]
    # fmt: on


class DummyDivertor(PhysicalComponent):
    # fmt: off
    default_params = [
        ['n_TF', 'Number of TF coils', 16, 'N/A', None, 'Input'],
        ['plasma_type', 'Type of plasma', 'SN', 'N/A', None, 'Input'],
        ['n_div_cassettes', 'Number of divertor cassettes per sector', 3, 'N/A', None, "Common decision"],
        ['coolant', 'Coolant', 'Water', None, 'Divertor coolant type', 'Common sense'],
        ['T_in', 'Coolant inlet T', 80, '°C', 'Coolant inlet T', None],
        ['T_out', 'Coolant outlet T', 120, '°C', 'Coolant inlet T', None],
        ['P_in', 'Coolant inlet P', 8, 'MPa', 'Coolant inlet P', None],
        ['dP', 'Coolant pressure drop', 1, 'MPa', 'Coolant pressure drop', None],
        ['rm_cl', 'RM clearance', 0.02, 'm', 'Radial and poloidal clearance between in-vessel components', 'Not so common sense']
    ]
    # fmt: on
