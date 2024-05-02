# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Test script to make the CSG branch work."""

from pathlib import Path

from bluemira.base.constants import raw_uc
from bluemira.display import plot_2d, plot_3d, show_cad  # noqa: F401
from bluemira.neutronics.make_materials import BlanketType
from bluemira.neutronics.neutronics_axisymmetric import NeutronicsReactor
from bluemira.neutronics.openmc.solver import (
    OpenMCNeutronicsSolver,
    OpenMCNeutronicsSolverParams,
)
from bluemira.neutronics.openmc.sources import make_pps_source

build_config = {
    "cross_section_xml": Path(
        "~/Documents/BLUEPRINT/cross_section_data/cross_section_data/cross_sections.xml"
    ).expanduser(),
    "particles": 16800,  # 16800 takes 5 seconds,  1000000 takes 280 seconds.
    "batches": 3,
    "photon_transport": True,
    "electron_treatment": "ttb",
    "run_mode": "run_and_plot",
    "openmc_write_summary": False,
    "parametric_source": True,
    "blanket_type": BlanketType.HCPB,
    "plot_axis": "xz",
    "plot_pixel_per_metre": 100,
}

params = OpenMCNeutronicsSolverParams.from_dict({
    "major_radius": {"value": 8.938, "unit": "m"},
    "aspect_ratio": {"value": 8.938 / 2.8938, "unit": "m"},
    "elongation": {"value": 1.65, "unit": ""},
    "triangularity": {"value": 0.333, "unit": ""},
    "reactor_power": {"value": 1998, "unit": "MW"},
    "peaking_factor": {"value": 1.508, "unit": ""},
    "temperature": {"value": raw_uc(15.4, "keV", "K"), "unit": "K"},
    "shaf_shift": {"value": 0, "unit": "m"},
    "vertical_shift": {"value": 0, "unit": "m"},
})


obj = OpenMCNeutronicsSolver(
    params,
    NeutronicsReactor(params, None, None, None),
    source=make_pps_source,
    build_config=build_config,
)

print(obj.execute())
