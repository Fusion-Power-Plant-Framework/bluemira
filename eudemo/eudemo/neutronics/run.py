# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Test script to make the CSG branch work."""

from __future__ import annotations

from typing import TYPE_CHECKING

from bluemira.codes.wrapper import neutronics_code_solver
from bluemira.radiation_transport.error import NeutronicsError
from bluemira.radiation_transport.neutronics.blanket_data import (
    create_materials,
    get_preset_physical_properties,
)
from bluemira.radiation_transport.neutronics.geometry import TokamakDimensions
from bluemira.radiation_transport.neutronics.neutronics_axisymmetric import (
    NeutronicsReactor,
    NeutronicsReactorParameterFrame,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy.typing as npt
    import openmc.source

    from bluemira.base.parameter_frame import ParameterFrame
    from bluemira.base.reactor import ComponentManager
    from bluemira.codes.openmc.output import OpenMCResult
    from bluemira.codes.openmc.params import PlasmaSourceParameters
    from bluemira.geometry.wire import BluemiraWire
    from eudemo.blanket import Blanket
    from eudemo.ivc import IVCShapes
    from eudemo.vacuum_vessel import VacuumVessel


class EUDEMONeutronicsCSGReactor(NeutronicsReactor):
    """EUDEMO Axis-symmetric neutronics model"""

    def _get_wires_from_components(
        self,
        ivc_shapes: IVCShapes,
        blanket: Blanket,
        vacuum_vessel: VacuumVessel,
    ) -> tuple[TokamakDimensions, BluemiraWire, npt.NDArray, BluemiraWire, BluemiraWire]:
        return (
            TokamakDimensions.from_parameterframe(self.params, blanket.r_inner_cut),
            ivc_shapes.div_internal_boundary,
            blanket.panel_points().T,
            ivc_shapes.outer_boundary,
            vacuum_vessel.xz_boundary(),
        )


def run_neutronics(
    params: dict | ParameterFrame,
    build_config: dict,
    blanket: ComponentManager,
    vacuum_vessel: ComponentManager,
    ivc_shapes: IVCShapes,
    source: Callable[[PlasmaSourceParameters], openmc.source.SourceBase] | None = None,
    tally_function=None,
) -> tuple[EUDEMONeutronicsCSGReactor, OpenMCResult | dict[int, float]]:
    """Runs the neutronics model"""
    # TODO get these materials from the componentmanager or something similar
    breeder_materials, tokamak_geometry = get_preset_physical_properties(
        build_config.pop("blanket_type")
    )
    material_library = create_materials(breeder_materials)

    csg_params = NeutronicsReactorParameterFrame.from_config_params(params)
    csg_params.update_from_dict({
        "inboard_fw_tk": {"value": tokamak_geometry.inb_fw_thick, "unit": "m"},
        "inboard_breeding_tk": {"value": tokamak_geometry.inb_bz_thick, "unit": "m"},
        "outboard_fw_tk": {"value": tokamak_geometry.outb_fw_thick, "unit": "m"},
        "outboard_breeding_tk": {"value": tokamak_geometry.outb_bz_thick, "unit": "m"},
    })
    neutronics_csg = EUDEMONeutronicsCSGReactor(
        csg_params, ivc_shapes, blanket, vacuum_vessel, material_library
    )
    if source is None:
        try:
            from bluemira.codes.openmc.sources import make_pps_source  # noqa: PLC0415
        except ImportError:
            raise NeutronicsError("Cannot import neutronics source") from None

    solver = neutronics_code_solver(
        params,
        build_config,
        neutronics_csg,
        source=source or make_pps_source,
        tally_function=tally_function,
    )

    res = solver.execute()

    return neutronics_csg, res
