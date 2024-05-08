# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Test script to make the CSG branch work."""

from __future__ import annotations

from typing import TYPE_CHECKING

from bluemira.neutronics.blanket_data import (
    create_materials,
    get_preset_physical_properties,
)
from bluemira.neutronics.neutronics_axisymmetric import (
    NeutronicsReactor,
    NeutronicsReactorParameterFrame,
)
from bluemira.neutronics.openmc.solver import (
    OpenMCNeutronicsSolver,
)
from bluemira.neutronics.openmc.sources import make_pps_source

if TYPE_CHECKING:
    from collections.abc import Callable

    import openmc.source

    from bluemira.base.parameter_frame import ParameterFrame
    from bluemira.base.reactor import ComponentManager
    from bluemira.geometry.wire import BluemiraWire
    from bluemira.neutronics.params import PlasmaSourceParameters
    from eudemo.blanket import Blanket
    from eudemo.ivc import IVCShapes
    from eudemo.vacuum_vessel import VacuumVessel


class EUDEMONeutronicsCSGReactor(NeutronicsReactor):
    def _get_wires_from_components(
        self,
        divertor: IVCShapes,
        blanket: Blanket,
        vacuum_vessel: VacuumVessel,
    ) -> tuple[BluemiraWire, npt.NDArray, BluemiraWire, BluemiraWire]:
        return (
            divertor.div_internal_boundary,
            blanket.panel_points().T,
            divertor.outer_boundary,
            vacuum_vessel.xz_boundary(),
        )


def run_neutronics(
    params: dict | ParameterFrame,
    build_config: dict,
    blanket: ComponentManager,
    divertor: ComponentManager,
    vacuum_vessel: ComponentManager,
    ivc_shapes: IVCShapes,
    source: Callable[[PlasmaSourceParameters], openmc.source.SourceBase] | None = None,
):
    """Runs the neutronics model"""
    # TODO get these materials from the componentmanager or something similar
    breeder_materials, tokamak_geometry = get_preset_physical_properties(
        build_config.pop("blanket_type")
    )
    material_library = create_materials(breeder_materials)

    csg_params = NeutronicsReactorParameterFrame.from_dict({
        "inboard_fw_tk": {"value": tokamak_geometry.inb_fw_thick, "unit": "m"},
        "inboard_breeding_tk": {"value": tokamak_geometry.inb_bz_thick, "unit": "m"},
        "outboard_fw_tk": {"value": tokamak_geometry.outb_fw_thick, "unit": "m"},
        "outboard_breeding_tk": {"value": tokamak_geometry.outb_bz_thick, "unit": "m"},
        "blanket_io_cut": {"value": params.global_params.R_0.value, "unit": "m"},
        "tf_inner_radius": {"value": 2, "unit": "m"},
        "tf_outer_radius": {"value": 4, "unit": "m"},
        "divertor_surface_tk": {"value": 0.1, "unit": "m"},
        "blanket_surface_tk": {"value": 0.01, "unit": "m"},
        "blk_ib_manifold": {"value": 0.02, "unit": "m"},
        "blk_ob_manifold": {"value": 0.2, "unit": "m"},
    })
    neutronics_csg = EUDEMONeutronicsCSGReactor(
        csg_params, ivc_shapes, blanket, vacuum_vessel, material_library
    )

    obj = OpenMCNeutronicsSolver(
        params,
        build_config,
        neutronics_csg,
        source=source or make_pps_source,
    )

    return obj.execute()
