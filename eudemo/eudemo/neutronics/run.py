# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Test script to make the CSG branch work."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from matproplib.library.fluids import Void

from bluemira.codes.openmc.solver import OpenMCDAGMCNeutronicsSolver
from bluemira.codes.openmc.sources import make_tokamak_source
from bluemira.codes.wrapper import neutronics_code_solver
from bluemira.radiation_transport.neutronics.blanket_data import (
    BlanketType,
    create_materials,
    get_preset_geometry,
)
from bluemira.radiation_transport.neutronics.geometry import TokamakDimensions
from bluemira.radiation_transport.neutronics.neutronics_axisymmetric import (
    NeutronicsReactor,
)

if TYPE_CHECKING:
    import numpy.typing as npt
    from matproplib.conditions import OperationalConditions

    from bluemira.base.parameter_frame import ParameterFrame
    from bluemira.base.reactor import ComponentManager
    from bluemira.codes.openmc.output import OpenMCCSGResult
    from bluemira.codes.openmc.solver import NeutronSourceCreator
    from bluemira.equilibria.equilibrium import Equilibrium
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
            blanket.panel_points.T,
            ivc_shapes.outer_boundary,
            vacuum_vessel.xz_boundary,
        )


def run_csg_neutronics(
    params: ParameterFrame,
    build_config: dict,
    blanket: ComponentManager,
    vacuum_vessel: ComponentManager,
    ivc_shapes: IVCShapes,
    eq: Equilibrium,
    op_cond: OperationalConditions,
    source: NeutronSourceCreator | None = None,
    tally_function=None,
) -> tuple[EUDEMONeutronicsCSGReactor, OpenMCCSGResult | dict[int, float]]:
    """Runs the neutronics model

    Returns
    -------
    neutronics_csg:
        The neutronics CSG reactor model
    res:
        The result of the neutronics run

    Raises
    ------
    NeutronicsError
        Can't import default neutron source
    """
    blanket_type = BlanketType(build_config.pop("blanket_type"))
    tokamak_geometry = get_preset_geometry(params)
    # TODO get these materials from the physical components
    material_library = create_materials(blanket_type)

    params.update_from_dict(
        {
            "inboard_fw_tk": {"value": tokamak_geometry.inb_fw_thick, "unit": "m"},
            "inboard_breeding_tk": {"value": tokamak_geometry.inb_bz_thick, "unit": "m"},
            "outboard_fw_tk": {"value": tokamak_geometry.outb_fw_thick, "unit": "m"},
            "outboard_breeding_tk": {
                "value": tokamak_geometry.outb_bz_thick,
                "unit": "m",
            },
        },
        source="Neutronics",
    )
    neutronics_csg = EUDEMONeutronicsCSGReactor(
        params, ivc_shapes, blanket, vacuum_vessel, material_library
    )

    solver = neutronics_code_solver(
        params,
        build_config,
        neutronics_csg,
        eq,
        source=source or make_tokamak_source,
        op_cond=op_cond,
        tally_function=tally_function,
    )

    outputs = solver.execute(build_config.get("run_mode", "run"))

    if len(outputs) == 2:  # noqa: PLR2004
        res = outputs[0]
        params.update_from_frame(outputs[1])
    else:
        res = outputs

    return neutronics_csg, res


def export_dagmc_model(reactor, build_config):
    """
    Export the reactor model to a DAGMC model.

    Parameters
    ----------
    reactor : EUDEMO
        The reactor instance to export.
    build_config : dict
        The build configuration parameters.
    """
    if build_config.get("export_dagmc_model", False):
        reactor.save_cad(
            directory=build_config.get("dagmc_export_dir", None),
            cad_format="dagmc",
            construction_params={
                "without_components": [
                    reactor.plasma,
                    reactor.blanket,
                    reactor.coil_structures,
                ],
                "group_by_materials": True,
            },
        )


def run_dagmc_neutronics(
    reactor,
    params,
    build_config,
    eq: Equilibrium,
    source: NeutronSourceCreator | None = None,
    tally_function=None,
):
    """Creates and runs the DAGMC neutronics model"""  # noqa: DOC201
    export_dagmc_model(reactor, build_config)

    mats = {"undef_material": Void(name="undef_material")}
    for m in reactor.materials:
        if m is not None and m.name not in mats:
            mats[m.name] = m

    solver = OpenMCDAGMCNeutronicsSolver(
        params,
        build_config,
        eq,
        source=source or make_tokamak_source,
        dagmc_model_path=Path(
            build_config.get("dagmc_export_dir", Path.cwd()), f"{reactor.name}.h5m"
        ),
        materials=[
            m.convert("openmc", {"temperature": 298, "pressure": 101325})
            for m in mats.values()
        ],
        tally_function=tally_function,
    )

    return solver.execute(build_config.get("run_mode", "run"))
