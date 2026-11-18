# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
Axissymetric Neutronics Reactor
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING

import mcio
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.gp import gp_Pnt, gp_Trsf

from bluemira.base.look_and_feel import bluemira_print
from bluemira.base.parameter_frame import ParameterFrame, make_parameter_frame
from bluemira.radiation_transport.axisymmetric_neutronics.geometry import (
    AxisymmetricNeutronicsGeomManager,
)
from bluemira.radiation_transport.axisymmetric_neutronics.materials import (
    to_mcio_material,
)
from bluemira.radiation_transport.axisymmetric_neutronics.plotting import (
    plot_desplining_comparisons,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matproplib.conditions import OperationalConditions
    from openmc.model.model import Model

    from bluemira.base.components import Component
    from bluemira.base.parameter_frame.typed import ParameterFrameLike
    from bluemira.base.reactor import Reactor


class NeutronicsModel(Enum):
    """
    Enumeration of different model to be
    considered for neutronics simulations.

    The interpretation of the enums would
    be user-determined through the abstract
    methods defined in the NeutronicsReactor
    class
    """

    DEBUG = auto()
    SIMPLE = auto()
    ADVANCED = auto()

    @classmethod
    def _missing_(cls, value: object):
        if isinstance(value, str):
            member = cls.__members__.get(value.upper())
            if member is not None:
                return member

        raise ValueError(
            f"{cls.__name__} has no type {value}. "
            f"Please select from {tuple(cls.__members__)}"
        ) from None


@dataclass
class AxisymmetricNeutronicsParameterFrame(ParameterFrame):
    """Neutronics reactor parameters"""


class AxisymmetricNeutronicsReactor(ABC):
    """
    Axisymmetric Neutronics Reactor
    """

    param_cls = AxisymmetricNeutronicsParameterFrame

    def __init__(
        self,
        neutronics_config: dict,
        params: ParameterFrameLike,
        model: NeutronicsModel,
        reactor: Reactor,
    ):
        bluemira_print("Creating axisymmetric CSG neutronics model")

        self.config = neutronics_config
        self.params = make_parameter_frame(params, self.param_cls)
        self.model = model
        self.geometry = self.get_desplined_geometry(reactor)
        self.material_name_mapping = None
        self.material_library = None  # No need

    def get_desplined_geometry(
        self, reactor: Reactor
    ) -> AxisymmetricNeutronicsGeomManager:
        """
        Creates a neutronics geometry manager from components,
        subject to the neutronics model

        Returns
        -------
        :
            Neutronics Geometry Manager
        """
        all_comps = self.get_all_components(reactor)
        discretisations = [
            self.default_discretisations(comp.name)
            or self.default_discretisations(comp.parent().name)
            for comp in all_comps
        ]
        desplined_geometry = AxisymmetricNeutronicsGeomManager.from_list_of_components(
            components=all_comps,
            discretisations=discretisations,
            overlap_tolerance=self.config.get("overlap_tolerance", 1e-10),
            gap_tolerance=self.config.get("gap_tolerance", 1e-6),
        )

        plot_comparisons = self.config.get("plot_comparisons", False)
        if plot_comparisons:
            neutronics_output_path = Path(
                self.config.get("neutronics_output_path", "./outputs/")
            )
            plot_desplining_comparisons(
                orig_comps=[
                    xz_comp
                    for comp in all_comps
                    if (xz_comp := comp.get_component("xz")) is not None
                ],
                desplined_comps=desplined_geometry.component().get_component(
                    "xz", first=False
                ),
                output_path=neutronics_output_path,
            )
        return desplined_geometry

    def plot_2d(self, *args, **kwargs):
        """
        Plot neutronics reactor 2d profile

        Returns
        -------
        :
            Axes on which the reactor is plotted.
        """
        show = kwargs.pop("show", True)
        ax = kwargs.pop("ax", None)
        return self.geometry.plot("xz", *args, ax=ax, show=show, **kwargs)

    @staticmethod
    def _scale_solid_to_cm(solid):
        """
        Scale an OCC solid from metres to centimetres.

        Returns
        -------
        :
            Scaled OCC shape with coordinates in centimetres.
        """
        transform = gp_Trsf()
        transform.SetScale(gp_Pnt(0, 0, 0), 100.0)

        return BRepBuilderAPI_Transform(
            solid,
            transform,
            theCopyGeom=True,
        ).Shape()

    @staticmethod
    def _get_global_bounding_box(
        components: Sequence[Component],
    ) -> Sequence[float] | None:
        """
        Calculate the global bounding box in metres.

        Returns
        -------
        :
            global bounding box in metres
        """
        if not components:
            return None

        boxes = [
            comp.get_component("xyz").children[0].shape.bounding_box
            for comp in components
        ]

        r_min = min(box.x_min for box in boxes)
        r_max = max(box.x_max for box in boxes)
        z_min = min(box.z_min for box in boxes)
        z_max = max(box.z_max for box in boxes)

        return z_max, z_min, r_max, r_min

    def run_mcio(
        self,
        op_cond: OperationalConditions,
    ) -> tuple[Model, Sequence[float] | None]:
        """
        Runs MCIO interface to create openmc geometry for neutronics

        Returns
        -------
        :
            openmc Model
        """
        all_desplined_components = self.geometry.get_all_components()

        # Calculate global bounding box (m)
        bounding_box = self._get_global_bounding_box(all_desplined_components)

        all_names = [comp.name for comp in all_desplined_components]

        # Convert OCC solids from metres to centimetres for OpenMC uses
        all_ocp_solids = [
            self._scale_solid_to_cm(
                comp.get_component("xyz").children[0].shape.shape.wrapped
            )
            for comp in all_desplined_components
        ]
        # all_ocp_solids = [
        #     comp.get_component("xyz").children[0].shape.shape.wrapped
        #     for comp in all_desplined_components
        # ]

        all_materials = [
            comp.get_component("xyz").children[0].material
            for comp in all_desplined_components
        ]

        # set up empty decomposer (you can also add step files and ocp
        # solids if you like)
        decomposer = mcio.decompose.Decomposer()
        # Manipulate Void generation in mcio
        decomposer.decom_params.void_max_recursion_depth = 5
        decomposer.decom_params.void_min_width = 50

        material_name_mapping = {}

        # NOTE: Every solid should have its own unique MCIO material name,
        # even if multiple solids use the same original material.
        bluemira_print("Adding solids to the decomposer")
        for mat_number, (solid, mat, name) in enumerate(
            zip(all_ocp_solids, all_materials, all_names, strict=True)
        ):
            # Renames MCIO material to the accepted format, e.g. "m0", "m1", ...
            mcio_material = to_mcio_material(mat, op_cond, mat_number + 1)
            decomposer.add_solid(solid, name=name, material=mcio_material)

            # Map original component name to MCIO material name
            material_name_mapping[name] = mcio_material.name

        # save material mapping
        self.material_name_mapping = material_name_mapping

        # run decomposer to generate void and model
        bluemira_print("Running decomposer")
        decomposer.run()

        return decomposer.model.openmc_model, bounding_box

    @abstractmethod
    def customised_tallies(self, model):
        """
        Abstract Methods for user to be able
        to add customised tallies for their particular
        reactor
        """
        ...

    @abstractmethod
    def get_all_components(self, reactor: Reactor) -> list[Component]:
        """
        Get all relevant components for the reactor

        Returns
        -------
        :
            list of components.
        """
        ...

    @staticmethod
    @abstractmethod
    def default_discretisations(comp_name: str) -> int:
        """
        Return the default spline discretisation for a component.

        Returns
        -------
        :
            default discretisation
        """
        ...
