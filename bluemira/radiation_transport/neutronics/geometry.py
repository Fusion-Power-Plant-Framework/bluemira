# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""dataclasses containing parameters used to set up the openmc model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from bluemira.radiation_transport.neutronics.neutronics_axisymmetric import (
        NeutronicsReactorParameterFrame,
    )


@dataclass
class BlanketThickness:
    """
    Give the depth of the interfaces between blanket layers.

    Parameters
    ----------
    surface
        Thickness of the surface layer of the blanket. Can be zero.
        Only used for tallying purpose, i.e. not a physical component.
    first_wall
        Thickness of the first wall.
    breeding_zone
        Thickness of the breedng zone. Could be zero if the breeding zone is absent.

    Note
    ----
    Thickness of the vacuum vessel is not required because we we assume it fills up the
    remaining space between the manifold's end and the outer_boundary.
    """

    surface: float
    first_wall: float
    breeding_zone: float
    manifold: float

    def get_interface_depths(self):
        """Return the depth of the interface layers"""
        return np.cumsum([
            self.surface,
            self.first_wall,
            self.breeding_zone,
        ])


@dataclass
class DivertorThickness:
    """
    Divertor dimensions.
    For now it only has 1 value: the surface layer thickness.

    Parameters
    ----------
    surface
        The surface layer of the divertor, which we expect to be made of a different
        material (e.g. Tungsten or alloy of Tungsten) from the bulk support & cooling
        structures of the divertor.
    """

    surface: float


@dataclass
class ToroidalFieldCoilDimension:
    """
    Gives the toroidal field coil diameters. Working with the simplest assumption, we
    assume that the tf coil is circular for now.

    Parameters
    ----------
    inner_diameter
        (i.e. inner diameter of the windings.)
    outer_diameter
        Outer diameter of the windings.
    """

    thickness: float
    inner_radius: float


@dataclass
class RadiationShieldThickness:
    """
    Radiation shield dimensions.
    For now it only has 1 value: the wall layer thickness.

    Parameters
    ----------
    wall
        The wall thickness of the radiation shield.
    """

    wall: float


@dataclass
class TokamakDimensions:
    """
    The dimensions of the simplest axis-symmetric case of the tokamak.

    Parameters
    ----------
    inboard
        thicknesses of the inboard blanket
    outboard
        thicknesses of the outboard blanket
    divertor
        thicknesses of the divertor components
    cs_coil
        diameters of the toroidal field coil in the
    """

    inboard: BlanketThickness
    inboard_outboard_transition_radius: float
    outboard: BlanketThickness
    divertor: DivertorThickness
    cs_coil: ToroidalFieldCoilDimension
    rad_shield: RadiationShieldThickness

    @classmethod
    def from_parameterframe(
        cls, params: NeutronicsReactorParameterFrame, r_inner_cut: float
    ):
        """Setup tokamak dimensions"""
        return cls(
            BlanketThickness(
                params.fw_blanket_surface_tk.value,
                params.inboard_fw_tk.value,
                params.inboard_breeding_tk.value,
                params.blk_ib_manifold.value,
            ),
            r_inner_cut,
            BlanketThickness(
                params.fw_blanket_surface_tk.value,
                params.outboard_fw_tk.value,
                params.outboard_breeding_tk.value,
                params.blk_ob_manifold.value,
            ),
            DivertorThickness(params.fw_divertor_surface_tk.value),
            ToroidalFieldCoilDimension(params.tk_tf_inboard.value, params.r_tf_in.value),
            RadiationShieldThickness(params.tk_rs.value),
        )
