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
Tests the divertor silhouette functionality for shaping the divertor system.
"""

import numpy as np
import pytest

import tests
from BLUEPRINT.base.error import SystemsError
from BLUEPRINT.geometry.loop import Loop
from BLUEPRINT.systems.divertor_silhouette import (
    DivertorSilhouette,
    DivertorSilhouetteFlatDome,
    DivertorSilhouetteFlatDomePsiBaffle,
    DivertorSilhouettePsiBaffle,
)


class TestDivertorSilhouette:
    """
    A class to test the DivertorSilhouette system.
    """

    locations = ["lower"]
    legs = ["inner", "outer"]
    required_geom_legs = ["targets", "baffles"]
    required_geom = ["domes"]
    required_geom_loop = [
        "inner",
        "outer",
        "first_wall",
        "gap",
        "divertors",
        "vessel_gap",
        "blanket_inner",
    ]
    required_div_geom = [
        "divertor_inner",
        "divertor_gap",
        "divertor",
    ]

    @pytest.mark.reactor
    @pytest.mark.parametrize(
        "div_class",
        [
            DivertorSilhouette,
            DivertorSilhouetteFlatDome,
            DivertorSilhouettePsiBaffle,
            DivertorSilhouetteFlatDomePsiBaffle,
        ],
    )
    def test_divertor_silhouette_build(self, reactor, div_class):
        reactor.RB.initialise_targets()
        to_dp = {
            "sf": reactor.RB.sf,
            "targets": reactor.RB.targets,
        }
        div = div_class(reactor.params, to_dp)
        div.build(reactor.RB.geom["inner_loop"])

        # Check that the geometry dictionary has been populated correctly
        for location in self.locations:
            for leg in self.legs:
                for key in self.required_geom_legs:
                    assert key in div.geom
                    assert isinstance(div.geom[key], dict)
                    assert location in div.geom[key]
                    assert isinstance(div.geom[key][location], dict)
                    assert leg in div.geom[key][location]
                    assert isinstance(div.geom[key][location][leg], np.ndarray)

            for key in self.required_geom:
                assert key in div.geom
                assert isinstance(div.geom[key], dict)
                assert location in div.geom[key]
                assert isinstance(div.geom[key][location], np.ndarray)

            for key in self.required_geom_loop:
                assert key in div.geom
                assert isinstance(div.geom[key], dict)
                assert location in div.geom[key]
                assert isinstance(div.geom[key][location], Loop)

    @pytest.mark.reactor
    @pytest.mark.parametrize(
        "div_class",
        [
            DivertorSilhouette,
            DivertorSilhouetteFlatDome,
            DivertorSilhouettePsiBaffle,
            DivertorSilhouetteFlatDomePsiBaffle,
        ],
    )
    def test_divertor_silhouette_make(self, reactor, div_class):
        reactor.RB.initialise_targets()
        to_dp = {
            "sf": reactor.RB.sf,
            "targets": reactor.RB.targets,
        }
        div = div_class(reactor.params, to_dp)
        div_geom = div.make_divertor(reactor.RB.geom["inner_loop"], "lower")

        for key in self.required_div_geom:
            assert key in div_geom
            assert isinstance(div_geom[key], Loop)

    @pytest.mark.reactor
    @pytest.mark.skipif(not tests.PLOTTING, reason="plotting disabled")
    @pytest.mark.parametrize(
        "div_class",
        [
            DivertorSilhouette,
            DivertorSilhouetteFlatDome,
            DivertorSilhouettePsiBaffle,
            DivertorSilhouetteFlatDomePsiBaffle,
        ],
    )
    def test_divertor_silhouette_debug(self, reactor, div_class):
        reactor.RB.initialise_targets()
        to_dp = {
            "sf": reactor.RB.sf,
            "first_wall": reactor.RB.geom["inner_loop"],
            "targets": reactor.RB.targets,
            "debug": True,
        }
        div = div_class(reactor.params, to_dp)
        div.build(reactor.RB.geom["inner_loop"])

    @pytest.mark.reactor
    @pytest.mark.parametrize(
        "div_class",
        [
            DivertorSilhouette,
            DivertorSilhouetteFlatDome,
            DivertorSilhouettePsiBaffle,
            DivertorSilhouetteFlatDomePsiBaffle,
        ],
    )
    def test_divertor_silhouette_bad_input(self, reactor, div_class):
        to_dp = {
            "bad_input": "a bad input",
        }
        with pytest.raises(SystemsError):
            div_class(reactor.params, to_dp)
