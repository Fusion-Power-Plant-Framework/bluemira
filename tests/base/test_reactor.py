# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import time
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest

from bluemira.base.error import ComponentError
from bluemira.base.parameter_frame._parameter import Parameter
from bluemira.base.reactor import ComponentManager, Reactor
from bluemira.builders.plasma import Plasma, PlasmaBuilder, PlasmaBuilderParams
from bluemira.geometry.base import BluemiraGeo
from bluemira.geometry.tools import make_polygon
from bluemira.materials.material import Void

REACTOR_NAME = "My Reactor"


class TFCoil(ComponentManager):
    """
    This component manager is purely for testing the reactor is still
    valid if this is not set, so we don't need to implement it.
    """


class MyReactor(Reactor):
    SOME_CONSTANT: str = "not a component"

    plasma: Plasma
    tf_coil: TFCoil


@pytest.mark.classplot
class TestReactor:
    @classmethod
    def setup_class(cls):
        cls.reactor = cls._make_reactor()

    def test_time_since_init(self):
        a = self.reactor.time_since_init()
        time.sleep(0.1)
        b = self.reactor.time_since_init()
        assert a > 0
        assert b >= a + 0.1

    def test_name_set_on_root_component(self):
        assert self.reactor.component().name == REACTOR_NAME

    def test_unset_components_are_not_part_of_component_tree(self):
        # Only the plasma component is in the tree, as we haven't set the
        # TF coil
        assert len(self.reactor.component().children) == 1

    def test_component_tree_built_from_class_properties(self):
        assert self.reactor.plasma.component().name == "Plasma"

    @pytest.mark.parametrize("dim", ["xz", "xy", "xyz"])
    def test_show_cad_displays_all_components(self, dim):
        with patch("bluemira.display.displayer.show_cad") as mock_show:
            self.reactor.show_cad(dim)

        call_arg = mock_show.call_args[0][0]
        assert isinstance(call_arg, BluemiraGeo)

    @pytest.mark.parametrize("bad_dim", ["not_a_dim", 1, ["x"]])
    def test_ComponentError_given_invalid_plotting_dimension(self, bad_dim):
        with pytest.raises(ComponentError):
            self.reactor.show_cad(bad_dim)

    @pytest.mark.parametrize("dim", ["xz", "xy"])
    def test_plot_displays_all_components(self, dim):
        with patch("bluemira.display.plotter.BasePlotter.show"):
            self.reactor.plot(dim)

    @pytest.mark.parametrize("bad_dim", ["i", 1, ["x"]])
    def test_ComponentError_given_invalid_plot_dimension_plot(self, bad_dim):
        with pytest.raises(ComponentError):
            self.reactor.plot(bad_dim)

    @pytest.mark.parametrize("material_filter", [True, False])
    @pytest.mark.parametrize("dim", ["xz", "xy", "xyz"])
    def test_reactor_doesnt_show_void_material_by_default(self, dim, material_filter):
        reactor = self._make_reactor()

        reactor.plasma.component().get_component(dim).get_component(
            "LCFS"
        ).material = Void("test")

        with patch("bluemira.display.displayer.show_cad") as mock_show:
            if material_filter:
                reactor.show_cad(dim)
            else:
                reactor.show_cad(dim, {"component_filter": None})

        call_arg = mock_show.call_args[0][0]
        assert (
            isinstance(call_arg, list)
            if material_filter
            else isinstance(call_arg, BluemiraGeo)
        )

    def test_save_cad(self, tmp_path):
        self.reactor.save_cad("xyz", directory=tmp_path)
        assert Path(tmp_path, f"{REACTOR_NAME}.stp").is_file()

    def test_show_cad_empty_reactor(self):
        reactor = Reactor(REACTOR_NAME, n_sectors=1)
        with pytest.raises(ComponentError):
            reactor.show_cad()

    def test_show_cad_illdefinded_reactor(self):
        class BadReactor(Reactor):
            A: int

        reactor = BadReactor(REACTOR_NAME, n_sectors=1)
        with pytest.raises(ComponentError):
            reactor.show_cad()

    @staticmethod
    def _make_reactor() -> MyReactor:
        reactor = MyReactor(REACTOR_NAME, n_sectors=1)
        # use a square plasma, as circle causes topological naming issue
        lcfs = make_polygon({"x": [1, 1, 5, 5], "z": [-2, 2, 2, -2]}, closed=True)
        reactor.plasma = Plasma(
            PlasmaBuilder(
                PlasmaBuilderParams(n_TF=Parameter(name="n_TF", value=1)),
                {},
                lcfs,
            ).build()
        )
        return reactor


class TestComponentMananger:
    def setup_method(self):
        lcfs = make_polygon({"x": [1, 1, 5, 5], "z": [-2, 2, 2, -2]}, closed=True)

        self.p_comp = PlasmaBuilder(
            PlasmaBuilderParams(n_TF=Parameter(name="n_TF", value=1)),
            {},
            lcfs,
        ).build()

        self.plasma = Plasma(self.p_comp)

    def test_tree_contains_components(self):
        plasmatree = self.plasma.tree()
        assert all(dim in plasmatree for dim in ("xz", "xy", "xyz"))

    def test_save_cad(self, tmp_path):
        self.plasma.save_cad("xyz", directory=tmp_path)
        assert Path(tmp_path, "Plasma.stp").is_file()

    @pytest.mark.parametrize("dim", ["xz", "xy", "xyz"])
    def test_show_cad_contains_components(self, dim):
        with patch("bluemira.display.displayer.show_cad") as mock_show:
            self.plasma.show_cad(dim)

        assert len(mock_show.call_args[0]) == 3

    @pytest.mark.parametrize("material_filter", [True, False])
    @pytest.mark.parametrize("dim", ["xz", "xy", "xyz"])
    def test_show_cad_ignores_void_by_default(self, dim, material_filter):
        p_comp = deepcopy(self.p_comp)

        p_comp.get_component(dim).get_component("LCFS").material = Void("test")

        plasma = Plasma(p_comp)

        with patch("bluemira.display.displayer.show_cad") as mock_show:
            if material_filter:
                plasma.show_cad(dim)
            else:
                plasma.show_cad(dim, {"component_filter": None})

        call_arg = mock_show.call_args[0][0]
        assert (
            isinstance(call_arg, list)
            if material_filter
            else isinstance(call_arg, BluemiraGeo)
        )

    @pytest.mark.parametrize("dim", ["xz", "xy"])
    def test_plot_displays_all_components(self, dim):
        with patch("bluemira.display.plotter.BasePlotter.show"):
            self.plasma.plot(dim)
