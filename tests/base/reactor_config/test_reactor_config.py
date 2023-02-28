from dataclasses import dataclass
from pathlib import Path

import pytest

from bluemira.base.constants import raw_uc
from bluemira.base.error import ReactorConfigError
from bluemira.base.parameter_frame import (
    EmptyFrame,
    Parameter,
    ParameterFrame,
    make_parameter_frame,
)
from bluemira.base.reactor_config import ReactorConfig


@dataclass
class TestGlobalParams(ParameterFrame):
    __test__ = False

    only_global: Parameter[int]
    height: Parameter[float]
    age: Parameter[int]
    extra_global: Parameter[int]


@dataclass
class TestCompADesignerParams(ParameterFrame):
    __test__ = False

    only_global: Parameter[int]
    height: Parameter[float]
    age: Parameter[int]
    name: Parameter[str]
    location: Parameter[str]


class TestReactorConfigClass:
    """
    Tests for the Reactor Config class functionality.
    """

    def test_file_loading_with_empty_config(self):
        config_path = Path(__file__).parent / "reactor_config.empty.json"
        reactor_config = ReactorConfig(config_path.as_posix(), EmptyFrame)

        # want to know explicitly if it is an EmptyFrame
        assert type(reactor_config.global_params) is EmptyFrame
        with pytest.raises(ReactorConfigError):
            reactor_config.params_for("dne")
            reactor_config.config_for("dne")

    def test_incorrect_global_config_type_empty_config(self):
        config_path = Path(__file__).parent / "reactor_config.empty.json"
        with pytest.raises(ValueError):
            ReactorConfig(config_path.as_posix(), TestGlobalParams)

    def test_incorrect_global_config_type_non_empty_config(self):
        config_path = Path(__file__).parent / "reactor_config.test.json"
        with pytest.raises(ValueError):
            ReactorConfig(config_path.as_posix(), EmptyFrame)

    def test_params_for_warnings_make_param_frame_type_value_overrides(
        self,
        caplog,
    ):
        config_path = Path(__file__).parent / "reactor_config.test.json"
        reactor_config = ReactorConfig(config_path.as_posix(), TestGlobalParams)

        cp = reactor_config.params_for("comp A", "designer")

        assert len(caplog.records) == 1
        for record in caplog.records:
            assert record.levelname == "WARNING"

        cpf = make_parameter_frame(cp, TestCompADesignerParams)

        # value checks
        assert cpf.only_global.value == raw_uc(1, "years", "s")
        assert cpf.height.value == 1.8
        assert cpf.age.value == raw_uc(30, "years", "s")
        assert cpf.name.value == "Comp A"
        assert cpf.location.value == "here"

        # instance checks
        assert cpf.only_global is reactor_config.global_params.only_global
        assert cpf.height is reactor_config.global_params.height
        assert cpf.age is reactor_config.global_params.age

    def test_config_for_warnings_value_overrides(
        self,
        caplog,
    ):
        config_path = Path(__file__).parent / "reactor_config.test.json"
        reactor_config = ReactorConfig(config_path.as_posix(), TestGlobalParams)

        cf_comp_a = reactor_config.config_for("comp A")
        cf_comp_a_des = reactor_config.config_for("comp A", "designer")

        assert len(caplog.records) == 1
        for record in caplog.records:
            assert record.levelname == "WARNING"

        assert cf_comp_a["config_a"] == cf_comp_a_des["config_a"]
        assert cf_comp_a["config_b"] == cf_comp_a_des["config_b"]
        assert cf_comp_a_des["config_c"]["c_value"] == "c_value"

    def test_no_arg_in_config_error(self):
        reactor_config = ReactorConfig(
            {
                "comp A": {
                    "designer": {},
                },
            },
            EmptyFrame,
        )

        with pytest.raises(ReactorConfigError):
            reactor_config.params_for("comp A", "dne")
        with pytest.raises(ReactorConfigError):
            reactor_config.config_for("comp A", "dne")

    def test_no_params_warning(self, caplog):
        reactor_config = ReactorConfig(
            {
                "comp A": {
                    "designer": {},
                },
            },
            EmptyFrame,
        )

        cp = reactor_config.params_for("comp A", "designer")

        assert len(caplog.records) == 2
        for record in caplog.records:
            assert record.levelname == "WARNING"

        assert len(cp.local_params) == 0

    def test_no_config_warning(self, caplog):
        reactor_config = ReactorConfig(
            {
                "comp A": {
                    "designer": {},
                },
            },
            EmptyFrame,
        )

        cf_comp_a = reactor_config.config_for("comp A")
        cf_comp_a_des = reactor_config.config_for("comp A", "designer")

        assert len(caplog.records) == 1
        for record in caplog.records:
            assert record.levelname == "WARNING"

        assert len(cf_comp_a) == 1
        assert len(cf_comp_a_des) == 0

    def test_invalid_rc_initialization(self, caplog):
        with pytest.raises(ReactorConfigError):
            ReactorConfig(
                ["wrong"],
                EmptyFrame,
            )

    def test_args_arent_str(self, caplog):
        reactor_config = ReactorConfig(
            {
                "comp A": {
                    "designer": {},
                },
            },
            EmptyFrame,
        )

        with pytest.raises(ReactorConfigError):
            reactor_config.config_for("comp A", 1)
