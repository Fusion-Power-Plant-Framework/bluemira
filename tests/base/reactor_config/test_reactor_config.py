from pathlib import Path

import pytest

from bluemira.base.error import ReactorConfigError
from bluemira.base.reactor_config import ReactorConfig

class


class TestReactorConfigClass:
    """
    Tests for the Reactor Config class functionality.
    """

    def test_file_loading_with_empty_config(self):
        config_path = Path(__file__).parent / "reactor_config.empty.json"
        reactor_config = ReactorConfig(config_path.as_posix())

        assert isinstance(reactor_config.global_params, dict)
        assert len(reactor_config.global_params) == 0

    def test_warning_global_local_sub_overwrites(self, caplog):
        config_path = Path(__file__).parent / "reactor_config.warnings.json"
        reactor_config = ReactorConfig(config_path.as_posix())

        dp = reactor_config.designer_params("comp A")
        bp = reactor_config.builder_params("comp A")

        assert len(caplog.records) == 5
        for record in caplog.records:
            assert record.levelname == "WARNING"

        assert dp["a"] == 10
        assert dp["b"] == 5

        assert bp["a"] == 10
        assert bp["b"] == 5
        assert bp["c"] == 1

    def test_warning_global_sub_overwrites(self, caplog):
        config_path = Path(__file__).parent / "reactor_config.warnings.json"
        reactor_config = ReactorConfig(config_path.as_posix())

        p = reactor_config.designer_params("comp B")

        assert len(caplog.records) == 1
        for record in caplog.records:
            assert record.levelname == "WARNING"
        assert p["a"] == 10

    def test_warning_global_local_overwrites(self, caplog):
        config_path = Path(__file__).parent / "reactor_config.warnings.json"
        reactor_config = ReactorConfig(config_path.as_posix())

        p = reactor_config.designer_params("comp C")

        assert len(caplog.records) == 1
        for record in caplog.records:
            assert record.levelname == "WARNING"
        assert p["a"] == 10

    def test_warning_local_sub_overwrites(self, caplog):
        config_path = Path(__file__).parent / "reactor_config.warnings.json"
        reactor_config = ReactorConfig(config_path.as_posix())

        p = reactor_config.designer_params("comp D")

        assert len(caplog.records) == 1
        for record in caplog.records:
            assert record.levelname == "WARNING"
        assert p["a"] == 10
        assert p["b"] == 5

    def test_no_warning_no_overwrites(self, caplog):
        config_path = Path(__file__).parent / "reactor_config.warnings.json"
        reactor_config = ReactorConfig(config_path.as_posix())

        p = reactor_config.designer_params("comp E")

        assert len(caplog.records) == 0
        assert p["a"] == 10
        assert p["b"] == 5
        assert p["c"] == 1

    def test_no_params_in_designer_error(self):
        reactor_config = ReactorConfig(
            {
                "comp A": {
                    "params": {"a": 5},
                    "designer": {},
                    "builder": {},
                },
            }
        )

        with pytest.raises(ReactorConfigError):
            reactor_config.designer_params("comp A")

    def test_no_params_in_builder_error(self):
        reactor_config = ReactorConfig(
            {
                "comp A": {
                    "params": {"a": 5},
                    "designer": {},
                    "builder": {},
                },
            }
        )

        with pytest.raises(ReactorConfigError):
            reactor_config.builder_params("comp A")

    def test_getting_config_with_no_params(self, caplog):
        reactor_config = ReactorConfig(
            {
                "comp A": {
                    "params": {"a": 5},
                    "designer": {"params": {}},
                    "builder": {"some_config": "a_value"},
                },
                "comp B": {
                    "params": {"b": 5},
                    "designer": {"some_config": "a_value"},
                    "builder": {"params": {}},
                },
            }
        )

        dp = reactor_config.designer_params("comp A")
        bc = reactor_config.builder_config("comp A")

        dc = reactor_config.designer_config("comp B")
        bp = reactor_config.builder_params("comp B")

        assert len(caplog.records) == 0

        assert dp["a"] == 5
        assert bc["some_config"] == "a_value"

        assert bp["b"] == 5
        assert dc["some_config"] == "a_value"
