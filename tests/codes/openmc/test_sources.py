from unittest.mock import patch

import pytest

from bluemira.codes.openmc.params import (
    OpenMCNeutronicsSolverParams,
    PlasmaSourceParameters,
)
from bluemira.codes.openmc.sources import make_pps_source
from bluemira.radiation_transport.error import SourceError


class TestSource:
    def setup_method(self):
        self.psp = PlasmaSourceParameters.from_parameterframe(
            OpenMCNeutronicsSolverParams.from_dict({
                "R_0": {"value": 9.0, "unit": "m"},
                "A": {"value": 2, "unit": ""},
                "kappa": {"value": 1.5, "unit": ""},
                "delta": {"value": 0.9, "unit": ""},
                "reactor_power": {"value": 2, "unit": "MW"},
                "peaking_factor": {"value": 1, "unit": ""},
                "T_e": {"value": 1e6, "unit": "K"},
                "shaf_shift": {"value": 0.1, "unit": "m"},
                "vertical_shift": {"value": 0.1, "unit": "m"},
            })
        )

    def test_error_on_import_failure(self):
        with (
            patch("bluemira.codes.openmc.sources.PPS_ISO_INSTALLED", new=False),
            pytest.raises(SourceError, match="installation not found"),
        ):
            make_pps_source(self.psp)

    def test_pss_creation(self):
        pytest.importorskip("pps_isotropic")
        source = make_pps_source(self.psp)
        assert source.parameters.startswith("major_r=900.0")
