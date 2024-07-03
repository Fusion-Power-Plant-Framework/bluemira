# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import json
from pathlib import Path

import pytest

from bluemira.base.file import get_bluemira_path
from bluemira.equilibria import Equilibrium
from eudemo.equilibria import EquilibriumDesigner
from eudemo.equilibria._designer import EquilibriumDesignerParams


class TestEquilibriumDesigner:
    EQDSK_FILE = Path(
        get_bluemira_path("equilibria", subfolder="data"), "EU-DEMO_EOF.json"
    )
    DATA_DIR = Path(Path(__file__).parent, "test_data")

    @classmethod
    def setup_class(cls):
        cls.param_dict = cls._read_json(Path(cls.DATA_DIR, "params.json"))

    def test_params_converted_to_parameter_frame(self):
        designer = EquilibriumDesigner(self.param_dict)

        assert isinstance(designer.params, EquilibriumDesignerParams)

    @pytest.mark.longrun
    def test_designer_converges_on_run(self):
        designer = EquilibriumDesigner(self.param_dict, {"plot_optimisation": True})

        eq = designer.execute()

        assert eq.get_LCFS()
        # check parameters have been updated
        assert designer.params != EquilibriumDesignerParams.from_dict(self.param_dict)

    def test_designer_reads_file_in_read_mode(self):
        eqdsk = self.EQDSK_FILE
        designer = EquilibriumDesigner(
            self.param_dict, {"run_mode": "read", "file_path": eqdsk}
        )

        eq = designer.execute()

        ref_eq = Equilibrium.from_eqdsk(eqdsk, from_cocos=17)
        assert eq.analyse_plasma() == ref_eq.analyse_plasma()

    def test_ValueError_on_init_given_read_mode_and_no_file_path(self):
        with pytest.raises(ValueError):  # noqa: PT011
            EquilibriumDesigner(self.param_dict, {"run_mode": "read"})

    @staticmethod
    def _read_json(file_path: str) -> dict:
        with open(file_path) as f:
            return json.load(f)
