# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
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
Test PROCESS template builder
"""
import os

import pytest

from bluemira.codes.process._equation_variable_mapping import Constraint, Objective
from bluemira.codes.process._model_mapping import (
    PROCESSOptimisationAlgorithm,
)
from bluemira.codes.process.template_builder import PROCESSTemplateBuilder


def extract_warning(caplog):
    result = [line for message in caplog.messages for line in message.split(os.linesep)]
    result = " ".join(result)
    caplog.clear()
    return result


class TestPROCESSTemplateBuilder:
    def test_no_error_on_nothing(self, caplog):
        t = PROCESSTemplateBuilder()
        _ = t.make_inputs()
        assert len(caplog.messages) == 0

    def test_warn_on_optimisation_with_no_objective(self, caplog):
        t = PROCESSTemplateBuilder()
        t.set_optimisation_algorithm(PROCESSOptimisationAlgorithm.VMCON)
        _ = t.make_inputs()
        assert len(caplog.messages) == 1
        warning = extract_warning(caplog)
        assert "You are running in optimisation mode, but" in warning

    @pytest.mark.parametrize(
        "objective",
        [Objective.FUSION_GAIN_PULSE_LENGTH, Objective.MAJOR_RADIUS_PULSE_LENGTH],
    )
    def test_error_on_maxmin_objective(self, objective):
        t = PROCESSTemplateBuilder()
        with pytest.raises(
            ValueError, match="can only be used as a minimisation objective"
        ):
            t.set_maximisation_objective(objective)

    @pytest.mark.parametrize("bad_name", ["spoon", "aaaaaaaaaaaaa"])
    def test_error_on_bad_itv_name(self, bad_name):
        t = PROCESSTemplateBuilder()
        with pytest.raises(ValueError, match="There is no iteration variable:"):
            t.add_variable(bad_name, 3.14159)

    @pytest.mark.parametrize("bad_name", ["spoon", "aaaaaaaaaaaaa"])
    def test_error_on_adjusting_bad_variable(self, bad_name):
        t = PROCESSTemplateBuilder()
        with pytest.raises(ValueError, match="There is no iteration variable:"):
            t.adjust_variable(bad_name, 3.14159)

    def test_warn_on_repeated_constraint(self, caplog):
        t = PROCESSTemplateBuilder()
        t.add_constraint(Constraint.BETA_CONSISTENCY)
        t.add_constraint(Constraint.BETA_CONSISTENCY)
        assert len(caplog.messages) == 1
        warning = extract_warning(caplog)
        assert "is already in" in warning

    def test_warn_on_repeated_itv(self, caplog):
        t = PROCESSTemplateBuilder()
        t.add_variable("bore", 2.0)
        t.add_variable("bore", 3.0)
        assert len(caplog.messages) == 1
        warning = extract_warning(caplog)
        assert "Iteration variable 'bore' is already" in warning

    def test_warn_on_adjusting_nonexistent_variable(self, caplog):
        t = PROCESSTemplateBuilder()
        t.adjust_variable("bore", 2.0)
        assert len(caplog.messages) == 1
        warning = extract_warning(caplog)
        assert "Iteration variable 'bore' is not in" in warning
        assert "bore" in t.variables

    def test_warn_on_missing_input_constraint(self, caplog):
        t = PROCESSTemplateBuilder()
        t.add_constraint(Constraint.NWL_UPPER_LIMIT)
        _ = t.make_inputs()
        assert len(caplog.messages) == 1
        warning = extract_warning(caplog)
        assert "requires inputs 'walalw'" in warning

    def test_automatic_fvalue_itv(self):
        t = PROCESSTemplateBuilder()
        t.set_minimisation_objective(Objective.MAJOR_RADIUS)
        t.add_constraint(Constraint.NET_ELEC_LOWER_LIMIT)
        assert "fpnetel" in t.variables
