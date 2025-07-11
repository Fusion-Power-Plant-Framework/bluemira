# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Simple 0-D neutronics model."""

from dataclasses import dataclass

import numpy as np

from bluemira.base.constants import EPS
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.base.parameter_frame._frame import ParameterFrame
from bluemira.base.parameter_frame._parameter import Parameter
from bluemira.radiation_transport.error import NeutronicsError


@dataclass
class ZeroDNeutronicsModelParams(ParameterFrame):
    """
    ZeroDNeutronicsModel input parameters
    """

    P_fus_DT: Parameter[float]
    e_mult: Parameter[float]
    e_decay_mult: Parameter[float]
    TBR: Parameter[float]
    f_n_blanket: Parameter[float]
    f_n_divertor: Parameter[float]
    f_n_vessel: Parameter[float]
    f_n_aux: Parameter[float]
    peak_NWL: Parameter[float]  # noqa: N815
    peak_bb_iron_dpa_rate: Parameter[float]
    peak_vv_iron_dpa_rate: Parameter[float]
    peak_div_cu_dpa_rate: Parameter[float]


@dataclass
class ZeroDNeutronicsResult(ParameterFrame):
    """
    ZeroDNeutronicsModel output parameters
    """

    e_mult: Parameter[float]
    TBR: Parameter[float]
    P_n_blanket: Parameter[float]
    P_n_divertor: Parameter[float]
    P_n_vessel: Parameter[float]
    P_n_aux: Parameter[float]
    P_n_e_mult: Parameter[float]
    P_n_decay: Parameter[float]
    peak_NWL: Parameter[float]  # noqa: N815
    peak_bb_iron_dpa_rate: Parameter[float]
    peak_vv_iron_dpa_rate: Parameter[float]
    peak_div_cu_dpa_rate: Parameter[float]


class ZeroDNeutronicsModel:
    """
    Simplified fraction distribution of neutron power and energy multiplication
    among components

    Parameters
    ----------
    params:
        ZeroDNeutornicsModel input parameter frame
    """

    _source = "0-D neutronics model"

    def __init__(self, params: ZeroDNeutronicsModelParams):
        self.params = params
        self._check_fractions()

        if params.e_mult.value < 1.0:
            raise NeutronicsError("Energy multiplication factor cannot be less than 1.0")
        if params.e_decay_mult.value < 1.0:
            raise NeutronicsError("Decay multiplication factor cannot be less than 1.0")

    def _check_fractions(self):
        frac_sum = (
            self.params.f_n_blanket.value
            + self.params.f_n_divertor.value
            + self.params.f_n_vessel.value
            + self.params.f_n_aux.value
        )

        if frac_sum > 1 + EPS:
            raise NeutronicsError(
                "Cannot have neutron power fractions summing greater than 1.0."
            )

        if not np.isclose(frac_sum, 1.0, atol=EPS, rtol=0.0):
            diff = 1.0 - frac_sum
            bluemira_warn(
                "Neutron fractions sum to less than 1.0, putting the change in auxiliary"
                " neutron power."
            )
            self.params.update_from_dict({"f_n_aux": diff}, source=self._source)

    def run(self) -> ZeroDNeutronicsResult:
        """
        Run the 0-D neutronics model

        Returns
        -------
        ZeroDNeutronicsModel results
        """
        # Energy multiplication power is assigned to blanket
        neutron_power = 0.8 * self.params.P_fus_DT.value
        blk_power = (
            self.params.f_n_blanket.value * self.params.e_mult.value * neutron_power
        )
        div_power = self.params.f_n_divertor.value * neutron_power
        vv_power = self.params.f_n_vessel.value * neutron_power
        aux_power = self.params.f_n_aux.value * neutron_power
        mult_power = (self.params.e_mult.value - 1.0) * blk_power
        decay_power = (self.params.e_decay_mult.value - 1.0) * neutron_power
        power_unit = self.params.P_fus_DT.unit
        return ZeroDNeutronicsResult(
            e_mult=self.params.e_mult,
            TBR=self.params.TBR,
            P_n_blanket=Parameter(
                "P_n_blanket", blk_power, unit=power_unit, source=self._source
            ),
            P_n_divertor=Parameter(
                "P_n_divertor", div_power, unit=power_unit, source=self._source
            ),
            P_n_vessel=Parameter(
                "P_n_vessel", vv_power, unit=power_unit, source=self._source
            ),
            P_n_aux=Parameter(
                "P_n_aux", aux_power, unit=power_unit, source=self._source
            ),
            P_n_decay=Parameter(
                "P_n_decay", decay_power, unit=power_unit, source=self._source
            ),
            P_n_e_mult=Parameter(
                "P_n_e_mult", mult_power, unit=power_unit, source=self._source
            ),
            peak_bb_iron_dpa_rate=self.params.peak_bb_iron_dpa_rate,
            peak_div_cu_dpa_rate=self.params.peak_div_cu_dpa_rate,
            peak_NWL=self.params.peak_NWL,
            peak_vv_iron_dpa_rate=self.params.peak_vv_iron_dpa_rate,
        )
