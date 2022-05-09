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
Parameters classes/structures for Plasmod
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Dict, Optional, TextIO, Union

import numpy as np

from bluemira.base.look_and_feel import bluemira_debug
from bluemira.codes.error import CodesError


@dataclass
class PlasmodOutputs:
    """
    Plasmod output values with defaults.
    """

    amin: Optional[float] = None
    av_nd: Optional[float] = None
    av_nhe: Optional[float] = None
    av_ni: Optional[float] = None
    av_nz: Optional[float] = None
    av_Te: Optional[float] = None
    av_Ten: Optional[float] = None
    av_Ti: Optional[float] = None
    betan: Optional[float] = None
    betapol: Optional[float] = None
    betator: Optional[float] = None
    bpolavg: Optional[float] = None
    car: Optional[float] = None
    che: Optional[float] = None
    che3: Optional[float] = None
    cprotium: Optional[float] = None
    cwol: Optional[float] = None
    cxe: Optional[float] = None
    d: Optional[float] = None
    dprof: Optional[float] = None
    f_gwpedtop: Optional[float] = None
    f_ni: Optional[float] = None
    fbs: Optional[float] = None
    fcd: Optional[float] = None
    ffprime: Optional[float] = None
    g2: Optional[float] = None
    g3: Optional[float] = None
    Hcorr: Optional[float] = None
    Hfact: Optional[float] = None
    i_flag: Optional[int] = None
    Ip: Optional[float] = None
    ipol: Optional[float] = None
    jbs: Optional[float] = None
    jcd: Optional[float] = None
    jpar: Optional[float] = None
    k: Optional[float] = None
    kprof: Optional[float] = None
    nalf: Optional[float] = None
    ndeut: Optional[float] = None
    ne: Optional[float] = None
    nfuel: Optional[float] = None
    nions: Optional[float] = None
    nped: Optional[float] = None
    nsep: Optional[float] = None
    ntrit: Optional[float] = None
    Palpha: Optional[float] = None
    Paux: Optional[float] = None
    Pbrehms: Optional[float] = None
    Peaux: Optional[float] = None
    perim: Optional[float] = None
    Pfus: Optional[float] = None
    Pfusdd: Optional[float] = None
    Pfusdt: Optional[float] = None
    phi: Optional[float] = None
    Piaux: Optional[float] = None
    PLH: Optional[float] = None
    Pline: Optional[float] = None
    Pneut: Optional[float] = None
    Pohm: Optional[float] = None
    pprime: Optional[float] = None
    Prad: Optional[float] = None
    Pradcore: Optional[float] = None
    Pradedge: Optional[float] = None
    press: Optional[float] = None
    Psep: Optional[float] = None
    psep_r: Optional[float] = None
    psepb_q95AR: Optional[float] = None
    psi: Optional[float] = None
    Psync: Optional[float] = None
    q_sep: Optional[float] = None
    q95: Optional[float] = None
    qcd: Optional[float] = None
    qdivt: Optional[float] = None
    qfus: Optional[float] = None
    qheat: Optional[float] = None
    qprof: Optional[float] = None
    qstar: Optional[float] = None
    rli: Optional[float] = None
    rplas: Optional[float] = None
    shif: Optional[float] = None
    Sp: Optional[float] = None
    tauee: Optional[float] = None
    taueff: Optional[float] = None
    tauei: Optional[float] = None
    Te: Optional[float] = None
    teped: Optional[float] = None
    Ti: Optional[float] = None
    torsurf: Optional[float] = None
    v_loop: Optional[float] = None
    volprof: Optional[float] = None
    Vp: Optional[float] = None
    vprime: Optional[float] = None
    Wth: Optional[float] = None
    x: Optional[float] = None
    Zeff: Optional[float] = None

    @classmethod
    def from_files(cls, scalar_stream: TextIO, profile_stream: TextIO) -> PlasmodOutputs:
        """
        Initialize outputs from a scalar and a profile file.
        """
        scalars = _read_plasmod_csv(scalar_stream)
        status_flag = scalars["i_flag"]
        _check_plasmod_return_value(int(status_flag))
        profiles = _read_plasmod_csv(profile_stream)
        return cls(**scalars, **profiles)


def _read_plasmod_csv(io_stream: TextIO) -> Dict[str, Union[np.ndarray, float]]:
    """Read a CSV file, generated by plasmod, into a dictionary."""
    output: Dict[str, Union[np.ndarray, float]] = {}
    for row in csv.reader(io_stream, delimiter="\t"):
        output_key, *output_value = row[0].split()
        if len(output_value) > 1:
            output[output_key] = np.array(output_value, dtype=float)
        else:
            output[output_key] = float(output_value[0])
    return output


def _check_plasmod_return_value(exit_flag: int):
    """
    Check the return value of plasmod

     1: PLASMOD converged successfully
    -1: Max number of iterations achieved
        (equilibrium oscillating, pressure too high, reduce H)
        0: transport solver crashed (abnormal parameters
        or too large dtmin and/or dtmin
    -2: Equilibrium solver crashed: too high pressure

    Raises
    ------
    CodesError
        If the exit flag is an error code, or its value is not a known
        code.
    """
    if exit_flag == 1:
        bluemira_debug("plasmod converged successfully")
    elif exit_flag == -2:
        raise CodesError("plasmod error: Equilibrium solver crashed: too high pressure")
    elif exit_flag == -1:
        raise CodesError(
            "plasmod error: "
            "Max number of iterations reached "
            "equilibrium oscillating probably as a result of the pressure being too high "
            "reducing H may help"
        )
    elif not exit_flag:
        raise CodesError(
            "plasmod error: " "Abnormal parameters, possibly dtmax/dtmin too large"
        )
    else:
        raise CodesError(f"plasmod error: Unknown error code {exit_flag}")
