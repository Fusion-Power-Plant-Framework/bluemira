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
Parameters classes/structures for plasmod outputs.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Dict, Optional, TextIO, Union

import numpy as np


@dataclass
class PlasmodOutputs:
    """
    Dataclass of plasmod output values.
    """

    # Scalars
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
    f_gwpedtop: Optional[float] = None
    f_ni: Optional[float] = None
    fbs: Optional[float] = None
    fcd: Optional[float] = None
    Hcorr: Optional[float] = None
    Hfact: Optional[float] = None
    i_flag: Optional[int] = None
    jiter: Optional[int] = None
    Ip: Optional[float] = None
    k: Optional[float] = None
    nped: Optional[float] = None
    nsep: Optional[float] = None
    Palpha: Optional[float] = None
    Paux: Optional[float] = None
    Pbrehms: Optional[float] = None
    Peaux: Optional[float] = None
    perim: Optional[float] = None
    Pfus: Optional[float] = None
    Pfusdd: Optional[float] = None
    Pfusdt: Optional[float] = None
    Piaux: Optional[float] = None
    PLH: Optional[float] = None
    Pline: Optional[float] = None
    Pneut: Optional[float] = None
    Pohm: Optional[float] = None
    Prad: Optional[float] = None
    Pradcore: Optional[float] = None
    Pradedge: Optional[float] = None
    Psep: Optional[float] = None
    psep_r: Optional[float] = None
    psepb_q95AR: Optional[float] = None
    Psync: Optional[float] = None
    q_sep: Optional[float] = None
    q95: Optional[float] = None
    qcd: Optional[float] = None
    qdivt: Optional[float] = None
    qfus: Optional[float] = None
    qheat: Optional[float] = None
    qstar: Optional[float] = None
    rli: Optional[float] = None
    rplas: Optional[float] = None
    Sp: Optional[float] = None
    tauee: Optional[float] = None
    taueff: Optional[float] = None
    tauei: Optional[float] = None
    teped: Optional[float] = None
    torsurf: Optional[float] = None
    v_loop: Optional[float] = None
    Vp: Optional[float] = None
    Wth: Optional[float] = None
    Zeff: Optional[float] = None

    # Profiles
    dprof: Optional[np.ndarray] = None
    ffprime: Optional[np.ndarray] = None
    g2: Optional[np.ndarray] = None
    g3: Optional[np.ndarray] = None
    ipol: Optional[np.ndarray] = None
    jbs: Optional[np.ndarray] = None
    jcd: Optional[np.ndarray] = None
    jpar: Optional[np.ndarray] = None
    kprof: Optional[np.ndarray] = None
    nalf: Optional[np.ndarray] = None
    ndeut: Optional[np.ndarray] = None
    ne: Optional[np.ndarray] = None
    nfuel: Optional[np.ndarray] = None
    nions: Optional[np.ndarray] = None
    ntrit: Optional[np.ndarray] = None
    phi: Optional[np.ndarray] = None
    pprime: Optional[np.ndarray] = None
    press: Optional[np.ndarray] = None
    psi: Optional[np.ndarray] = None
    qprof: Optional[np.ndarray] = None
    shif: Optional[np.ndarray] = None
    Te: Optional[np.ndarray] = None
    Ti: Optional[np.ndarray] = None
    volprof: Optional[np.ndarray] = None
    vprime: Optional[np.ndarray] = None
    x: Optional[np.ndarray] = None

    @classmethod
    def from_files(cls, scalar_stream: TextIO, profile_stream: TextIO) -> PlasmodOutputs:
        """
        Initialise outputs from a scalar and a profiles file.
        """
        return cls.from_dict(
            read_plasmod_output(scalar_stream), read_plasmod_output(profile_stream)
        )

    @classmethod
    def from_dict(cls, scalar_dict: Dict, profile_dict: Dict):
        """
        Initialise outputs from scalar and profile dictionaries
        """
        return cls(**scalar_dict, **profile_dict)


def read_plasmod_output(io_stream: TextIO) -> Dict[str, Union[np.ndarray, float]]:
    """Read an output file, generated by plasmod, into a dictionary."""
    output: Dict[str, Union[np.ndarray, float]] = {}
    for row in csv.reader(io_stream, delimiter="\t"):
        output_key, *output_value = row[0].split()
        if len(output_value) > 1:
            output[output_key] = np.array(output_value, dtype=float)
        elif len(output_value) == 0:
            output[output_key] = np.array([])
        else:
            output[output_key] = float(output_value[0])
    return output
