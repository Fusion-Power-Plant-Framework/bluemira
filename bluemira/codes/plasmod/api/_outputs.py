# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
Parameters classes/structures for plasmod outputs.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import TextIO

import numpy as np


@dataclass
class PlasmodOutputs:
    """
    Dataclass of plasmod output values.
    """

    # Scalars
    amin: float | None = None
    av_nd: float | None = None
    av_nhe: float | None = None
    av_ni: float | None = None
    av_nz: float | None = None
    av_Te: float | None = None
    av_Ten: float | None = None
    av_Ti: float | None = None
    betan: float | None = None
    betapol: float | None = None
    betator: float | None = None
    bpolavg: float | None = None
    car: float | None = None
    che: float | None = None
    che3: float | None = None
    cprotium: float | None = None
    cwol: float | None = None
    cxe: float | None = None
    d: float | None = None
    f_gwpedtop: float | None = None
    f_ni: float | None = None
    fbs: float | None = None
    fcd: float | None = None
    Hcorr: float | None = None
    Hfact: float | None = None
    i_flag: int | None = None
    jiter: int | None = None
    Ip: float | None = None
    k: float | None = None
    nped: float | None = None
    nsep: float | None = None
    Palpha: float | None = None
    Paux: float | None = None
    Pbrehms: float | None = None
    Peaux: float | None = None
    perim: float | None = None
    Pfus: float | None = None
    Pfusdd: float | None = None
    Pfusdt: float | None = None
    Piaux: float | None = None
    PLH: float | None = None
    Pline: float | None = None
    Pneut: float | None = None
    Pohm: float | None = None
    Prad: float | None = None
    Pradcore: float | None = None
    Pradedge: float | None = None
    Psep: float | None = None
    psep_r: float | None = None
    psepb_q95AR: float | None = None
    Psync: float | None = None
    q_sep: float | None = None
    q95: float | None = None
    qcd: float | None = None
    qdivt: float | None = None
    qfus: float | None = None
    qheat: float | None = None
    qstar: float | None = None
    rli: float | None = None
    rplas: float | None = None
    Sp: float | None = None
    tauee: float | None = None
    taueff: float | None = None
    tauei: float | None = None
    teped: float | None = None
    torsurf: float | None = None
    v_loop: float | None = None
    Vp: float | None = None
    Wth: float | None = None
    Zeff: float | None = None

    # Profiles
    dprof: np.ndarray | None = None
    ffprime: np.ndarray | None = None
    g2: np.ndarray | None = None
    g3: np.ndarray | None = None
    ipol: np.ndarray | None = None
    jbs: np.ndarray | None = None
    jcd: np.ndarray | None = None
    jpar: np.ndarray | None = None
    kprof: np.ndarray | None = None
    nalf: np.ndarray | None = None
    ndeut: np.ndarray | None = None
    ne: np.ndarray | None = None
    nfuel: np.ndarray | None = None
    nions: np.ndarray | None = None
    ntrit: np.ndarray | None = None
    phi: np.ndarray | None = None
    pprime: np.ndarray | None = None
    press: np.ndarray | None = None
    psi: np.ndarray | None = None
    qprof: np.ndarray | None = None
    shif: np.ndarray | None = None
    Te: np.ndarray | None = None
    Ti: np.ndarray | None = None
    volprof: np.ndarray | None = None
    vprime: np.ndarray | None = None
    x: np.ndarray | None = None

    @classmethod
    def from_files(cls, scalar_stream: TextIO, profile_stream: TextIO) -> PlasmodOutputs:
        """
        Initialise outputs from a scalar and a profiles file.
        """
        scalars = read_plasmod_output(scalar_stream)
        profiles = read_plasmod_output(profile_stream)
        return cls(**scalars, **profiles)


def read_plasmod_output(io_stream: TextIO) -> dict[str, np.ndarray | float]:
    """Read an output file, generated by plasmod, into a dictionary."""
    output: dict[str, np.ndarray | float] = {}
    for row in csv.reader(io_stream, delimiter="\t"):
        output_key, *output_value = row[0].split()
        if len(output_value) > 1:
            output[output_key] = np.array(output_value, dtype=float)
        elif len(output_value) == 0:
            output[output_key] = np.array([])
        else:
            output[output_key] = float(output_value[0])
    return output
