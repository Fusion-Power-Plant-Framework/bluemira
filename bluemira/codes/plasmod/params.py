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

from dataclasses import dataclass
from typing import TextIO

import fortranformat as ff

from bluemira.base.look_and_feel import bluemira_warn


@dataclass
class PlasmodInputs:
    """Plasmod parameters with defaults."""

    FORTRAN_INT_FORMAT = "a20,  i10"
    FORTRAN_FLOAT_FORMAT = "a20, e17.9"

    def check_model(self):
        """Check selected plasmod models are known"""
        # TODO(hsaunders1904)
        pass

    def write(self, io_stream: TextIO):
        """Write Plasmod input."""
        f_int = ff.FortranRecordWriter(self.FORTRAN_INT_FORMAT)
        f_float = ff.FortranRecordWriter(self.FORTRAN_FLOAT_FORMAT)

        for k, v in vars(self).items():
            if isinstance(v, int):
                line = f_int.write([k, v])
            elif isinstance(v, float):
                line = f_float.write([k, v])
            else:
                bluemira_warn(f"May produce fortran read errors, type: {type(v)}")
                line = f"{k} {v}"
            io_stream.write(line)
            io_stream.write("\n")

    A: float = 3.1
    Ainc: float = 1.1
    amin: float = 2.9039
    Bt: float = 5.855
    c_car: float = 10.0
    capA: float = 0.1
    car_qdivt: float = 1.0e-4
    car: float = 0.0
    che: float = 0.0
    che3: float = 0.0
    contrpovr: float = 0.0
    contrpovs: float = 0.0
    cprotium: float = 0.0
    cwol: float = 0.0
    cxe_psepfac: float = 1.0e-3
    cxe: float = 0.0
    d: float = 0.38491934960310104
    d95: float = 0.333
    dgy: float = 1e-5
    dt: float = 0.01
    dtinc: float = 2.0
    dtmax: float = 1e-2
    dtmaxmax: float = 1.0
    dtmaxmin: float = 0.1
    dtmin: float = 1e-2
    dtminmax: float = 5.0
    dx_cd_ech: float = 0.03
    dx_cd_nbi: float = 0.2
    dx_control_ech: float = 0.03
    dx_control_nbi: float = 0.2
    dx_fus_ech: float = 0.03
    dx_fus_nbi: float = 0.2
    dx_heat_ech: float = 0.03
    dx_heat_nbi: float = 0.2
    eccdeff: float = 0.3
    eopt: float = 0.1
    f_gw: float = 0.85
    f_gws: float = 0.5
    f_ni: float = 0.0
    fcdp: float = -1.0
    fcoreraditv: float = -1.0
    fpion: float = 0.5
    fuehe3: float = 0.0
    fuelmix: float = 0.5
    globtau_ar: float = 10.0
    globtau_d: float = 10.0
    globtau_he: float = 10.0
    globtau_t: float = 10.0
    globtau_xe: float = 10.0
    Hfact: float = 1.1
    i_diagz: int = 0
    i_equiltype: int = 2
    i_impmodel: int = 1
    i_modeltype: int = 1
    i_pedestal: int = 2
    Ip: float = 17.75
    isawt: int = 1
    isiccir: int = 0
    k: float = 1.6969830041844367
    k95: float = 1.652
    maxpauxor: float = 20.0
    nbcdeff: float = 0.3
    nbi_energy: float = 1000.0
    nchannels: int = 3
    ntglf: int = 11
    nx: int = 41
    nxt: int = 5
    pech: float = 0.0
    pedscal: float = 1.0
    pfus_req: float = 0.0
    pheat_max: float = 130.0
    plh: int = 6
    pnbi: float = 0.0
    pradfrac: float = 0.6
    pradpos: float = 0.7
    psep_r_sup: float = 230.0
    psepb_q95AR_sup: float = 9.2
    psepplh_inf: float = 0.1
    psepplh_sup: float = 1000.0
    q_cd: float = 0.0
    q_control: float = 130.0
    q_fus: float = 0.0
    q_heat: float = 0.0
    q95: float = 3.5
    qdivt_sup: float = 0.0
    qnbi_psepfac: float = 100.0
    R: float = 9.002
    rho_n: float = 0.94
    rho_T: float = 0.94
    teped: float = 5.5
    tesep: float = 0.1
    test: int = 10000
    tol: float = 1e-10
    tolmin: float = 10.1
    v_loop: float = -1.0e-6
    volume_in: int = -2500
    x_cd_ech: float = 0.0
    x_cd_nbi: float = 0.0
    x_control_ech: float = 0.0
    x_control_nbi: float = 0.0
    x_fus_ech: float = 0.0
    x_fus_nbi: float = 0.0
    x_heat_ech: float = 0.0
    x_heat_nbi: float = 0.0
    xtglf_1: float = 0.1
    xtglf_10: float = 0.75
    xtglf_11: float = 0.8
    xtglf_2: float = 0.15
    xtglf_3: float = 0.2
    xtglf_4: float = 0.25
    xtglf_5: float = 0.3
    xtglf_6: float = 0.4
    xtglf_7: float = 0.5
    xtglf_8: float = 0.6
    xtglf_9: float = 0.7
