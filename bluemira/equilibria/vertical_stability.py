# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Vertical stability calculations
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bluemira.base.look_and_feel import bluemira_debug, bluemira_warn
from bluemira.equilibria.coils._tools import make_mutual_inductance_matrix

if TYPE_CHECKING:
    import numpy.typing as npt

    from bluemira.equilibria.coils._grouping import CoilSet
    from bluemira.equilibria.equilibrium import Equilibrium


def calculate_rzip_stability_criterion(
    eq: Equilibrium, *, with_active: bool = False
) -> float:
    """
    Calculate the rzip stability criterion for a given equilibrium and coilset

    Parameters
    ----------
    eq:
        The equilibrium object
    with_active:
        Include the stabilising effect of acitve coils along with passive structures

    Returns
    -------
    :
        Stability criterion

    Notes
    -----
    See ~:class:`~bluemira.equilibria.vertical_stability.RZIp` for details
    """
    return RZIp(eq.coilset)(eq, with_active=with_active)


class RZIp:
    """
    RZIp model

    Parameters
    ----------
    coilset:
        The full coilset

    Notes
    -----
    A value ~ 1.5 considered optimal and controllable.
    See https://doi.org/10.13182/FST89-A39747 for further explanation

        < 1 plasma mass becomes a factor
        = 1 massless plasma solution not valid (said to represent MHD effects)
        > 1 displacement growth dominated by L/R of passive system

    .. math::

        f = -\\frac{F_s}{F_d}
          = \\frac{I_p^T M^{\\prime}_p|_s [M_s|_s]^{-1} M^{\\prime}_s|_p I_p}
                  {I_p^T M^{\\prime\\prime}_p|_c I_c}
    """

    def __init__(self, coilset):
        self.coilset = coilset

    @property
    def coilset(self):
        """Coilset used for calculation"""
        return self._coilset

    @coilset.setter
    def coilset(self, cs: CoilSet):
        """Calculate coilset inductance"""
        # TODO @je-cook: Possibly implement quad indexing?
        # 1231231231231
        self._coilset = cs
        self.ind_mat = make_mutual_inductance_matrix(
            cs, square_coil=True, with_quadratures=True
        )
        diag = np.diag(np.diag(self.ind_mat))
        non_diag = (
            2 * np.pi * (self.ind_mat - diag)
        )  # extra 2pi term due to per circle rather than per radian
        self.ind_mat = diag + non_diag

    def __call__(
        self,
        eq: Equilibrium,
        *,
        with_active: bool = False,
    ) -> float:
        """
        Parameters
        ----------
        eq:
            The equilibrium object to analyse

        Returns
        -------
        :
            The stability criterion
        """
        if not hasattr(eq, "profiles"):
            bluemira_warn("No profiles found on equilibrium")
            # this constraint is pretty irrelevant for breakdown
            return 0

        return stab_destab(
            cc_current=self.coilset.get_control_coils().current,
            ind_mat=self.ind_mat,
            control_ind=self.coilset._control_ind,
            uncontrolled_ind=list(
                set(self.coilset._get_type_index()) - set(self.coilset._control_ind)
            ),
            r_struct=np.tile(eq.x.reshape(-1), (len(self.coilset._get_type_index()), 1)),
            i_plasma=eq._jtor * eq.grid.step,
            br_struct_grid=np.rollaxis(eq._bx_green, 2, 0),
            dbrdz_struct_grid=np.rollaxis(eq._db_green, 2, 0),
            with_active=with_active,
        )


def stab_destab(
    cc_current: npt.NDArray,
    ind_mat: npt.NDArray,
    control_ind: list[int],
    uncontrolled_ind: list[int],
    r_struct: npt.NDArray,
    i_plasma: npt.NDArray,
    br_struct_grid: npt.NDArray,
    dbrdz_struct_grid: npt.NDArray,
    *,
    with_active: bool = False,
) -> float:
    """
    Calculate the stabilising / destabilising effect of the equilibria and structures

    Parameters
    ----------
    cc_current:
        array of control coil currents
    ind_mat:
        Inductance matrix of passive and active structures
    control_ind:
        indicies of active structures
    uncontrolled_ind:
        indicies of passive structures
    r_struct:
        flattened array of eq R points duplicated by number of coils
    i_plasma:
        the plasma jtor x grid step
    br_struct_grid:
        Bx field
    dbrdz_struct_grid:
        dBxdz field
    with_active:
        Include the stabilising effect of acitve coils along with passive structures

    Returns
    -------
    :
        The stability criterion
    """
    r_active = r_struct[control_ind, :]
    if with_active:
        l_ps_ps = ind_mat[uncontrolled_ind][:, uncontrolled_ind]
        l_ac_ps = ind_mat[control_ind][:, uncontrolled_ind]
        l_ac_ac = ind_mat[control_ind][:, control_ind]

        mss = np.vstack([
            np.hstack([l_ac_ac, l_ac_ps]),
            np.hstack([l_ac_ps.T, l_ps_ps]),
        ])
        br_size = len(uncontrolled_ind) + len(control_ind)
    else:
        mss = ind_mat[uncontrolled_ind][:, uncontrolled_ind]
        r_struct = r_struct[uncontrolled_ind, :]
        br_struct_grid = br_struct_grid[uncontrolled_ind, ...]
        br_size = len(uncontrolled_ind)
    shape = np.shape(br_struct_grid)
    br = br_struct_grid.reshape((br_size, shape[1] * shape[2]))
    dbrdz = dbrdz_struct_grid[control_ind, ...]
    dbrdz = dbrdz.reshape((len(control_ind), shape[1] * shape[2]))
    msp_prime = (2 * np.pi * r_struct * br).T
    i_plasma = i_plasma.reshape(-1)
    grid_coil = (2 * np.pi * r_active * dbrdz).T
    destabilising = np.einsum(
        "b, bd, d",
        i_plasma,
        grid_coil,
        cc_current,
        optimize=["einsum_path", (0, 1), (0, 1)],
    )

    stabilising = np.einsum(
        "d, db, bc, ac, a",
        i_plasma,
        msp_prime,
        np.linalg.inv(mss),
        msp_prime,
        i_plasma,
        optimize=["einsum_path", (0, 1), (1, 2), (0, 1), (0, 1)],
    )
    # stabilising force/ destabilising force differentiated wrt to z coord
    # f = -d_fs / d_fd
    # not infinite if destabilising is 0 because therefore it is stable
    criterion = (-stabilising / destabilising) if destabilising != 0 else 0
    bluemira_debug(f"{stabilising=}, {destabilising=}, {criterion=}")
    return criterion
