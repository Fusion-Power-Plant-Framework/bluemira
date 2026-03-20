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

from bluemira.base.constants import CoilType
from bluemira.base.look_and_feel import bluemira_debug, bluemira_warn
from bluemira.equilibria.coils._coil import Coil
from bluemira.equilibria.coils._grouping import CoilGroup
from bluemira.equilibria.coils._tools import make_mutual_inductance_matrix

if TYPE_CHECKING:
    import numpy.typing as npt

    from bluemira.equilibria.coils._grouping import CoilSet
    from bluemira.equilibria.equilibrium import Equilibrium
    from bluemira.geometry.wire import BluemiraWire


def calculate_rzip_stability_criterion(
    eq: Equilibrium,
) -> float:
    """
    Calculate the rzip stability criterion for a given equilibrium and coilset

    Parameters
    ----------
    eq:
        The equilibrium object

    Returns
    -------
    :
        Stability criterion

    Notes
    -----
    See ~:class:`~bluemira.equilibria.vertical_stability.RZIp` for details
    """
    return RZIp(eq.coilset)(eq)


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
        try:
            self.ind_mat = make_mutual_inductance_matrix(
                cs, square_coil=True, with_quadratures=False
            )
        except IndexError:
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

    Returns
    -------
    :
        The stability criterion
    """
    r_active = r_struct[control_ind, :]
    mss = ind_mat[uncontrolled_ind][:, uncontrolled_ind]
    r_struct = r_struct[uncontrolled_ind, :]
    br_struct_grid = br_struct_grid[uncontrolled_ind, ...]
    shape = np.shape(br_struct_grid)
    br = br_struct_grid.reshape((len(uncontrolled_ind), shape[1] * shape[2]))
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


def _length_step(p1, p2, delta) -> float:
    """
    Calculates the tangent angle for two points and uses this to determine
    the wire length to use as the difference for a square with thickness
    delta.

    Returns
    -------
    float:
        Length value for a wire discretisation step.
    """
    theta = np.arctan2(p2[0] - p1[0], p2[2] - p1[2])
    return 0.5 * delta * ((np.sqrt(2) + 1) - (np.sqrt(2) - 1) * np.cos(4 * theta))


def _get_coil_points_along_wire(wire: BluemiraWire, thickness: float) -> np.ndarray:
    """
    Discretises input wire in such a way that squares centred on
    those points will not overlap whilst minimising gaps.

    Achieves by calculating tangent angle at given point and using
    this to determine how far along the wire to put the next point.

    Paramters
    ---------
    wire:
        The wire that the coilset will be centred on.
    thickness:
        The thickness of the coils, will also impact the number of coils.

    Returns
    -------
    np.ndarray:
        An array containing the discretised points of the input wire in 3D.
    """
    ip = wire.start_point().T[0]
    n_max = wire.length / thickness
    p = ip
    dl = thickness
    current_length = 0
    points = [ip]
    for _ in range(int(n_max)):
        p2 = wire.value_at(distance=current_length + dl).T
        g_val = _length_step(p, p2, thickness)
        point = wire.value_at(distance=current_length + g_val).T
        p = point
        if current_length + g_val < wire.length - thickness:
            points = np.append(points, [p], axis=0)
            current_length += g_val
        else:
            continue
    return points


def make_coils_along_wire(
    wire: BluemiraWire,
    thickness: float,
    simple: bool = True,  # noqa: FBT001, FBT002
    name_prefix: str = "Passive",
    ctype: CoilType = CoilType.DUM,
    resistivity: float = 0.0,
) -> CoilGroup:
    """
    Function to create a coilset from a wire, where the coils making up the coilset
    will have a dx and dz equal to half the given thickness, with coils separated by
    the full thickness value. Additionally the coils will be make of the provided
    material.

    The created coils will follow the wire by treating it as a centreline with coils
    centred on the line.

    Parameters
    ----------
    wire:
        The wire that the coilset will be centred on.
    thickness:
        The thickness of the coils, will also impact the number of coils.
    simple:
        Method of discretising the input wire.
    ctype:
        Coil type
    resistivity:
        Resistivity of the coil material [Ohm . m]

    Returns
    -------
    CoilGroup:
        A group of coils following the input wire.
    """
    if simple:
        coil_points = wire.discretise(dl=np.sqrt(2) * thickness).T
    else:
        coil_points = _get_coil_points_along_wire(wire, thickness)
    coil_area = thickness**2
    resistance_factor = resistivity * 2 * np.pi / coil_area
    coils = [
        Coil(
            x=point[0],
            z=point[2],
            dx=0.5 * thickness,
            dz=0.5 * thickness,
            name=f"{name_prefix}_{i}",
            ctype=ctype,
            current=0.0,
            n_turns=1,
            discretisation=np.nan,
            resistance=resistance_factor * point[0],
        )
        for i, point in enumerate(coil_points)
    ]
    return CoilGroup(*coils)
