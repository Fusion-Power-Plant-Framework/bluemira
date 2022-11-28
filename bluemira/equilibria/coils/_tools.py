import numpy as np

from bluemira.magnetostatics.greens import circular_coil_inductance_elliptic, greens_psi


def make_mutual_inductance_matrix(coilset):
    """
    Calculate the mutual inductance matrix of a coilset.

    Parameters
    ----------
    coilset: CoilSet
        Coilset for which to calculate the mutual inductance matrix

    Returns
    -------
    M: np.ndarray
        The symmetric mutual inductance matrix [H]

    Notes
    -----
    Single-filament coil formulation; serves as a useful approximation.
    """
    n_coils = coilset.n_coils()
    M = np.zeros((n_coils, n_coils))  # noqa
    xcoord = coilset.x
    zcoord = coilset.z
    dx = coilset.dx
    dz = coilset.dz
    n_turns = coilset.n_turns

    itri, jtri = np.triu_indices(n_coils, k=1)

    M[itri, jtri] = (
        n_turns[itri]
        * n_turns[jtri]
        * greens_psi(xcoord[itri], zcoord[itri], xcoord[jtri], zcoord[jtri])
    )
    M[jtri, itri] = M[itri, jtri]

    radius = np.hypot(dx, dz)
    for i in range(n_coils):
        M[i, i] = n_turns[i] ** 2 * circular_coil_inductance_elliptic(
            xcoord[i], radius[i]
        )

    return M


def symmetrise_coilset():
    """
    Dummy
    """
    pass


def check_coilset_symmetric():
    """
    Dummy
    """
    pass


def get_max_current(dx, dz, j_max):
    """
    Get the maximum current in a coil cross-sectional area

    Parameters
    ----------
    dx: float
        Coil half-width [m]
    dz: float
        Coil half-height [m]
    j_max: float
        Coil current density [A/m^2]

    Returns
    -------
    max_current: float
        Maximum current [A]
    """
    return abs(j_max * (4 * dx * dz))
