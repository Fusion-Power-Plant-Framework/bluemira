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
    n_coils = coilset.n_coils
    M = np.zeros((n_coils, n_coils))  # noqa
    coils = list(coilset.coils.values())

    itri, jtri = np.triu_indices(n_coils, k=1)

    for i, j in zip(itri, jtri):
        coil_1 = coils[i]
        coil_2 = coils[j]
        n1 = coil_1.n_turns
        n2 = coil_2.n_turns
        mi = n1 * n2 * greens_psi(coil_1.x, coil_1.z, coil_2.x, coil_2.z)
        M[i, j] = M[j, i] = mi

    for i, coil in enumerate(coils):
        radius = np.hypot(coil.dx, coil.dz)
        M[i, i] = coil.n_turns**2 * circular_coil_inductance_elliptic(coil.x, radius)

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
