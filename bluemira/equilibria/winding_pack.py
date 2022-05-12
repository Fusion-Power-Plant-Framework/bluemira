import matplotlib.pyplot as plt
import numpy as np

# from bluemira.equilibria import Equilibrium
# import examples.equilibria.double_null_ST as double_null_ST

# Critical current is set against a paper specifying an electric current value at 1e5!

# IMPORT TF OBJECT
#
#    Should import the TF current source to construct the TF object
#    Currently unavalable so waiting for a fix on this
#
#    True B field is stored in `current_source` stored in def _field_map parameter b_tf
#    Could solve this in the future by either having a imported minimal TF object somewhere or creating a dummy class to import the TF object

# renames the TF current source object into <TF_source> to keep consistent with the main code
#        #TF object is called in reactor"
#        #TF_source=TF_current_source(TF)


# Return magnetic field [B] maps

#     This in theory should get positions of coil centrepoints
#     from coilset and input them through the magnetic field calculations in equilibria
#     in theory should return an array of Bx and Bz with field for coordinates

# %%


def select_temperature(conductor_id, temperature_id):
    """
    Selects temperature appropriate for conductor based on name

    Parameters
    ----------
    conductor_id : list
        conductor list from APECS
    temperature_id : dictionary
        Dictionary containing HTS and LTS temperatures

    Returns
    -------
    T: Float
        HTS or LTS temperature
    """
    if conductor_id == "ACT CORC-CICC REBCO":
        T = temperature_id["T_hts"]
    else:
        T = temperature_id["T_lts"]
    return T


def generate_cable_current(conductor_id, conductors, B, T):
    """
    Generates a cable current for a selected conductors

    Parameters
    ----------
    conductor_id : string
        name of the conductor material
    conductors:
        conductors from Apecs
    B : float
        the peak magnetic field at the coil (T)
    T : float
        HTS or LTS temperature

    Returns
    -------
    cable_current:
        Values of the Magnetic Field in each coil location for the strand level
    """
    currents = []
    for B in B:
        currents += [
            conductors.gen_current[conductor_id](B, T)
            * conductors.current_ratio[conductor_id]
        ]
    cable_current = np.array(currents)  # maybe a better way to define this?
    return cable_current


def contour_plot(eq, hmc, tf):
    """
    Attempts to plot B field in PF coils

    Parameters
    ----------
    hmc: HelmHoltz cage object

    eq:
        Equilibrium object
    tf:
        TF object

    Returns
    -------
    Plot of magnetic field in coils
    """
    keys = eq.coilset.coils.keys()  # produces all the coil keys
    x_corners = np.zeros((7, 4))
    z_corners = np.zeros((7, 4))

    for no, key in enumerate(keys):
        x_corners[no, :] = eq.coilset.coils[key].x_corner
        z_corners[no, :] = eq.coilset.coils[key].z_corner

    x, z = eq.coilset.get_positions()  # arrays

    x_matrix = np.zeros((7, 5))
    x_matrix[:, [0, 1, 3, 4]] = x_corners
    x_matrix[:, 2] = x
    z_matrix = np.zeros((7, 5))
    z_matrix[:, [0, 1, 3, 4]] = z_corners
    z_matrix[:, 2] = z
    y_matrix = np.zeros_like(x_matrix)

    B_vect = np.zeros((5, 3, 5))
    B_pf_plasmat = np.zeros((5, 2, 5))
    B = np.zeros((5, 5))

    fig, ax = plt.subplots()
    eq.coilset.plot(ax=ax, label=False, mask=False, alpha=0)
    tf.plot_xz(ax=ax, alpha=0)

    for coil, row in enumerate(x_matrix):
        X, Z = np.meshgrid(x_matrix[coil], z_matrix[coil])
        XZ = np.concatenate([X[None], Z[None]], axis=0)

        for no, row in enumerate(B):
            B_vect[no, :, :] = hmc.field(X[no], y_matrix[no], Z[no])
            B_pf_plasmat[no, :, :] = np.stack(
                [eq.Bx(X[no], Z[no]), eq.Bz(X[no], Z[no])], axis=0
            )

        B_vect[
            :,
            (0, 2),
        ] += B_pf_plasmat  # stacked array of all three magnetic field contributions

        for no, row in enumerate(B):
            B[no] = np.sqrt(
                B_vect[no, 0] ** 2 + B_vect[no, 1] ** 2 + B_vect[no, 2] ** 2
            )  # stacked array of magnetic field in PF coil locations
        cf = ax.contour(
            XZ[0],
            XZ[1],
            B,
            # np.meshgrid(B, B)[0],
            levels=100,
            cmap=plt.get_cmap("jet"),
        )
    plt.colorbar(cf, label=r"$\|\mathbf{B}\|$ [T]")
    plt.show()


def calculate_B(hmc, eq):
    """
    Returns total magnetic field at PF coil center coordinates

    Parameters
    ----------
    hmc: HelmHoltz cage object

    eq:
        Equilibrium object
    Returns
    -------
    B : np.array
        Total magnetic field at PF coil center coordinates
    """
    x, z = eq.coilset.get_positions()  # arrays
    y = np.zeros_like(x)
    # coords = np.stack([x, z], axis=0)

    if hmc == None:
        B = np.sqrt((eq.Bx(x, z)) ** 2 + (eq.Bz(x, z)) ** 2 + (eq.Bt(x)) ** 2)
    else:
        B_vec = hmc.field(x, y, z)  # B toroidal field calculated using HMC cage
        B_pf_plasma = np.stack(
            [eq.Bx(x, z), eq.Bz(x, z)], axis=0
        )  # Bt and Br stacked into a single array
        B_vec[
            (0, 2),
        ] += B_pf_plasma  # stacked array of all three magnetic field contributions
        B = np.sqrt(
            B_vec[0] ** 2 + B_vec[1] ** 2 + B_vec[2] ** 2
        )  # stacked array of magnetic field in PF coil locations
    return B
