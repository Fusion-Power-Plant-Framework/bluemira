import matplotlib.pyplot as plt
import numpy as np

# from bluemira.equilibria import Equilibrium
import examples.equilibria.double_null_ST as double_null_ST

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


def generate_material_current(conductor_id, conductors, B, T):
    """
    Generates a material current for a selected conductors

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
    material_current : np.array
        Values of the Magnetic Field in each coil location for the strand level
    """
    material_current = []

    for B in B:
        material_current += [conductors.gen_current[conductor_id](B, T)]
    return material_current


def generate_cable_current(conductor_id, conductors, B, T):
    """
    Generates a strand a.k.a cable current for a selected conductors

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
    # use np.array

    currents = []
    for B in B:
        currents += [
            conductors.gen_current[conductor_id](B, T)
            * conductors.current_ratio[conductor_id]
        ]
    cable_current = np.array(currents)  # maybe a better way to define this?
    return cable_current


def generate_winding_pack():
    return


def plot_2D_field_map(coords, B, coilset, tf):
    """
    plot 2D field map including coil outlines

    Parameters
    ----------
    coords: np.ndarray
        coordinates at which the field map is calculated
    fieldmap: np.ndarray
        field values at each point in coords
    coilset: bluemira.coils.Coilset
        coilset to plot
    TF: TF coil object
        TF coil object to plot
    """
    fig, ax = plt.subplots()

    coilset.plot(ax=ax, label=False, mask=False, alpha=0)
    tf.plot_xz(ax=ax, alpha=0)

    cf = ax.contourf(
        coords[0],
        coords[1],
        np.meshgrid(B, B)[0],
        levels=100,
        cmap=plt.get_cmap("jet"),
    )

    plt.colorbar(cf, label=r"$\|\mathbf{B}\|$ [T]")
    plt.show()
