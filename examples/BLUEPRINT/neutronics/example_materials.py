###############################################################################
# This file writes all of the materials data (multi-group nuclear
# cross-sections) for the OECD's C5G7 deterministic neutron transport
# benchmark problem to an HDF5 file. The script uses the h5py Python package
# to interact with the HDF5 file format. This may be a good example for those
# wishing ot write their nuclear data to an HDF5 file to import using the
# OpenMOC 'materialize' Python module.
###############################################################################


"""
Example materials definitions as provided by OpenMOC
"""

import h5py
import numpy

from BLUEPRINT.base.file import get_bluemira_root


# Create the file to store C5G7 multi-groups cross-sections
f = h5py.File(f"{get_bluemira_root()}/examples/neutronics/example_materials.h5", "w")
f.attrs["# groups"] = 7

# Create a group to specify that MGXS are split by material (vs. cell)
material_group = f.create_group("material")


#####
# UO2
#####

# Create a subgroup for UO2 materials data
uo2 = material_group.create_group("UO2")

sigma_t = numpy.array(
    [
        1.779490e-01,
        3.298050e-01,
        4.803880e-01,
        5.543670e-01,
        3.118010e-01,
        3.951680e-01,
        5.644060e-01,
    ]
)
sigma_s = numpy.array(
    [
        1.275370e-01,
        4.237800e-02,
        9.437400e-06,
        5.516300e-09,
        0.0,
        0.0,
        0.0,
        0.0,
        3.244560e-01,
        1.631400e-03,
        3.142700e-09,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        4.509400e-01,
        2.679200e-03,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        4.525650e-01,
        5.566400e-03,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.252500e-04,
        2.714010e-01,
        1.025500e-02,
        1.002100e-08,
        0.0,
        0.0,
        0.0,
        0.0,
        1.296800e-03,
        2.658020e-01,
        1.680900e-02,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        8.545800e-03,
        2.730800e-01,
    ]
)

sigma_f = numpy.array(
    [
        7.212060e-03,
        8.193010e-04,
        6.453200e-03,
        1.856480e-02,
        1.780840e-02,
        8.303480e-02,
        2.160040e-01,
    ]
)
nu_sigma_f = numpy.array(
    [
        2.005998e-02,
        2.027303e-03,
        1.570599e-02,
        4.518301e-02,
        4.334208e-02,
        2.020901e-01,
        5.257105e-01,
    ]
)
chi = numpy.array([5.87910e-01, 4.11760e-01, 3.39060e-04, 1.17610e-07, 0.0, 0.0, 0.0])

# Create datasets for each cross-section type
uo2.create_dataset("total", data=sigma_t)
uo2.create_dataset("scatter matrix", data=sigma_s)
uo2.create_dataset("fission", data=sigma_f)
uo2.create_dataset("nu-fission", data=nu_sigma_f)
uo2.create_dataset("chi", data=chi)


############
# MOX (4.3%)
############

# Create a subgroup for MOX-4.3%  materials data
mox43 = material_group.create_group("MOX-4.3%")

sigma_t = numpy.array(
    [
        1.787310e-01,
        3.308490e-01,
        4.837720e-01,
        5.669220e-01,
        4.262270e-01,
        6.789970e-01,
        6.828520e-01,
    ]
)
sigma_s = numpy.array(
    [
        1.288760e-01,
        4.141300e-02,
        8.229000e-06,
        5.040500e-09,
        0.0,
        0.0,
        0.0,
        0.0,
        3.254520e-01,
        1.639500e-03,
        1.598200e-09,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        4.531880e-01,
        2.614200e-03,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        4.571730e-01,
        5.539400e-03,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.604600e-04,
        2.768140e-01,
        9.312700e-03,
        9.165600e-09,
        0.0,
        0.0,
        0.0,
        0.0,
        2.005100e-03,
        2.529620e-01,
        1.485000e-02,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        8.494800e-03,
        2.650070e-01,
    ]
)
sigma_f = numpy.array(
    [
        7.62704e-03,
        8.76898e-04,
        5.69835e-03,
        2.28872e-02,
        1.07635e-02,
        2.32757e-01,
        2.48968e-01,
    ]
)
nu_sigma_f = numpy.array(
    [
        2.175300e-02,
        2.535103e-03,
        1.626799e-02,
        6.547410e-02,
        3.072409e-02,
        6.666510e-01,
        7.139904e-01,
    ]
)
chi = numpy.array([5.87910e-01, 4.11760e-01, 3.39060e-04, 1.17610e-07, 0.0, 0.0, 0.0])

# Create datasets for each cross-section type
mox43.create_dataset("total", data=sigma_t)
mox43.create_dataset("scatter matrix", data=sigma_s)
mox43.create_dataset("fission", data=sigma_f)
mox43.create_dataset("nu-fission", data=nu_sigma_f)
mox43.create_dataset("chi", data=chi)


##########
# MOX (7%)
##########

# Create a subgroup for MOX-7% materials data
mox7 = material_group.create_group("MOX-7%")

sigma_t = numpy.array(
    [
        1.813230e-01,
        3.343680e-01,
        4.937850e-01,
        5.912160e-01,
        4.741980e-01,
        8.336010e-01,
        8.536030e-01,
    ]
)
sigma_s = numpy.array(
    [
        1.304570e-01,
        4.179200e-02,
        8.510500e-06,
        5.132900e-09,
        0.0,
        0.0,
        0.0,
        0.0,
        3.284280e-01,
        1.643600e-03,
        2.201700e-09,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        4.583710e-01,
        2.533100e-03,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        4.637090e-01,
        5.476600e-03,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.761900e-04,
        2.823130e-01,
        8.728900e-03,
        9.001600e-09,
        0.0,
        0.0,
        0.0,
        0.0,
        2.276000e-03,
        2.497510e-01,
        1.311400e-02,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        8.864500e-03,
        2.595290e-01,
    ]
)
sigma_f = numpy.array(
    [
        8.25446e-03,
        1.32565e-03,
        8.42156e-03,
        3.28730e-02,
        1.59636e-02,
        3.23794e-01,
        3.62803e-01,
    ]
)
nu_sigma_f = numpy.array(
    [
        2.381395e-02,
        3.858689e-03,
        2.413400e-02,
        9.436622e-02,
        4.576988e-02,
        9.281814e-01,
        1.043200e00,
    ]
)
chi = numpy.array([5.87910e-01, 4.11760e-01, 3.39060e-04, 1.17610e-07, 0.0, 0.0, 0.0])

# Create datasets for each cross-section type
mox7.create_dataset("total", data=sigma_t)
mox7.create_dataset("scatter matrix", data=sigma_s)
mox7.create_dataset("fission", data=sigma_f)
mox7.create_dataset("nu-fission", data=nu_sigma_f)
mox7.create_dataset("chi", data=chi)


############
# MOX (8.7%)
############

# Create a subgroup for MOX-8.7% materials data
mox87 = material_group.create_group("MOX-8.7%")

sigma_t = numpy.array(
    [
        1.830450e-01,
        3.367050e-01,
        5.005070e-01,
        6.061740e-01,
        5.027540e-01,
        9.210280e-01,
        9.552310e-01,
    ]
)
sigma_s = numpy.array(
    [
        1.315040e-01,
        4.204600e-02,
        8.697200e-06,
        5.193800e-09,
        0.0,
        0.0,
        0.0,
        0.0,
        3.304030e-01,
        1.646300e-03,
        2.600600e-09,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        4.617920e-01,
        2.474900e-03,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        4.680210e-01,
        5.433000e-03,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.859700e-04,
        2.857710e-01,
        8.397300e-03,
        8.928000e-09,
        0.0,
        0.0,
        0.0,
        0.0,
        2.391600e-03,
        2.476140e-01,
        1.232200e-02,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        8.968100e-03,
        2.560930e-01,
    ]
)
sigma_f = numpy.array(
    [
        8.67209e-03,
        1.62426e-03,
        1.02716e-02,
        3.90447e-02,
        1.92576e-02,
        3.74888e-01,
        4.30599e-01,
    ]
)
nu_sigma_f = numpy.array(
    [
        2.518600e-02,
        4.739509e-03,
        2.947805e-02,
        1.122500e-01,
        5.530301e-02,
        1.074999e00,
        1.239298e00,
    ]
)
chi = numpy.array([5.87910e-01, 4.11760e-01, 3.39060e-04, 1.17610e-07, 0.0, 0.0, 0.0])

# Create datasets for each cross-section type
mox87.create_dataset("total", data=sigma_t)
mox87.create_dataset("scatter matrix", data=sigma_s)
mox87.create_dataset("fission", data=sigma_f)
mox87.create_dataset("nu-fission", data=nu_sigma_f)
mox87.create_dataset("chi", data=chi)


#################
# Fission Chamber
#################

# Create a subgroup for fission chamber materials data
fiss_chamber = material_group.create_group("Fission Chamber")

sigma_t = numpy.array(
    [
        1.260320e-01,
        2.931600e-01,
        2.842500e-01,
        2.810200e-01,
        3.344600e-01,
        5.656400e-01,
        1.172140e00,
    ]
)
sigma_s = numpy.array(
    [
        6.616590e-02,
        5.907000e-02,
        2.833400e-04,
        1.462200e-06,
        2.064200e-08,
        0.0,
        0.0,
        0.0,
        2.403770e-01,
        5.243500e-02,
        2.499000e-04,
        1.923900e-05,
        2.987500e-06,
        4.214000e-07,
        0.0,
        0.0,
        1.834250e-01,
        9.228800e-02,
        6.936500e-03,
        1.079000e-03,
        2.054300e-04,
        0.0,
        0.0,
        0.0,
        7.907690e-02,
        1.699900e-01,
        2.586000e-02,
        4.925600e-03,
        0.0,
        0.0,
        0.0,
        3.734000e-05,
        9.975700e-02,
        2.067900e-01,
        2.447800e-02,
        0.0,
        0.0,
        0.0,
        0.0,
        9.174200e-04,
        3.167740e-01,
        2.387600e-01,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        4.979300e-02,
        1.09910e00,
    ]
)
sigma_f = numpy.array(
    [
        4.79002e-09,
        5.82564e-09,
        4.63719e-07,
        5.24406e-06,
        1.45390e-07,
        7.14972e-07,
        2.08041e-06,
    ]
)
nu_sigma_f = numpy.array(
    [
        1.323401e-08,
        1.434500e-08,
        1.128599e-06,
        1.276299e-05,
        3.538502e-07,
        1.740099e-06,
        5.063302e-06,
    ]
)
chi = numpy.array([5.87910e-01, 4.11760e-01, 3.39060e-04, 1.17610e-07, 0.0, 0.0, 0.0])

# Create datasets for each cross-section type
fiss_chamber.create_dataset("total", data=sigma_t)
fiss_chamber.create_dataset("scatter matrix", data=sigma_s)
fiss_chamber.create_dataset("fission", data=sigma_f)
fiss_chamber.create_dataset("nu-fission", data=nu_sigma_f)
fiss_chamber.create_dataset("chi", data=chi)


############
# Guide Tube
############

# Create a subgroup for guide tube materials data
guide_tube = material_group.create_group("Guide Tube")

sigma_t = numpy.array(
    [
        1.260320e-01,
        2.931600e-01,
        2.842400e-01,
        2.809600e-01,
        3.344400e-01,
        5.656400e-01,
        1.172150e00,
    ]
)
sigma_s = numpy.array(
    [
        6.616590e-02,
        5.907000e-02,
        2.833400e-04,
        1.462200e-06,
        2.064200e-08,
        0.0,
        0.0,
        0.0,
        2.403770e-01,
        5.243500e-02,
        2.499000e-04,
        1.923900e-05,
        2.987500e-06,
        4.214000e-07,
        0.0,
        0.0,
        1.832970e-01,
        9.239700e-02,
        6.944600e-03,
        1.0803000e-03,
        2.056700e-04,
        0.0,
        0.0,
        0.0,
        7.885110e-02,
        1.701400e-01,
        2.588100e-02,
        4.929700e-03,
        0.0,
        0.0,
        0.0,
        3.733300e-05,
        9.973720e-02,
        2.067900e-01,
        2.447800e-02,
        0.0,
        0.0,
        0.0,
        0.0,
        9.172600e-04,
        3.167650e-01,
        2.387700e-01,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        4.979200e-02,
        1.099120e00,
    ]
)

sigma_f = numpy.zeros(7)
nu_sigma_f = numpy.zeros(7)
chi = numpy.zeros(7)

# Create datasets for each cross-section type
guide_tube.create_dataset("total", data=sigma_t)
guide_tube.create_dataset("scatter matrix", data=sigma_s)
guide_tube.create_dataset("fission", data=sigma_f)
guide_tube.create_dataset("nu-fission", data=nu_sigma_f)
guide_tube.create_dataset("chi", data=chi)


#######
# Water
#######

# Create a subgroup for water materials data
water = material_group.create_group("Water")

sigma_t = numpy.array(
    [
        1.592060e-01,
        4.129700e-01,
        5.903100e-01,
        5.843500e-01,
        7.180000e-01,
        1.254450e00,
        2.650380e00,
    ]
)
sigma_s = numpy.array(
    [
        4.447770e-02,
        1.134000e-01,
        7.234700e-04,
        3.749900e-06,
        5.318400e-08,
        0.0,
        0.0,
        0.0,
        2.823340e-01,
        1.299400e-01,
        6.234000e-04,
        4.800200e-05,
        7.448600e-06,
        1.045500e-06,
        0.0,
        0.0,
        3.452560e-01,
        2.245700e-01,
        1.699900e-02,
        2.644300e-03,
        5.034400e-04,
        0.0,
        0.0,
        0.0,
        9.102840e-02,
        4.155100e-01,
        6.373200e-02,
        1.213900e-02,
        0.0,
        0.0,
        0.0,
        7.143700e-05,
        1.391380e-01,
        5.118200e-01,
        6.122900e-02,
        0.0,
        0.0,
        0.0,
        0.0,
        2.215700e-03,
        6.999130e-01,
        5.373200e-01,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.324400e-01,
        2.480700e00,
    ]
)
sigma_f = numpy.zeros(7)
nu_sigma_f = numpy.zeros(7)
chi = numpy.zeros(7)

# Create datasets for each cross-section type
water.create_dataset("total", data=sigma_t)
water.create_dataset("scatter matrix", data=sigma_s)
water.create_dataset("fission", data=sigma_f)
water.create_dataset("nu-fission", data=nu_sigma_f)
water.create_dataset("chi", data=chi)


#############
# Control Rod
#############

# Create a subgroup for control rod materials data
control_rod = material_group.create_group("Control Rod")

sigma_t = numpy.array(
    [
        2.16768e-01,
        4.80098e-01,
        8.86369e-01,
        9.70009e-01,
        9.10482e-01,
        1.13775e00,
        1.84048e00,
    ]
)
sigma_s = numpy.array(
    [
        1.70563e-01,
        4.44012e-02,
        9.83670e-05,
        1.27786e-07,
        0.0,
        0.0,
        0.0,
        0.0,
        4.71050e-01,
        6.85480e-04,
        3.91395e-10,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        8.01859e-01,
        7.20132e-04,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        5.70752e-01,
        1.46015e-03,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        6.55562e-05,
        2.07838e-01,
        3.81486e-03,
        3.69760e-09,
        0.0,
        0.0,
        0.0,
        0.0,
        1.02427e-03,
        2.02465e-01,
        4.75290e-03,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        3.53043e-03,
        6.58597e-01,
    ]
)
sigma_f = numpy.zeros(7)
nu_sigma_f = numpy.zeros(7)
chi = numpy.zeros(7)

# Create datasets for each cross-section type
control_rod.create_dataset("total", data=sigma_t)
control_rod.create_dataset("scatter matrix", data=sigma_s)
control_rod.create_dataset("fission", data=sigma_f)
control_rod.create_dataset("nu-fission", data=nu_sigma_f)
control_rod.create_dataset("chi", data=chi)


######
# Clad
######

# Create a subgroup for Clad materials data
clad = material_group.create_group("Clad")

sigma_t = numpy.array(
    [
        1.30060e-01,
        3.05480e-01,
        3.29910e-01,
        2.69700e-01,
        2.72780e-01,
        2.77940e-01,
        2.95630e-01,
    ]
)
sigma_s = numpy.array(
    [
        9.72490e-02,
        3.25480e-02,
        0.00000e00,
        0.00000e00,
        0.00000e00,
        0.00000e00,
        0.00000e00,
        0.00000e00,
        3.03980e-01,
        7.72850e-04,
        0.00000e00,
        0.00000e00,
        0.00000e00,
        0.00000e00,
        0.00000e00,
        0.00000e00,
        3.24280e-01,
        5.94050e-04,
        0.00000e00,
        0.00000e00,
        0.00000e00,
        0.00000e00,
        0.00000e00,
        0.00000e00,
        2.63200e-01,
        5.31350e-03,
        0.00000e00,
        0.00000e00,
        0.00000e00,
        0.00000e00,
        0.00000e00,
        2.12680e-03,
        2.53950e-01,
        1.39080e-02,
        0.00000e00,
        0.00000e00,
        0.00000e00,
        0.00000e00,
        0.00000e00,
        1.48500e-02,
        2.41850e-01,
        1.65340e-02,
        0.00000e00,
        0.00000e00,
        0.00000e00,
        0.00000e00,
        0.00000e00,
        2.98990e-02,
        2.57160e-01,
    ]
)
sigma_f = numpy.zeros(7)
nu_sigma_f = numpy.zeros(7)
chi = numpy.zeros(7)

# Create datasets for each cross-section type
clad.create_dataset("total", data=sigma_t)
clad.create_dataset("scatter matrix", data=sigma_s)
clad.create_dataset("fission", data=sigma_f)
clad.create_dataset("nu-fission", data=nu_sigma_f)
clad.create_dataset("chi", data=chi)

######
# Void
######

# Create a subgroup for void materials data
void = material_group.create_group("Void")

sigma_t = numpy.array([0.0] * 7)
sigma_s = numpy.array([0.0] * 7 ** 2)
sigma_f = numpy.array([0.0] * 7)
nu_sigma_f = numpy.array([0.0] * 7)
chi = numpy.array([0.0] * 7)

# Create datasets for each cross-section type
void.create_dataset("total", data=sigma_t)
void.create_dataset("scatter matrix", data=sigma_s)
void.create_dataset("fission", data=sigma_f)
void.create_dataset("nu-fission", data=nu_sigma_f)
void.create_dataset("chi", data=chi)

# Close the hdf5 data file
f.close()
