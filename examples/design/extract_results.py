from pathlib import Path

import openmc
import vtk
from vtkmodules.util import numpy_support


def numpyToVTK(data, output_file, scaling=(1, 1, 1)):
    data_type = vtk.VTK_FLOAT
    shape = data.shape

    flat_data_array = data.flatten()
    vtk_data = numpy_support.numpy_to_vtk(
        num_array=flat_data_array, deep=True, array_type=data_type
    )
    vtk_data.SetName("Heat Flux (mean)")

    half_x = int(0.5 * scaling[0] * (shape[0] - 1))
    half_y = int(0.5 * scaling[1] * (shape[1] - 1))
    half_z = int(0.5 * scaling[2] * (shape[2] - 1))

    img = vtk.vtkImageData()
    img.GetPointData().SetScalars(vtk_data)
    img.SetSpacing(scaling[0], scaling[1], scaling[2])
    img.SetDimensions(shape[0], shape[1], shape[2])
    img.SetOrigin(-half_x, -half_y, -half_z)

    # Save the VTK file
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(img)
    writer.Write()


# open the results file
omc_output_path = Path(__file__).parent / "omc"
dag_fn = "OptimisedReactor.h5m"
sp_path = omc_output_path / "statepoint.10.h5"
sp = openmc.StatePoint(sp_path.as_posix())

dagmc_univ = openmc.DAGMCUniverse(
    filename=dag_fn,
    auto_geom_ids=True,
).bounded_universe()

# access the TBR tally using pandas dataframes
tbr_cell_tally = sp.get_tally(name="tbr")

# print cell tally for the TBR
print(f"The reactor has a TBR of {tbr_cell_tally.mean.sum()}")
print(f"Standard deviation on the TBR is {tbr_cell_tally.std_dev.sum()}")

# extracts the mesh tally result
tbr_mesh_tally = sp.get_tally(name="tbr_on_mesh")

# gets the mesh used for the tally
mesh = tbr_mesh_tally.find_filter(openmc.MeshFilter).mesh

# writes the TBR mesh tally as a vtk file
mesh.write_data_to_vtk(
    filename="tritium_production_map.vtk",
    datasets={
        "mean": tbr_mesh_tally.mean
    },  # the first "mean" is the name of the data set label inside the vtk file
)

# access the heating tally using pandas dataframes
heating_cell_tally = sp.get_tally(name="heating")

# print cell tally results with unit conversion
# raw tally result is multipled by 4 as this is a sector model of 1/4 of the total model (90 degrees from 360)
# raw tally result is divided by 1e6 to convert the standard units of eV to MeV
print(
    f"The heating of {heating_cell_tally.mean.sum() / 1e6} MeV per source particle is deposited"
)
print(f"Standard deviation on the heating tally is {heating_cell_tally.std_dev.sum()}")

# extracts the mesh tally result
heating_mesh_tally = sp.get_tally(name="heating_on_mesh")

# gets the mesh used for the tally
mesh = heating_mesh_tally.find_filter(openmc.MeshFilter).mesh

# writes the TBR mesh tally as a vtk file
# mesh.write_data_to_vtk(
#     filename="heating_map.vtu",
#     datasets={"mean": heating_mesh_tally.mean},
#     volume_normalization=True,
# )

flux_mean = heating_mesh_tally.mean.reshape(100, 100, 100)
a = dagmc_univ.bounding_box.width
scaling = tuple(round(t / c) for c, t in zip(flux_mean.shape, a, strict=False))
numpyToVTK(flux_mean, "heat_flux_tot.vti", scaling)
