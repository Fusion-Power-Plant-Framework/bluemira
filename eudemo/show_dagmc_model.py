from dagmc_geometry_slice_plotter import plot_axis_slice, plot_slice

fn = "dagmc_models/mac.h5m"

plot = plot_axis_slice(
    dagmc_file_or_trimesh_object=fn,
    view_direction="y",
    plane_origin=[0, 200, 0],
)
# plot = plot_slice(
#     dagmc_file_or_trimesh_object=fn,
#     plane_normal=[0, 0, 1],
#     plane_origin=[0, 0, -600],
#     # rotate_plot=45,
# )

plot.show()
