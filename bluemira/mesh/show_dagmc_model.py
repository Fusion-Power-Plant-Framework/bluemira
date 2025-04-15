from dagmc_geometry_slice_plotter import plot_slice

fn = "dagmc.h5m"

# plot = plot_axis_slice(
#     dagmc_file_or_trimesh_object=fn,
#     view_direction="z",
#     plane_origin=[0, 500, 0],
# )
plot = plot_slice(
    dagmc_file_or_trimesh_object=fn,
    plane_normal=[0, 1, 0],
    plane_origin=[0, 0, 0],
    # rotate_plot=45,
)

plot.show()
