import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from bluemira.base.file import get_bluemira_path
from bluemira.geometry.tools import deserialise_shape
from bluemira.base.reactor_config import ConfigParams
from bluemira.display import plot_2d

NEUTRONICS_EUDEMO_DATA = get_bluemira_path("radiation_transport/neutronics/eudemo_test", subfolder="tests")

with open(Path(NEUTRONICS_EUDEMO_DATA, "panel_pts.json")) as j:
    panels = np.array(json.load(j))
with open(Path(NEUTRONICS_EUDEMO_DATA, "fw_curve.json")) as j:
    first_wall = deserialise_shape(json.load(j))
with open(Path(NEUTRONICS_EUDEMO_DATA, "div_curve.json")) as j:
    divertor = deserialise_shape(json.load(j))
with open(Path(NEUTRONICS_EUDEMO_DATA, "vv_interior.json")) as j:
    vv_int = deserialise_shape(json.load(j))
with open(Path(NEUTRONICS_EUDEMO_DATA, "vv_curve.new.json")) as j:
    vv_new = deserialise_shape(json.load(j))
with open(Path(NEUTRONICS_EUDEMO_DATA, "params.global_params.json")) as j:
    params = ConfigParams(global_params=json.load(j), local_params={})
with open(Path(NEUTRONICS_EUDEMO_DATA, "build_config.json")) as j:
    build_config = json.load(j)

def find_mc_values(x1: float, y1: float, x2: float, y2: float):
    m = (y2-y1)/(x2-x1)
    c = np.mean([y2-m*x2, y1-m*x1])
    return m, c

# transformation in the x-axis
m_x, c_x = find_mc_values(
            *(panels[:,0].min(), panels[:,0].min()),
            *(panels[:,0].max(), first_wall.get_optimal_bounding_box().x_max),
        )
# transformation in the z-axis
m_z, c_z = find_mc_values(
            *(panels[:,2].min(), divertor.start_point()[-1][-1]),
            *(panels[:,2].max(), first_wall.get_optimal_bounding_box().z_max),
        )
panel_points = np.zeros_like(panels)
panel_points[:, 0] = panels[:, 0]*m_x+c_x
panel_points[:, 2] = panels[:, 2]*m_z+c_z
ax = plot_2d(panel_points, show=False, point_options={"color":"blue"})
plot_2d(first_wall, ax=ax, show=False, wire_options={"color":"red", "linewidth":1.5})
plot_2d(divertor, ax=ax, show=False, wire_options={"color":"blue", "linewidth":1.5})
plot_2d(vv_int, ax=ax, show=False, wire_options={"color":"orange", "linewidth":1.5})
plot_2d(vv_new, ax=ax, show=False, wire_options={"color":"green", "linewidth":1.5})
plt.show()
with open("eudemo_test/panel_pts.updated.json", "w") as j:
    json.dump(panel_points.tolist())
    
import sys; sys.exit()


NEUTRONICS_STEP_DATA = get_bluemira_path("radiation_transport/neutronics/step_test", subfolder="tests")
with open(Path(NEUTRONICS_STEP_DATA, "lcfs.json")) as j:
    plasma_lcfs = deserialise_shape(json.load(j))
with open(Path(NEUTRONICS_STEP_DATA, "fw_inner.json")) as j:
    fw_inner = deserialise_shape(json.load(j))
with open(Path(NEUTRONICS_STEP_DATA, "fw_outer.json")) as j:
    fw_outer = deserialise_shape(json.load(j))
with open(Path(NEUTRONICS_STEP_DATA, "vv_outer.json")) as j:
    vv_outer = deserialise_shape(json.load(j))
with open(Path(NEUTRONICS_STEP_DATA, "vv_inner.json")) as j:
    vv_inner = deserialise_shape(json.load(j))
