# cd ~/Others/bluemira/
# run eudemo/blanket/panelling/_designer.py
import json
from bluemira.geometry.tools import deserialize_shape
    
with open("/home/ocean/Others/bluemira/examples/radiation_transport/data/inner_boundary") as j:
    inner_boundary = deserialize_shape(json.load(j))
with open("/home/ocean/Others/bluemira/examples/radiation_transport/data/outer_boundary") as j:
    outer_boundary = deserialize_shape(json.load(j))

def calculate_panels(fw_a_max, fw_dL_min):
    p = PanellingDesignerParams(Parameter("fw_a_max",fw_a_max, "degree"), Parameter("fw_dL_min",fw_dL_min, "m"))
    designer = PanellingDesigner(p, inner_boundary)
    XY_outline = designer.run()
    return XY_outline

_firstwall_A_max = (10, 25, 50)
_firstwall_dL_min = (0.1, 0.3, 0.5)
all_fw_panelling_variations = [
        (10, 0.1),
        (25, 0.1),
        (25, 0.3),
        (50, 0.3),
        (50, 0.5),
    ]
panel_arrangements = []
for (fw_a_max, fw_dL_min) in all_fw_panelling_variations:
    interior_panels = calculate_panels(fw_a_max, fw_dL_min)
    panel_arrangements.append(interior_panels)
    np.save(f"~/Others/bluemira/examples/radiation_transport/data/fw_panels_{fw_a_max}_{fw_dL_min}.npy", interior_panels)
