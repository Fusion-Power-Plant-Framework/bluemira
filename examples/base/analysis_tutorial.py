"""
A basic tutorial for configuring running and running an analysis.
"""

import matplotlib.pyplot as plt

from bluemira.base import Analysis


build_config = {
    "Plasma": {
        "class": "MakeParameterisedShape",
        "param_class": "bluemira.equilibria.shapes::JohnerLCFS",
        "variables_map": {
            "r_0": "R_0",
            "a": "A",
        },
        "target": "xz/Plasma/Shape",
    },
    "TF Coils": {
        "class": "MakeParameterisedShape",
        "param_class": "PrincetonD",
        "variables_map": {
            "x1": "r_tf_in_centre",
            "x2": {
                "value": "r_tf_out_centre",
                "lower_bound": 8.0,
            },
            "dz": 0.0,
        },
        "target": "xz/TF Coils/Shape",
    },
}
params = {
    "R_0": (9.0, "Input"),
    "A": (3.5, "Input"),
    "r_tf_in_centre": (5.0, "Input"),
    "r_tf_out_centre": (15.0, "Input"),
}
analysis = Analysis(params, build_config)
analysis.run()

_, ax = plt.subplots()
for build in build_config.values():
    component = analysis.component_manager.get_by_path(build["target"])
    shape = component.shape.discretize()
    ax.plot(*shape.T[0::2])
ax.set_aspect("equal")
plt.show()
