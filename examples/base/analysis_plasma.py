import matplotlib.pyplot as plt

import bluemira.base as bm_base
import bluemira.geometry as geo


build_config = {
    "Plasma": {
        "class": "MakeParameterisedPlasma",
        "param_class": "bluemira.equilibria.shapes::JohnerLCFS",
        "variables_map": {
            "r_0": "R_0",
            "a": "A",
        },
        "target": "Plasma/LCFS",
        "segment_angle": 270.0,
    },
}
params = {
    "R_0": (9.0, "Input"),
    "A": (3.5, "Input"),
}
analysis = bm_base.Analysis(params, build_config)
analysis.run()

for dims in ["xy", "xz"]:
    component: bm_base.PhysicalComponent = analysis.component_manager.get_by_path(
        "/".join([dims, build_config["Plasma"]["target"]])
    )

    _, ax = plt.subplots()
    for wire in component.shape.boundary:
        shape = wire.discretize()
        ax.plot(*shape.T[0::2])
        ax.set_aspect("equal")
    plt.show()

component: bm_base.PhysicalComponent = analysis.component_manager.get_by_path(
    "/".join(["xyz", build_config["Plasma"]["target"]])
)
geo.tools.save_as_STEP(component.shape, "plasma")
