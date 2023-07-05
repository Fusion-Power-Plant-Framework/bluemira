from bluemira.display.displayer import show_cad
from bluemira.equilibria.shapes import JohnerLCFS
from bluemira.geometry.parameterisations import (
    PictureFrame,
    PolySpline,
    PrincetonD,
    TripleArc,
)

d = PrincetonD()
johner_wire = JohnerLCFS(var_dict={"r_0": {"value": 10.5}}).create_shape()
pf = PictureFrame().create_shape()
ps = PolySpline(
    {
        "bottom": {"value": 0.509036},
        "flat": {"value": 1},
        "height": {"value": 10.1269},
        "lower": {"value": 0.2},
        "tilt": {"value": 19.6953},
        "top": {"value": 0.46719},
        "upper": {"value": 0.326209},
        "x1": {"value": 5},
        "x2": {"value": 11.8222},
        "z2": {"value": -0.170942},
    }
).create_shape()
p = TripleArc()
p.adjust_variable("x1", value=4)
p.adjust_variable("dz", value=0)
p.adjust_variable("sl", value=0, lower_bound=0)
p.adjust_variable("f1", value=3)
p.adjust_variable("f2", value=3)
p.adjust_variable("a1", value=45)
p.adjust_variable("a2", value=45)
wire = p.create_shape()

parameterisation = TripleArc(
    {
        "x1": {
            "value": 3.2,
            "lower_bound": 3.0,
            "upper_bound": 3.2,
            "fixed": True,
        },
        "dz": {"value": -0.5, "upper_bound": -0.3},
        "a1": {"value": 100, "fixed": True},
    }
)

show_cad(wire)
