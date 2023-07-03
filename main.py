from bluemira.display.displayer import show_cad
from bluemira.equilibria.shapes import JohnerLCFS
from bluemira.geometry.parameterisations import PrincetonD

d = PrincetonD()
johner_wire = JohnerLCFS(var_dict={"r_0": {"value": 10.5}}).create_shape()
show_cad(johner_wire)
