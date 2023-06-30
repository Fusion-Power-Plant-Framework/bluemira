from bluemira.display.displayer import show_cad
from bluemira.geometry.parameterisations import PrincetonD

d = PrincetonD()
show_cad(d.create_shape())
