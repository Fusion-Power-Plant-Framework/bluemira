from bluemira.display import show_cad
from bluemira.geometry.tools import make_polygon

l = make_polygon(
    {
        "x": [0, 1],
        "z": [0, 1],
    }
)

show_cad(l)
