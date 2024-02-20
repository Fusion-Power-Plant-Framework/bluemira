import json
from pathlib import Path
# from enum import Enum

import numpy as np
from matplotlib import pyplot as plt

from bluemira.codes import _freecadapi as cadapi
from bluemira.display import plot_2d, show_cad
from bluemira.geometry.wire import BluemiraWire
# from bluemira.geometry.tools import serialize_shape, deserialize_shape
# from Part import ArcOfCircle, LineSegment, BSplineCurve, BezierCurve
# from FreeCAD import Vector

_firstwall_A_max = (10, 25, 50)
_firstwall_dL_min = (0.1, 0.3, 0.5)
all_fw_panelling_variations = [
        (10, 0.1),
        (25, 0.1),
        (25, 0.3),
        (50, 0.3),
        (50, 0.5),
    ]
data_types = ("blanket_face", "divertor_face", "inner_boundary", "outer_boundary")

def read_file(data_type, fw_a_max, fw_dL_min):
    base_path = Path("~/Others/bluemira/examples/radiation_transport/data/.failed").expanduser()
    with open(Path(base_path, f"{data_type}_{fw_a_max}_{fw_dL_min}")) as j:
        return json.load(j)

if __name__=='__main__':
    # # confirmed that all the boundaries are the same
    # v = all_fw_panelling_variations[0]
    # first_in_bound, first_out_bound = get_inner_boundary(*v), get_outer_boundary(*v)
    # for fw_a_max, fw_dL_min in all_fw_panelling_variations[1:]:
    #     in_bound = get_inner_boundary(fw_a_max, fw_dL_min)
    #     out_bound = get_outer_boundary(fw_a_max, fw_dL_min)
    #     plot_2d([first_in_bound, first_out_bound, in_bound, out_bound])
    # For some reason all of the blanket_face files are the same??? (see hash)
    
    with open("divertor_face") as j:
        divertor_face_dict = json.load(j)
    cadWire = cadapi.deserialize_shape(divertor_face_dict['BluemiraFace']['boundary'][0]['BluemiraWire']['boundary'][0])
    divertor_bmwire = BluemiraWire(cadWire)
    plot_2d(divertor_bmwire)
