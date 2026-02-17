import json

import numpy as np

from bluemira.display import plot_2d, show_cad
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import (
    boolean_cut,
    deserialise_shape,
    distance_to,
    make_polygon,
    revolve_shape,
    save_as_STP,
    split_wire,
)
from bluemira.geometry.wire import BluemiraWire

DEMO_MODE = False
if DEMO_MODE:
    REVOLVE_DEGREES = 180
    print(
        "Running in demonstrative mode: Run again with DEMO_MODE=False to save the 360°model!"
    )
else:
    REVOLVE_DEGREES = 360
    print("Running in non-demonstrative mode.")

with open("2d_outline_data.json") as j:
    data = json.load(j)
boundary = deserialise_shape(data["boundary"])
divertor_top_wire = deserialise_shape(data["divertor_wire"])
panel_break_points = np.array(data["panel_break_points"])
vacuum_vessel_wire = deserialise_shape(data["vacuum_vessel_wire"])
cut_inboard = distance_to(boundary, divertor_top_wire.start_point())[1][0][0]
cut_outboard = distance_to(boundary, divertor_top_wire.end_point())[1][0][0]

long_wire, residual_wire = split_wire(boundary, cut_inboard)
upper_wire, residual_2 = split_wire(long_wire, cut_outboard)
upper_boundary = BluemiraWire([residual_wire, upper_wire])
divertor_wire = BluemiraWire([divertor_top_wire, residual_2])

panel_break_points[0] = distance_to(divertor_top_wire, panel_break_points[0])[1][0][0]
panel_break_points[-1] = distance_to(divertor_top_wire, panel_break_points[-1])[1][0][0]
upper_bottom_wire = make_polygon([cut_inboard, *panel_break_points, cut_outboard])
blanket_wire = BluemiraWire([upper_bottom_wire, upper_boundary])

if DEMO_MODE:
    plot_2d([blanket_wire, divertor_wire, vacuum_vessel_wire])
vacuum_vessel_face = boolean_cut(
    BluemiraFace(vacuum_vessel_wire), BluemiraFace(boundary)
)[0]
blanket_face = BluemiraFace(blanket_wire)
divertor_face = BluemiraFace(divertor_wire)

blanket = revolve_shape(blanket_face, degree=REVOLVE_DEGREES)
divertor = revolve_shape(divertor_face, degree=REVOLVE_DEGREES)
vacuum_vessel = revolve_shape(vacuum_vessel_face, degree=REVOLVE_DEGREES)
if DEMO_MODE:
    show_cad([blanket, divertor, vacuum_vessel])

save_as_STP([blanket], "blanket.stp")
save_as_STP([divertor], "divertor.stp")
save_as_STP([vacuum_vessel], "vacuum_vessel.stp")
