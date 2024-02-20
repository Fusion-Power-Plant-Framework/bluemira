import json
from pathlib import Path

from bluemira.codes import _freecadapi as cadapi
# from bluemira.display import plot_2d, show_cad
from bluemira.geometry.wire import BluemiraWire

base_path = Path("~/Others/bluemira/examples/radiation_transport/data/").expanduser()
with open(Path(base_path, "divertor_face")) as j:
    divertor_face_dict = json.load(j)
cadWire = cadapi.deserialize_shape(divertor_face_dict['BluemiraFace']['boundary'][0]['BluemiraWire']['boundary'][0])
divertor_bmwire = BluemiraWire(cadWire)
# plot_2d(divertor_bmwire)