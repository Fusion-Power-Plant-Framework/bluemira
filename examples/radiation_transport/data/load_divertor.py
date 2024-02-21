import json
from pathlib import Path

import numpy as np
from numpy import typing as npt

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.codes import _freecadapi as cadapi
from bluemira.geometry.wire import BluemiraWire
from bluemira.geometry.constants import EPS_FREECAD

base_path = Path("~/Others/bluemira/examples/radiation_transport/data/").expanduser()

def convert_dict_to_cad_wire_list(d):
    """Convert dictionary as read in as a json file into a list of cad wire objects."""
    wire = d['BluemiraFace']['boundary'][0]['BluemiraWire']['boundary'][0]
    return [cadapi.deserialize_shape(cadapiWire) for cadapiWire in wire['Wire']]

def match_wire_start_and_ends(wire_list: list):
    """Find out where the wires' start and ends overlap"""
    start_points = [w.OrderedEdges[0].firstVertex().Point for w in wire_list]
    end_points = [w.OrderedEdges[-1].lastVertex().Point for w in wire_list]

    difference_matrix = [[e-s for s in start_points] for e in end_points]
    overlap_matrix = [[np.isclose(np.linalg.norm(diff), 0, rtol=0, atol=EPS_FREECAD) for diff in row] for row in difference_matrix]
    return overlap_matrix

def warn_when_ne_1(array, message_start, message_end):
    if any(array):
        bluemira_warn(message_start+str(array.nonzero()[0].tolist())+message_end)

def pop_matrix_row_and_column(matrix, index):
    """Remove row==index and column==index from that matrix"""
    smaller_matrix = matrix.copy()
    smaller_matrix.pop(index)
    [row.pop(index) for row in smaller_matrix]
    return smaller_matrix

def using_overlap_matrix(wire_list: list, overlap_matrix: npt.NDArray[bool]):
    """Reorder wires using the overlap matrix."""
    wire_list_copy = wire_list.copy()
    reordered_wire_list, discarded_wire_list = [], []

    warn_when_ne_1(np.sum(overlap_matrix, axis=0)<1, "The end of wire", "matches no other wire")
    warn_when_ne_1(np.sum(overlap_matrix, axis=0)>1, "The end of wire", "matches multiple wires")
    warn_when_ne_1(np.sum(overlap_matrix, axis=1)<1, "The start of wire", "matches no other wire")
    warn_when_ne_1(np.sum(overlap_matrix, axis=1)>1, "The start of wire", "matches multiple wires")
    current_focus = overlap_matrix[0]
    # Holy fuck I don't have time to get into graph theory so I shall stop here.
    # while wire_list_copy:
    #     nonzeros = np.nonzero(overlap_matrix[current_focus])[0]
    #     current_focus = np.zeros()

    #     top_row = overlap_matrix[0]
    #     if len(nonzeros)<1:
    #         discarded_wire_list.append(wire_list_copy[0])
    #         overlap_matrix = pop_matrix_row_and_column(overlap_matrix, 0)
    #     else:
    #         index = nonzeros[0]
    #         reordered_wire_list.append(wire_list_copy[index])
    #         overlap_matrix = pop_matrix_row_and_column(overlap_matrix, index)
    # return reordered_wire_list, discarded_wire_list

BMWire_from_list = lambda l: [BluemiraWire(w) for w in l]

if __name__=="__main__":
    from pprint import pprint
    import seaborn as sns
    import matplotlib.pyplot as plt
    from bluemira.display import plot_2d, show_cad
    from bluemira.geometry.tools import serialize_shape, deserialize_shape

    with open(Path(base_path, "blanket_face")) as j:
        blanket_face_dict = json.load(j)
    blanket_face_cad_wires = convert_dict_to_cad_wire_list(blanket_face_dict)
    matrix = match_wire_start_and_ends(blanket_face_cad_wires)
    # sns.heatmap(matrix)
    # plt.show()

    with open(Path(base_path, "divertor_face")) as j:
        divertor_face_dict = json.load(j)
    divertor_face_cad_wires = convert_dict_to_cad_wire_list(divertor_face_dict)
    matrix = match_wire_start_and_ends(divertor_face_cad_wires)
    blanket_face_reordered = np.array(divertor_face_cad_wires)[[0]+list(range(32, 0, -1))]
    inner_boundary = blanket_face_reordered[-1]
    plot_2d(BluemiraWire(inner_boundary))
    # plot_2d(BluemiraWire(blanket_face_reordered))
    just_divertor = BluemiraWire(blanket_face_reordered[:-1])
    plot_2d(just_divertor)

    with open("divertor_face.correct.json", "w") as j:
        json.dump(serialize_shape(just_divertor), j)
    with open("divertor_face.correct.json") as j:
        deserialized_object = deserialize_shape(json.load(j))
    plot_2d(deserialized_object)
    print(type(deserialized_object))
    print(deserialized_object)
    