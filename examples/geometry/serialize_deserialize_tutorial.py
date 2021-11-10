# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
#                    D. Short
#
# bluemira is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# bluemira is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with bluemira; if not, see <https://www.gnu.org/licenses/>.

import bluemira.geometry as geo
import jsonpickle
import json

import freecad
import Part
from FreeCAD import Base

import bluemira.geometry._freecadapi as _freecadapi

# subclass JSONEncoder
class BluemiraGeoEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, geo.wire.BluemiraWire):
            d = dict()
            d["type"] = type(obj).__name__
            d["label"] = obj.label
            d["boundary"] = [self.default(v) for v in
                             obj.boundary]
            return d
        elif isinstance(obj, Part.Wire):
            return _freecadapi.serialize_shape(obj)
        return super(BluemiraGeoEncoder, self).default(obj)

class BluemiraGeoDencoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.hook, *args, **kwargs)

    def hook(self, obj):
        if isinstance(obj, list):
            return [self.hook(x) for x in obj]
        if isinstance(obj, dict):
            if 'type' not in obj:
                for k,v in obj.items():
                    if k in ['Wire']:
                        return _freecadapi.deserialize_shape(obj)
                return obj
            type = obj['type']
            if type == 'BluemiraWire':
                label = obj['label']
                boundary = [self.hook(w) for w in obj['boundary']]
                bmwire = geo.wire.BluemiraWire(boundary=boundary, label=label)
                return bmwire
            return obj
        else:
            return obj

points = [[0,0,0], [1,0,0]]
bmwire = geo.wire.BluemiraWire(geo.wire.BluemiraWire(geo.tools.make_polygon(points,
                                                                            label="poly"),
                               label='wire1'), label="wire2")

print(BluemiraGeoEncoder().encode(bmwire))

geo_JSON_data = json.dumps(bmwire, indent=4, cls=BluemiraGeoEncoder)
#print(geo_JSON_data)

new_geo = json.loads(geo_JSON_data, cls=BluemiraGeoDencoder)
#print(new_geo)