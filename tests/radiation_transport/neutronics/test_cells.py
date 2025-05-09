# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import numpy as np
import pytest

from bluemira.geometry.coordinates import Coordinates
from bluemira.radiation_transport.neutronics.cells import Vertices

rng = np.random.default_rng(0)
rand = rng.random


class TestVertices:
    v = Vertices(rand(2), rand(2), rand(2), rand(2))

    def test_as_3d_array(self):
        assert (self.v.as_3d_array[0][::2] == self.v[0]).all()
        assert (self.v.as_3d_array[1][::2] == self.v[1]).all()
        assert (self.v.as_3d_array[2][::2] == self.v[2]).all()
        assert (self.v.as_3d_array[3][::2] == self.v[3]).all()

    def test_centroid(self):
        assert self.v.centroid[0] == np.mean([
            self.v[0][0],
            self.v[1][0],
            self.v[2][0],
            self.v[3][0],
        ])
        assert self.v.centroid[1] == np.mean([
            self.v[0][1],
            self.v[1][1],
            self.v[2][1],
            self.v[3][1],
        ])

    def test_convert_to_coordinates(self):
        a_vertex = self.v[0]
        as_coords = Vertices.convert_to_coordinates(a_vertex)
        assert (self.v[0] == as_coords.xz.flatten()).all()

    def test_convert_from_coordinates(self):
        copy = Vertices(
            Vertices.convert_from_coordinates(
                Coordinates([self.v[0][0], 0, self.v[0][1]])
            ),
            Vertices.convert_from_coordinates(
                Coordinates([self.v[1][0], 0, self.v[1][1]])
            ),
            Vertices.convert_from_coordinates(
                Coordinates([self.v[2][0], 0, self.v[2][1]])
            ),
            Vertices.convert_from_coordinates(
                Coordinates([self.v[3][0], 0, self.v[3][1]])
            ),
        )
        assert self.v == copy

    def test_from_bluemira_coordinates(self):
        copy = Vertices.from_bluemira_coordinates(
            Coordinates([self.v[0][0], 0, self.v[0][1]]),
            Coordinates([self.v[1][0], 0, self.v[1][1]]),
            Coordinates([self.v[2][0], 0, self.v[2][1]]),
            Coordinates([self.v[3][0], 0, self.v[3][1]]),
        )
        assert self.v == copy

    def test_eq(self):
        u = Vertices(self.v[0], self.v[1], self.v[2], self.v[3])
        assert self.v == u
        with pytest.raises(TypeError):
            bool(self.v == u.as_3d_array)

    def test_getitem(self):
        assert (self.v[0] == getattr(self.v, Vertices.index_mapping[0])).all()
        assert (self.v[1] == getattr(self.v, Vertices.index_mapping[1])).all()
        assert (self.v[2] == getattr(self.v, Vertices.index_mapping[2])).all()
        assert (self.v[3] == getattr(self.v, Vertices.index_mapping[3])).all()
        with pytest.raises(TypeError):
            self["an_unknown_variable"]

    def test_iter(self):
        assert len(list(self.v)) == 4

    def test_hash(self):
        assert isinstance(hash(self.v), int)
