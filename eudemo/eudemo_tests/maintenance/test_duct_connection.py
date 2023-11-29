# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import pytest

from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import boolean_fuse, extrude_shape, make_circle
from eudemo.maintenance.duct_connection import pipe_pipe_join


class TestPipePipeJoin:
    c1 = make_circle(radius=2.0, center=(5, -5, 0), axis=(0, 1, 0))
    c2 = make_circle(radius=2.2, center=(5, -5, 0), axis=(0, 1, 0))
    face = BluemiraFace([c2, c1])
    void_face = BluemiraFace(c1)
    target = extrude_shape(face, vec=(0, 10, 0))
    target_void = extrude_shape(void_face, vec=(0, 10, 0))

    @pytest.mark.parametrize(("tool_length", "n_faces"), [(5, 7), (10, 10)])
    def test_pipe_pipe_join(self, tool_length, n_faces):
        c1 = make_circle(radius=1.0, center=(5, 0, 5), axis=(0, 0, 1))
        c2 = make_circle(radius=1.1, center=(5, 0, 5), axis=(0, 0, 1))
        face = BluemiraFace([c2, c1])
        void_face = BluemiraFace(c1)
        tool = extrude_shape(face, vec=(0, 0, -tool_length))
        tool_void = extrude_shape(void_face, vec=(0, 0, -tool_length))
        new_shape_pieces = pipe_pipe_join(self.target, self.target_void, tool, tool_void)
        n_actual = len(boolean_fuse(new_shape_pieces).faces)
        assert n_actual == n_faces
