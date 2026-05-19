# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Tests for the geometry-backend dispatcher (``bluemira.codes._geometryapi``)."""

from __future__ import annotations

import os
import subprocess  # noqa: S404
import sys

import pytest

from bluemira.codes import _geometryapi as cadapi


class TestGeometryapiDispatcher:
    def test_unknown_backend_value_raises(self):
        """Importing the dispatcher with an invalid backend env var raises."""
        # Run in a subprocess so we don't pollute the in-process module cache.
        env = dict(os.environ)
        env["BLUEMIRA_GEOMETRY_BACKEND"] = "blender"
        proc = subprocess.run(
            [sys.executable, "-c", "import bluemira.codes._geometryapi"],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode != 0
        assert "Unknown BLUEMIRA_GEOMETRY_BACKEND" in proc.stderr
        assert "blender" in proc.stderr

    def test_getattr_dispatches_private_names(self):
        """Underscore-prefixed names aren't star-exported, so attribute access
        has to go through the module-level ``__getattr__``. ``_wire_is_planar``
        exists on both backends.
        """
        assert callable(cadapi._wire_is_planar)

    def test_getattr_unknown_name_raises_attribute_error(self):
        with pytest.raises(AttributeError):
            _ = cadapi.this_attribute_definitely_does_not_exist
