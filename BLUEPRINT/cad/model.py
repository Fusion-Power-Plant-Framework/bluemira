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

"""
CAD model object for assemblies of components
"""
import os
import re
from collections import OrderedDict
from itertools import cycle

import numpy as np
from matplotlib.colors import to_rgb

from bluemira.base.look_and_feel import bluemira_print, bluemira_warn
from bluemira.utilities.tools import json_writer
from BLUEPRINT.base.names import name_short_long
from BLUEPRINT.base.palettes import BLUE
from BLUEPRINT.cad.cadtools import (
    make_axis,
    make_compound,
    rotate_shape,
    save_as_STEP_assembly,
)
from BLUEPRINT.cad.display import QtDisplayer
from BLUEPRINT.utilities.colortools import force_rgb


class CADModel:
    """
    CADModel is a generic object to contain other CAD objects.
    It's more of an agglomeration than anything else.

    Parameters
    ----------
    n_TF: int
        Number of toroidal field coils (used for patterning only). For
        non-cyclically symmetric CADModels, use n_TF=1.
    """

    # Formerly CAD
    def __init__(self, n_TF=1):

        self.n_TF = n_TF
        self.n_parts = 0
        self.parts = OrderedDict()  # container for build cad components
        self.palette = BLUE
        self.partcolors = {}

        # Constructors
        self.silo = None
        self._Q = None
        self._P = None

    def add_part(self, part):
        """
        Adds a part to the patterner

        Parameters
        ----------
        part: ComponentCAD object or Child thereof
            The part object to be added to the CADModel object
        """
        self.parts[part.name] = part
        self.n_parts += 1

    def map_global_pattern(self, pattern):
        """
        Defines the pattern for the CAD model.

        Parameters
        ----------
        pattern: str
            Global patterns for components in the CAD model.
                ['full', 'half', 'threequarter', 'third', 'quarter', 'sector']
                or alternatively:
                ['f', 'h', 't, 'q', 's', 0-9]
        """
        if pattern in ["full", "half", "threequarter", "third", "quarter", "sector"]:
            return [pattern] * self.n_parts
        if isinstance(pattern, str):
            if not bool(re.compile(r"[^thfqs0-9]").search(pattern)):
                while len(pattern) < self.n_parts:
                    pattern += "q"
                return pattern
        raise ValueError("Invalid pattern format.")

    def set_palette(self, palette):
        """
        Defines the color palette for the CAD model.

        Parameters
        ----------
        palette: dict
            Palette dictionary (keys: parts.keys(), values: colors)
        """
        palette = force_rgb(palette)
        palette = name_short_long(palette)
        self.palette = palette

        # Pick up colors for parts that have a matching name in the palette
        for name, colors in palette.items():
            # name = name_short_long(name)
            if name in self.parts.keys():
                if not isinstance(colors, list):
                    colors = [colors]
                self.partcolors[name] = cycle(colors)
            else:
                pass
                # bluemira_warn(f'Palette key "{name}" not in CAD model parts silo.')

        # Catch colors for parts that don't have an assigned name
        for name in self.parts:
            if name not in self.partcolors:
                colors = self.parts[name].component["colors"]
                if isinstance(colors, list):
                    self.partcolors[name] = cycle(colors)
                else:
                    self.partcolors[name] = cycle([to_rgb("grey")])
            # else:
            #    self.partcolors[name] = cycle(['grey'])

    def pattern(self, pattern):
        """
        Patterns the components in the CADModel object

        Parameters
        ----------
        pattern: str
            The patterning to apply to the CADModel
        """
        patterner = Patterner(self.n_TF, palette=self.palette)
        pindex = self.map_global_pattern(pattern)
        for partname, patt in zip(self.parts.keys(), pindex):
            patterner.add_part(self.parts[partname], pattern=patt)
        patterner.build()
        self._P = patterner  # for debugging
        self.silo = patterner.silo

    def display(self, pattern="sector"):
        """
        Displays the CADModel in a qt window

        Parameters
        ----------
        pattern: TBD
        """
        # If the palette hasn't been set, do that now
        if not self.partcolors:
            self.set_palette(self.palette)

        self.pattern(pattern)
        qt_display = QtDisplayer(wireframe=False)

        for part in self.silo:
            compounds = self.silo[part]["compounds"]
            colors = self.silo[part]["colors"]
            if hasattr(self, "partcolors"):
                colors = self.partcolors[part]
            trans = self.silo[part]["transparencies"]
            for compound, color, t in zip(compounds, colors, trans):
                qt_display.add_shape(compound, color, transparency=1 - t)
        self._Q = qt_display  # for debugging
        qt_display.show()

    def save_as_STEP_assembly(self, filename, partname=None, scale=1):  # noqa :N802
        """
        Saves the CADModel to a STEP assembly file

        Parameters
        ----------
        filename: str
            Full path filename to save the CADModel to
        partname: str
            The part name in the STEP file
        scale: float (default=1)
            The factor with which to scale the geometry
        """
        shapes = []
        if self.silo is None:
            self.pattern("sector")
        for component in self.silo.values():
            for compounds in component["compounds"]:
                shapes.append(compounds)
        save_as_STEP_assembly(shapes, filename=filename, partname=partname, scale=scale)

    def save_as_STEP(self, filepath, scale=1):  # noqa :N802
        """
        Exports the CADModel into individual STEP files

        Parameters
        ----------
        filepath:str
            DIrectory string in which to save the STEP files
        scale: float (default = 1)
            The factor with which to scale the geometry
        """
        for name, part in self.parts.items():
            filename = os.sep.join([filepath, name])
            part.save_as_STEP(filename, scale=scale)

    def save_as_STL(self, filepath, scale=1):  # noqa :N802
        """
        Exports the CADModel into individual STL files

        Parameters
        ----------
        filepath: str
            Directory string in which to save the STL files
        scale: float (default=1)
            The factor with which to scale the geometry
        """
        for name, part in self.parts.items():
            filename = os.sep.join([filepath, name])
            part.save_as_STL(filename, scale=scale)

    def save_component_names_as_json(self, filename):
        """
        Save the mapping of system to components as JSON format.

        Each system maps to a list of tuples, where is tuple contains
        a string holding the component name, and a unique ID that is
        equivalent to the order in which components are written to STP
        file in assembly mode.

        Parameters
        ----------
        filename: str
            Name of file in which to save.
        """
        # Populate the silo dict with all CAD components and their data
        if self.silo is None:
            self.pattern("sector")

        # Only get the name data
        component_dict = {}
        current_id = 0
        for system, component in self.silo.items():

            # Replace spaces with underscores for parity with STP output
            sysname = system.replace(" ", "_")

            # Get sub component names with spaces replaced
            names = [sub_name.replace(" ", "_") for sub_name in component["names"]]

            # Generate ids
            start_id = current_id + 1
            end_id = start_id + len(names)
            ids = list(range(start_id, end_id))
            current_id = end_id - 1

            # Make tuples of names and ids
            sub_components = list(zip(names, ids))

            # Save mapping of system to subcomponents
            component_dict[sysname] = sub_components

        if not filename.endswith(".json"):
            filename += ".json"
        bluemira_print(f"Writing {filename}")
        json_writer(component_dict, filename)


class Patterner:
    """
    Utility class to assist the cyclical patterning of CADModels
    """

    # Formerly buildCAD
    def __init__(self, n_TF, palette=None, **kwargs):
        self.n_TF = n_TF
        self.palette = palette
        self.slice_flag = kwargs.get("slice_flag", False)
        self.N = np.zeros(int(self.n_TF))  # left / right pattern
        self.N[1::2] = [i + 1 for i in range(len(self.N[1::2]))]
        self.N[2::2] = [-(i + 1) for i in range(len(self.N[2::2]))]
        self.axis = make_axis([0, 0, 0], [0, 0, 1])
        self.angle = 360 / self.n_TF
        self.parts = {}
        self.silo = {}

    def initalize_silo(self):
        """
        Initialises a dictionary data structure
        silo = {'partname': {'compounds': [], 'names': [], 'colors': [],
        'transparencies': []}, ..}
        """
        self.silo = {}

    def add_part(self, part, pattern=1):
        """
        Adds a part to the patterner

        Parameters
        ----------
        part: ComponentCAD object or Child thereof
            The part object to be patterned
        pattern: int or str
            The patterning to apply. Integers pattern the part n times. Strings
            are treated relative to nTF: ['full', 'half', 'threequarter',
            'third', 'quarter', 'sector'] or alternatively: ['f', 'h', 't,
            'q', 's']
        """
        n = self.set_pattern(pattern)
        self.parts[part.name] = {"part": part, "n": n}

    def set_pattern(self, pattern):
        """
        Translates pattern into number of rotated instances

        Parameters
        ----------
        pattern: int or str
            The patterning to apply. Integers pattern the part n times. Strings
            are treated relative to nTF: ['full', 'half', 'threequarter',
            'third', 'quarter', 'sector']

        Returns
        -------
        pattern: int
            The number of rotated instances to apply
        """
        if isinstance(pattern, int):
            n = pattern
        elif isinstance(pattern, float):
            n = int(pattern)
        elif pattern in ["full", "f"]:
            n = self.n_TF
        elif pattern in ["half", "h"]:
            n = int(np.ceil(self.n_TF / 2))
        elif pattern in ["threequarter", "t"]:
            n = int(np.ceil(3 / 4 * self.n_TF))
        elif pattern == "third":
            n = int(np.ceil(self.n_TF / 3))
        elif pattern in ["quarter", "q"]:
            n = int(np.ceil(self.n_TF / 4))
        elif pattern in ["sector", "s"]:
            n = 1
        elif pattern == "n":
            bluemira_warn("This thing you thought you could deprecate is being used.")
            n = 0  # use part subpattern number
        else:
            raise ValueError(
                "Specify part pattern as int or one of: ['full', "
                "'half', 'threequarter', 'third', 'quarter', "
                "'sector']."
            )
        return n

    def build(self):
        """
        Builds the silo of patterned parts and components
        """
        self.initalize_silo()
        for partname in self.parts:
            self.silo[partname] = {
                "compounds": [],
                "names": [],
                "colors": [],
                "transparencies": [],
            }
            n = self.parts[partname]["n"]
            component = self.parts[partname]["part"].component
            for shape, name, color, transp in zip(
                component["shapes"],
                component["names"],
                component["colors"],
                component["transparencies"],
            ):
                compound = self.pattern(shape, n, name)
                if compound:
                    self.add_compound(partname, name, compound, color, transp)

    def pattern(self, shape, n, name):
        """
        Creates a compound of patterned parts

        Parameters
        ----------
        shape: OCC TopoDS_* object
            The shape to be patterned
        n: int
            The patterning to apply
        name: str
            The name of the shape

        Returns
        -------
        compound: OCC Compound object
            The compound object of the patterned shapes
        """
        shapes = []
        if "TF_" in name and self.slice_flag is False:
            for i in (self.N - 1)[:n]:
                rshape = rotate_shape(shape, self.axis, np.pi + i * self.angle)
                shapes.append(rshape)

        if "Plasma" in name:
            if float(name.split("_")[-1]) == n - 1:
                rshape = rotate_shape(shape, self.axis, np.pi)
                compound = rshape
            else:
                compound = []  # TODO: Test None instead..
        else:
            for i in self.N[:n]:
                rshape = rotate_shape(shape, self.axis, np.pi + i * self.angle)
                shapes.append(rshape)
            compound = make_compound(shapes)
        return compound

    def add_compound(self, partname, name, compound, color, transp):
        """
        Adds a compound to the silo dictionary for a given part

        Parameters
        ----------
        partname: str
            The name of the part
        name: str
            The name of the component
        compound: OCC Compound object
            The compound for the component
        color: TODO
        transp: float
            The transparency of the components
        """
        self.silo[partname]["compounds"].append(compound)
        self.silo[partname]["names"].append(name)
        self.silo[partname]["colors"].append(color)
        self.silo[partname]["transparencies"].append(transp)
