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
OpenMC neutronics interface
"""
import json
import os
from copy import deepcopy
from itertools import cycle

from bluemira.base.file import get_bluemira_path, get_files_by_ext
from bluemira.materials import MaterialCache
from BLUEPRINT.base.palettes import BLUE
from BLUEPRINT.cad.blanketCAD import BlanketCAD
from BLUEPRINT.cad.buildingCAD import RadiationCAD
from BLUEPRINT.cad.coilCAD import CSCoilCAD, PFCoilCAD, TFCoilCAD
from BLUEPRINT.cad.cryostatCAD import CryostatCAD
from BLUEPRINT.cad.divertorCAD import DivertorCAD
from BLUEPRINT.cad.vesselCAD import VesselCAD
from BLUEPRINT.neutronics.constants import L_BP_TO_OMC

material_data_path = get_bluemira_path("materials", subfolder="data")
material_cache = MaterialCache()
material_cache.load_from_file(os.sep.join([material_data_path, "materials.json"]))
material_cache.load_from_file(os.sep.join([material_data_path, "mixtures.json"]))


# Populate materials

ss316 = material_cache.get_material("SS316-LN")
concrete = material_cache.get_material("HeavyConcrete")
HCPB_FW = material_cache.get_material("Homogenised_HCPB_2015_v3_FW")
HCPB_BZ = material_cache.get_material("Homogenised_HCPB_2015_v3_BZ")
HCPB_MB = material_cache.get_material("Homogenised_HCPB_2015_v3_MB")
HCPB_BSS = material_cache.get_material("Homogenised_HCPB_2015_v3_BSS")
div_mat = material_cache.get_material("Homogenised_Divertor_2015")
vessel_mat = material_cache.get_material("Steel Water 60/40")
tf_mat = material_cache.get_material("Toroidal_Field_Coil_2015")
pf_mat = material_cache.get_material("Poloidal_Field_Coil")


class MaterialFile:
    """
    A simple text writer for a list of material cards.
    """

    def __init__(self, filename, path):
        self.filename = filename
        self.path = path
        self.txt = """
        % Material input file.
        """

    def add_material(self, name, material, color, addition="-m"):
        """
        Add a material to the OpenMC material file.
        """
        self.txt += material.material_card(
            material_card_name=name + addition, color=color, code="openmc"
        )
        self.txt += "\n\n"

    def write(self):
        """
        Write the MaterialFile.
        """
        with open(os.path.join(self.path, self.filename), "w") as f:
            f.write(self.txt)
        return self.filename


def make_neutronics_CAD(fp, reactor):
    """
    Make the neutronics CAD for a reactor.

    Parameters
    ----------
    fp: str
        The path to the folder in which to place the CAD files
    reactor: Reactor
        The reactor for which to build the neutronics CAD.
    """
    a = {
        "BB": BlanketCAD,
        "DIV": DivertorCAD,
        "VV": VesselCAD,
        "RS": RadiationCAD,
        "CR": CryostatCAD,
        "ATEC": [PFCoilCAD, TFCoilCAD, CSCoilCAD],
    }
    for k, function in a.items():
        attr = getattr(reactor, k)
        if isinstance(function, list):
            for f in function:
                part = f(attr, neutronics=True)
                part.save_as_STEP(fp, scale=L_BP_TO_OMC)
        else:
            part = function(attr, neutronics=True)
            part.save_as_STEP(fp, scale=L_BP_TO_OMC)


def make_source(fp, reactor):
    """
    Make the neutron source file.

    Parameters
    ----------
    fp: str
        The path to the folder in which to place the neutron source file.
    reactor: Reactor
        The reactor for which to build the neutron source.
    """
    d = reactor.PL.export_neutron_source()
    fn = os.sep.join([fp, "plasma.json"])
    with open(fn, "w") as f:
        json.dump(d, f, indent=4)


LINK = {
    "_breeding_blanket_fw": HCPB_FW,
    "_breeding_blanket_bz": HCPB_BZ,
    "_breeding_blanket_mb": HCPB_MB,
    "_breeding_blanket_bss": HCPB_BSS,
    "_divertor": div_mat,
    "_radiation_shield": concrete,
    "_toroidal_field_coils_case": ss316,
    "_toroidal_field_coils_wp": tf_mat,
    "_poloidal_field_coils": pf_mat,
    "_central_solenoid": tf_mat,
    "_cryostat": ss316,
    "_reactor_vacuum_vessel": vessel_mat,
}

BLUEWHEEL = deepcopy(BLUE)


for k, v in BLUEWHEEL.items():
    if not isinstance(v, list):
        BLUEWHEEL[k] = [v]
for k, v in BLUEWHEEL.items():
    if not isinstance(v, cycle):
        BLUEWHEEL[k] = cycle(v)


CLINK = {
    "_breeding_blanket_fw": "BB",
    "_breeding_blanket_bz": "BB",
    "_breeding_blanket_mb": "BB",
    "_breeding_blanket_bss": "BB",
    "_divertor": "DIV",
    "_radiation_shield": "RS",
    "_toroidal_field_coils_case": "TF",
    "_toroidal_field_coils_wp": "TF",
    "_poloidal_field_coils": "PF",
    "_central_solenoid": "CS",
    "_cryostat": "CR",
    "_reactor_vacuum_vessel": "VV",
}


def _get_color(matname):
    k = CLINK[matname]
    return next(BLUEWHEEL[k])


def make_matfile(fp):
    """
    Make the materials file.

    Parameters
    ----------
    fp: str
        The path to the folder in which to place the material file.
    """
    mat_file = MaterialFile("materials.mc", fp)
    for k, v in LINK.items():
        mat_file.add_material(k, v, [0, 0, 0], addition="")
    mat_file.write()


def make_linkfile(fp):
    """
    Make the CAD-material linkage file.

    Parameters
    ----------
    fp: str
        The path to the folder in which to place the linkage file.
    """
    files = get_files_by_ext(fp, ".stp")
    d = []
    for file in files:
        entry = {}
        if "breeding_blanket" in file:
            if "fw" in file:
                entry["material"] = "_breeding_blanket_fw"
            elif "bss" in file:
                entry["material"] = "_breeding_blanket_bss"
            elif "manifold" in file:
                entry["material"] = "_breeding_blanket_mb"
            elif "bz" in file:
                entry["material"] = "_breeding_blanket_bz"
        else:
            for name in LINK.keys():
                if file.startswith(name):
                    entry["material"] = name
        entry["filename"] = file
        entry["colorRGB"] = _get_color(entry["material"])
        d.append(entry)
    with open(fp + "geometry_details.json", "w") as f:
        json.dump(d, f, indent=4)
