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
Example of a meshed BLUEPRINT geometry imported into OpenMOC.
"""

# Common imports
from datetime import datetime
import sectionproperties.pre.sections as sections
from sectionproperties.analysis.cross_section import CrossSection

# OpenMOC must be installed to run this tutorial
import openmoc

# BLUEPRINT imports
from BLUEPRINT.base.file import get_bluemira_root, FileManager
from bluemira.base.look_and_feel import bluemira_print
from BLUEPRINT.neutronics.openmoc_geometry_tools import (
    create_system_cells,
    PlaneHelper,
)
from BLUEPRINT.systems.blanket import BreedingBlanket
from BLUEPRINT.systems.buildings import RadiationShield
from BLUEPRINT.systems.cryostat import Cryostat
from BLUEPRINT.systems.divertor import Divertor
from BLUEPRINT.systems.pfcoils import PoloidalFieldCoils
from BLUEPRINT.systems.plasma import Plasma
from BLUEPRINT.systems.tfcoils import ToroidalFieldCoils
from BLUEPRINT.systems.vessel import VacuumVessel

# Get the configuration from EUDEMO
from tests.test_reactor import config, build_config


#####################
# Define some options
#####################

start = datetime.now()


# Set boundary range
x_min, x_max = (0.0, 23.0)
y_min, y_max = (-23.0, 15.0)

# Set default cleaning parameters
mesh_sizes = [1.0]
min_length = 0.2
min_angle = 30

# Set systems to include
include_blanket = True
include_cryostat = True
include_divertor = True
include_pf_coils = True
include_rad_shield = True
include_tf_coils = True
include_vessel = True

# Set the OpenMOC logging level
openmoc.log.set_log_level("NORMAL")

# Write verbose mesh cleaning output?
verbose = True

# Import some materials
# Note that example_materials.py should be run before loading this
materials = openmoc.materialize.load_from_hdf5(
    f"{get_bluemira_root()}/examples/neutronics/example_materials.h5", ""
)

bluemira_print("Creating surfaces...")

# Initialise the PlaneHelper cache to avoid coplanar lines.
plane_helper = PlaneHelper()

# Define boundary points, facets and planes
boundary1 = [x_min, y_min]
boundary2 = [x_min, y_max]
boundary3 = [x_max, y_max]
boundary4 = [x_max, y_min]

boundary_points = [boundary1, boundary2, boundary3, boundary4]
boundary_facets = [
    [i, (i + 1) % len(boundary_points)] for i in range(len(boundary_points))
]

# Use a reflective left boundary as we assume the reactor in symmetric in x-z around x=0
left = openmoc.XPlane(x=x_min, name="left")
left.setBoundaryType(openmoc.REFLECTIVE)
plane_helper.add_plane(left)

# Use vacuum boundaries for all other bounding planes
right = openmoc.XPlane(x=x_max, name="right")
right.setBoundaryType(openmoc.VACUUM)
plane_helper.add_plane(right)

bottom = openmoc.YPlane(y=y_min, name="bottom")
bottom.setBoundaryType(openmoc.VACUUM)
plane_helper.add_plane(bottom)

top = openmoc.YPlane(y=y_max, name="top")
top.setBoundaryType(openmoc.VACUUM)
plane_helper.add_plane(top)

# Define a FileManager so that we can get to reference data for this reactor
reactor_name = config["Name"]
file_manager = FileManager(
    reactor_name=reactor_name,
    reference_data_root=build_config.get("reference_data_root", "data/BLUEPRINT"),
    generated_data_root=build_config.get(
        "generated_data_root", "generated_data/BLUEPRINT"
    ),
)
file_manager.build_dirs()
reactor_path = file_manager.reference_data_dirs["root"]

# Load up the systems from saved pickles
plasma = Plasma.load(f"{reactor_path}/{reactor_name}_PL.pkl")

if include_blanket:
    blanket = BreedingBlanket.load(f"{reactor_path}/{reactor_name}_BB.pkl")

if include_cryostat:
    cryostat = Cryostat.load(f"{reactor_path}/{reactor_name}_CR.pkl")

if include_divertor:
    divertor = Divertor.load(f"{reactor_path}/{reactor_name}_DIV.pkl")

if include_pf_coils:
    pf_coils = PoloidalFieldCoils.load(f"{reactor_path}/{reactor_name}_PF.pkl")

if include_rad_shield:
    rad_shield = RadiationShield.load(f"{reactor_path}/{reactor_name}_RS.pkl")

if include_tf_coils:
    tf_coils = ToroidalFieldCoils.load(f"{reactor_path}/{reactor_name}_TF.pkl")

if include_vessel:
    vessel = VacuumVessel.load(f"{reactor_path}/{reactor_name}_VV.pkl")

# Get the systems' cross sections, points, facets, control points, and holes
bluemira_print("Getting plasma sections")
(
    plasma_sections,
    plasma_points,
    plasma_facets,
    plasma_control_points,
    plasma_holes,
) = plasma.generate_cross_sections(
    mesh_sizes=mesh_sizes, min_length=min_length, min_angle=min_angle, verbose=verbose
)

if include_blanket:
    bluemira_print("Getting blanket sections")
    (
        blanket_sections,
        blanket_points,
        blanket_facets,
        blanket_control_points,
        blanket_holes,
    ) = blanket.generate_cross_sections(
        mesh_sizes=mesh_sizes,
        min_length=min_length,
        min_angle=min_angle,
        verbose=verbose,
    )

if include_cryostat:
    bluemira_print("Getting cryostat sections")
    (
        cryostat_sections,
        cryostat_points,
        cryostat_facets,
        cryostat_control_points,
        cryostat_holes,
    ) = cryostat.generate_cross_sections(
        mesh_sizes=mesh_sizes,
        geometry_names=["plates"],
        min_length=min_length,
        min_angle=min_angle,
        verbose=verbose,
    )

if include_divertor:
    bluemira_print("Getting divertor sections")
    (
        divertor_sections,
        divertor_points,
        divertor_facets,
        divertor_control_points,
        divertor_holes,
    ) = divertor.generate_cross_sections(
        mesh_sizes=mesh_sizes,
        min_length=min_length,
        min_angle=min_angle,
        verbose=verbose,
    )

if include_pf_coils:
    bluemira_print("Getting poloidal field coils sections")
    (
        pf_coils_sections,
        pf_coils_points,
        pf_coils_facets,
        pf_coils_control_points,
        pf_coils_holes,
    ) = pf_coils.generate_cross_sections(mesh_sizes=mesh_sizes, verbose=verbose)

if include_rad_shield:
    bluemira_print("Getting radiation shield sections")
    (
        rad_shield_sections,
        rad_shield_points,
        rad_shield_facets,
        rad_shield_control_points,
        rad_shield_holes,
    ) = rad_shield.generate_cross_sections(
        mesh_sizes=mesh_sizes,
        geometry_names=["plates"],
        min_length=min_length,
        min_angle=min_angle,
        verbose=verbose,
    )

if include_tf_coils:
    bluemira_print("Getting toroidal field coils sections")
    (
        tf_coils_sections,
        tf_coils_points,
        tf_coils_facets,
        tf_coils_control_points,
        tf_coils_holes,
    ) = tf_coils.generate_cross_sections(
        mesh_sizes=mesh_sizes,
        min_length=min_length,
        min_angle=min_angle,
        verbose=verbose,
    )

if include_vessel:
    bluemira_print("Getting vacuum vessel sections")
    (
        vessel_sections,
        vessel_points,
        vessel_facets,
        vessel_control_points,
        vessel_holes,
    ) = vessel.generate_cross_sections(
        mesh_sizes=mesh_sizes,
        geometry_names=["2D profile"],
        min_length=min_length,
        min_angle=min_angle,
        verbose=verbose,
    )


# Define the cross section for the void
bluemira_print("Generating void cross section")
void_points = boundary_points + plasma_points

void_facets = boundary_facets
void_facets += [
    [facet[0] + len(void_facets), facet[1] + len(void_facets)] for facet in plasma_facets
]

void_holes = plasma_control_points

void_control_points = [[0.5 * x_max, 0.9 * y_max]]

if include_blanket:
    void_points += blanket_points

    void_facets += [
        [facet[0] + len(void_facets), facet[1] + len(void_facets)]
        for facet in blanket_facets
    ]

    void_holes += blanket_control_points

if include_cryostat:
    void_points += cryostat_points

    void_facets += [
        [facet[0] + len(void_facets), facet[1] + len(void_facets)]
        for facet in cryostat_facets
    ]

    void_holes += cryostat_control_points

if include_divertor:
    void_points += divertor_points

    void_facets += [
        [facet[0] + len(void_facets), facet[1] + len(void_facets)]
        for facet in divertor_facets
    ]

    void_holes += divertor_control_points

if include_pf_coils:
    void_points += pf_coils_points

    void_facets += [
        [facet[0] + len(void_facets), facet[1] + len(void_facets)]
        for facet in pf_coils_facets
    ]

    void_holes += pf_coils_control_points

if include_rad_shield:
    void_points += rad_shield_points

    void_facets += [
        [facet[0] + len(void_facets), facet[1] + len(void_facets)]
        for facet in rad_shield_facets
    ]

    void_holes += rad_shield_control_points

if include_tf_coils:
    void_points += tf_coils_points

    void_facets += [
        [facet[0] + len(void_facets), facet[1] + len(void_facets)]
        for facet in tf_coils_facets
    ]

    void_holes += tf_coils_control_points

if include_vessel:
    void_points += vessel_points

    void_facets += [
        [facet[0] + len(void_facets), facet[1] + len(void_facets)]
        for facet in vessel_facets
    ]

    void_holes += vessel_control_points

void_section = sections.CustomSection(
    void_points,
    void_facets,
    void_holes,
    void_control_points,
)
void_section.clean_geometry(verbose=verbose)

void_mesh = void_section.create_mesh(mesh_sizes=mesh_sizes)
void_cross_section = CrossSection(void_section, void_mesh)


########################################
# Create the OpenMOC universe and cells.
########################################


bluemira_print("Creating OpenMOC universe and cells...")

# Create the Universe
root_universe = openmoc.Universe(name="root universe")

# Create the cells for systems and add them to the universe
plasma_cells = create_system_cells(
    root_universe, plasma_sections, "plasma", materials["Void"], plane_helper
)

system_material = materials["MOX-4.3%"]

if include_blanket:
    # Blanket
    blanket_cells = create_system_cells(
        root_universe, blanket_sections, "blanket", system_material, plane_helper
    )

if include_cryostat:
    # Cryostat
    cryostat_cells = create_system_cells(
        root_universe, cryostat_sections, "cryostat", system_material, plane_helper
    )

if include_divertor:
    # Divertor
    divertor_cells = create_system_cells(
        root_universe, divertor_sections, "divertor", system_material, plane_helper
    )

if include_pf_coils:
    # PF Coils
    pf_coils_cells = create_system_cells(
        root_universe, pf_coils_sections, "pf_coils", system_material, plane_helper
    )

if include_rad_shield:
    # Radiation Shield
    rad_shield_cells = create_system_cells(
        root_universe, rad_shield_sections, "rad_shield", system_material, plane_helper
    )

if include_tf_coils:
    # TF Coils
    tf_coils_cells = create_system_cells(
        root_universe, tf_coils_sections, "tf_coils", system_material, plane_helper
    )

if include_vessel:
    # Vessel
    vessel_cells = create_system_cells(
        root_universe, vessel_sections, "vessel", system_material, plane_helper
    )


# Create cells for the void and add them to the universe.
void_cells = create_system_cells(
    root_universe, [void_cross_section], "void", materials["Void"], plane_helper
)


##############################
# Create the OpenMOC Geometry.
##############################


bluemira_print("Creating OpenMOC geometry...")

geometry = openmoc.Geometry()
geometry.setRootUniverse(root_universe)

end = datetime.now()

bluemira_print(f"Geometry created in {end - start}")
