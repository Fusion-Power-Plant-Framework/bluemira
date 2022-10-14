import math

import openmc

import volume_functions as vf


def check_geometry(geometry_variables):

    if geometry_variables.divertor_width > 0.95 * geometry_variables.minor_r:
        raise ValueError("The divertor width must be less than 95% of the minor radius")

    inboard_build = (
        geometry_variables.minor_r
        + geometry_variables.inb_fw_thick
        + geometry_variables.inb_bz_thick
        + geometry_variables.inb_mnfld_thick
        + geometry_variables.inb_vv_thick
        + geometry_variables.tf_thick
    )

    if inboard_build > geometry_variables.major_r:
        raise ValueError("The inboard build does not fit within the major radius")


def make_geometry(geometry_variables, material_lib):

    # Creates a OpenMC CSG geometry for a EU Demo reactor

    minor_r = geometry_variables.minor_r
    major_r = geometry_variables.major_r
    elong = geometry_variables.elong

    inb_fw_thick = geometry_variables.inb_fw_thick
    inb_bz_thick = geometry_variables.inb_bz_thick
    inb_mnfld_thick = geometry_variables.inb_mnfld_thick
    inb_vv_thick = geometry_variables.inb_vv_thick
    tf_thick = geometry_variables.tf_thick

    outb_fw_thick = geometry_variables.outb_fw_thick
    outb_bz_thick = geometry_variables.outb_bz_thick
    outb_mnfld_thick = geometry_variables.outb_mnfld_thick
    outb_vv_thick = geometry_variables.outb_vv_thick

    divertor_width = geometry_variables.divertor_width

    inner_plasma_r = major_r - minor_r
    outer_plasma_r = major_r + minor_r

    # Shifting the major radius of modelled torus (not the actual reactor major r)
    #  to get sensible fw profile and divertor whilst still using a torus surface.
    #  1 /3 of the minor radius is arbitrary number but looks sensible.
    dummy_maj_r = major_r - minor_r / 3
    dummy_min_r = (
        outer_plasma_r - dummy_maj_r
    )  # Need to keep outboard radial extent at the same place

    # This is a thin geometry layer to score peak surface values
    fw_surf_score_depth = 0.01

    ######################
    ### Inboard surfaces
    ######################

    bore_surface = openmc.ZCylinder(
        r=inner_plasma_r
        - (tf_thick + inb_vv_thick + inb_mnfld_thick + inb_bz_thick + inb_fw_thick)
    )
    tf_coil_surface = openmc.ZCylinder(
        r=inner_plasma_r
        - (inb_vv_thick + inb_mnfld_thick + inb_bz_thick + inb_fw_thick)
    )
    vv_inb_surface = openmc.ZCylinder(
        r=inner_plasma_r - (inb_mnfld_thick + inb_bz_thick + inb_fw_thick)
    )
    manifold_inb_surface = openmc.ZCylinder(
        r=inner_plasma_r - (inb_bz_thick + inb_fw_thick)
    )
    bz_inb_surface = openmc.ZCylinder(r=inner_plasma_r - (inb_fw_thick))
    fw_inb_surface_offset = openmc.ZCylinder(r=inner_plasma_r - fw_surf_score_depth)
    fw_inb_surface = openmc.ZCylinder(r=inner_plasma_r)

    divertor_surface = openmc.ZCylinder(r=inner_plasma_r + divertor_width)
    divertor_chop_surface = openmc.ZPlane(z0=0.0)

    maxz = (
        elong * minor_r
        + outb_fw_thick
        + outb_bz_thick
        + outb_mnfld_thick
        + outb_vv_thick
    )

    inb_top = openmc.ZPlane(z0=maxz, boundary_type="vacuum")
    inb_bot = openmc.ZPlane(z0=-maxz, boundary_type="vacuum")

    ######################
    ### Outboard surfaces
    ######################

    vert_semi_axis = elong * minor_r
    fw_outb_inner_surface = openmc.ZTorus(
        a=dummy_maj_r,  # Major radius of torus
        b=vert_semi_axis,  # Minor radius of torus in Z direction
        c=dummy_min_r,
    )  # Minor radius of torus in X-Y direction
    fw_outb_inner_surface_offset = openmc.ZTorus(
        a=dummy_maj_r,
        b=vert_semi_axis + fw_surf_score_depth,
        c=dummy_min_r + fw_surf_score_depth,
    )
    fw_rear_surface = openmc.ZTorus(
        a=dummy_maj_r, b=vert_semi_axis + outb_fw_thick, c=dummy_min_r + outb_fw_thick
    )
    bz_rear_surface = openmc.ZTorus(
        a=dummy_maj_r,
        b=vert_semi_axis + outb_fw_thick + outb_bz_thick,
        c=dummy_min_r + outb_fw_thick + outb_bz_thick,
    )  # 52
    manifold_rear_surface = openmc.ZTorus(
        a=dummy_maj_r,
        b=vert_semi_axis + outb_fw_thick + outb_bz_thick + outb_mnfld_thick,
        c=dummy_min_r + outb_fw_thick + outb_bz_thick + outb_mnfld_thick,
    )
    vv_rear_surface = openmc.ZTorus(
        a=dummy_maj_r,
        b=vert_semi_axis
        + outb_fw_thick
        + outb_bz_thick
        + outb_mnfld_thick
        + outb_vv_thick,
        c=dummy_min_r
        + outb_fw_thick
        + outb_bz_thick
        + outb_mnfld_thick
        + outb_vv_thick,
    )

    # Currently it is not possible to tally on boundary_type='vacuum' surfaces
    outer_surface = openmc.Sphere(
        r=major_r + minor_r + maxz + 1000.0, boundary_type="vacuum"
    )

    ######################
    ### Cells
    ######################

    cells = {}
    cells["bore_cell"] = openmc.Cell(
        region=-bore_surface & -inb_top & +inb_bot, name="Inner bore"
    )
    cells["tf_coil_cell"] = openmc.Cell(
        region=-tf_coil_surface & +bore_surface & -inb_top & +inb_bot, name="TF Coils"
    )
    cells["vv_inb_cell"] = openmc.Cell(
        region=-vv_inb_surface & +tf_coil_surface & -inb_top & +inb_bot,
        name="Vacuum Vessel",
    )
    cells["manifold_inb_cell"] = openmc.Cell(
        region=-manifold_inb_surface & +vv_inb_surface & -inb_top & +inb_bot,
        name="Inboard Manifold",
    )
    cells["bz_inb_cell"] = openmc.Cell(
        region=-bz_inb_surface & +manifold_inb_surface & -inb_top & +inb_bot,
        name="Inboard Breeder Zone",
    )
    cells["fw_inb_cell"] = openmc.Cell(
        region=-fw_inb_surface_offset & +bz_inb_surface & -inb_top & +inb_bot,
        name="Inboard First Wall",
    )

    divertor_region = (
        +fw_inb_surface
        & -divertor_surface
        & +fw_outb_inner_surface_offset
        & -vv_rear_surface
        & -divertor_chop_surface
    )
    div_surf_region = (
        +fw_inb_surface
        & -divertor_surface
        & +fw_outb_inner_surface
        & -fw_outb_inner_surface_offset
        & -divertor_chop_surface
    )
    cells["divertor_cell"] = openmc.Cell(region=divertor_region, name="Divertor")
    cells["div_surf_cell"] = openmc.Cell(
        region=div_surf_region, name="Divertor Surface"
    )
    r_inner = inner_plasma_r
    r_outer = inner_plasma_r + divertor_width
    cells["div_surf_cell"].volume = vf.calc_outb_inner_surf_vol(
        r_outer, r_inner, dummy_maj_r, dummy_min_r, vert_semi_axis, fw_surf_score_depth
    )

    cells["inner_vessel_cell"] = openmc.Cell(
        region=-fw_outb_inner_surface
        & +fw_inb_surface
        & ~divertor_region
        & ~div_surf_region,
        name="Plasma Cavity",
    )
    cells["fw_cell"] = openmc.Cell(
        region=-fw_rear_surface
        & +fw_outb_inner_surface_offset
        & +fw_inb_surface
        & ~divertor_region,
        name="Outboard First Wall",
    )
    cells["bz_cell"] = openmc.Cell(
        region=-bz_rear_surface & +fw_rear_surface & +fw_inb_surface & ~divertor_region,
        name="Outboard Breeder Zone",
    )
    cells["manifold_cell"] = openmc.Cell(
        region=-manifold_rear_surface
        & +bz_rear_surface
        & +fw_inb_surface
        & ~divertor_region,
        name="Outboard Manifold",
    )
    cells["vv_cell"] = openmc.Cell(
        region=-vv_rear_surface
        & +manifold_rear_surface
        & +fw_inb_surface
        & ~divertor_region,
        name="Outboard Vacuum Vessel",
    )

    cells["outer_vessel_cell1"] = openmc.Cell(
        region=-outer_surface & +vv_rear_surface & +fw_inb_surface
    )
    cells["outer_vessel_cell2"] = openmc.Cell(
        region=-outer_surface & +inb_top & -fw_inb_surface
    )
    cells["outer_vessel_cell3"] = openmc.Cell(
        region=-outer_surface & -inb_bot & -fw_inb_surface
    )

    ########################
    ### Assigning Materials
    ########################

    cells["divertor_cell"].fill = material_lib["water_cooled_steel_mat"]  #
    cells["div_surf_cell"].fill = material_lib["eurofer_mat"]
    cells["bore_cell"].fill     = material_lib["eurofer_mat"]
    cells["tf_coil_cell"].fill  = material_lib["eurofer_mat"]
    cells["vv_inb_cell"].fill   = material_lib["eurofer_mat"] #
    cells["manifold_inb_cell"].fill = material_lib["manifold_mat"]
    cells["bz_inb_cell"].fill   = material_lib["bz_mat"]
    cells["fw_inb_cell"].fill   = material_lib["fw_mat"]

    cells["fw_cell"].fill       = material_lib["fw_mat"]
    cells["bz_cell"].fill       = material_lib["bz_mat"]
    cells["manifold_cell"].fill = material_lib["manifold_mat"]
    cells["vv_cell"].fill       = material_lib["eurofer_mat"] #

    ################################
    ### Outboard Surface cells
    ################################

    # Specifies the number of outboard angles in the
    # Must be odd for surface maths to work
    ang_divs_in_180 = 7
    if ang_divs_in_180 % 2 == 0:
        raise ValueError(
            "For the surface to work the number of angular division must be odd"
        )

    ang_increment_rad = math.pi / ang_divs_in_180

    cells["fw_outb_surf_cells"] = []

    cyl_at_maj_r_sur = openmc.ZCylinder(r=major_r)

    # Bottom region inside major radius
    bottom_fw_cell = openmc.Cell(
        region=-fw_outb_inner_surface_offset
        & +fw_outb_inner_surface
        & -cyl_at_maj_r_sur
        & -divertor_chop_surface
        & +fw_inb_surface
        & ~divertor_region
        & ~div_surf_region,
        fill=material_lib["eurofer_mat"],
        name="Outboard FW Surface 0",
    )
    cells["fw_outb_surf_cells"].append(bottom_fw_cell)
    r_outer = major_r
    r_inner = inner_plasma_r + divertor_width
    cells["fw_outb_surf_cells"][-1].volume = vf.calc_outb_inner_surf_vol(
        r_outer, r_inner, dummy_maj_r, dummy_min_r, vert_semi_axis, fw_surf_score_depth
    )

    last_fw_ang_surface = cyl_at_maj_r_sur
    last_theta_tan = 0.0
    r, z_lower = vf.get_ellipse_rz_from_theta(
        dummy_min_r, dummy_maj_r, major_r, 0.0, vert_semi_axis
    )

    for div_surf_i in range(1, ang_divs_in_180 + 1):

        theta_rad = ang_increment_rad * div_surf_i
        theta_tan = math.tan(theta_rad)
        r, z_upper = vf.get_ellipse_rz_from_theta(
            dummy_min_r, dummy_maj_r, major_r, theta_rad, vert_semi_axis
        )

        # x2 + y2 = maj_r   when z = 0
        # maj_r2 = r2 z02
        # Creating chop surfaces at equal theta divisions from the major radius
        if (
            math.pi / 2 - 0.0001 < theta_rad < math.pi / 2 + 0.0001
        ):  # Avoiding tan blowing up
            fw_ang_surface = divertor_chop_surface

        elif abs(theta_tan) < 0.00001:  # Avoiding major_r / theta_tan blowing up
            fw_ang_surface = cyl_at_maj_r_sur

        else:
            fw_ang_surface = openmc.ZCone(
                x0=0.0, y0=0.0, z0=major_r / theta_tan, r2=theta_tan**2
            )

        # Creating cells
        if theta_tan > 0:
            cells["fw_outb_surf_cells"].append(
                openmc.Cell(
                    region=-fw_outb_inner_surface_offset
                    & +fw_outb_inner_surface
                    & -fw_ang_surface
                    & +last_fw_ang_surface
                    & -divertor_chop_surface
                    & +fw_inb_surface
                    & ~divertor_region
                    & ~div_surf_region
                )
            )

        elif theta_tan < 0 and last_theta_tan > 0:
            cells["fw_outb_surf_cells"].append(
                openmc.Cell(
                    region=-fw_outb_inner_surface_offset
                    & +fw_outb_inner_surface
                    & +fw_ang_surface
                    & +last_fw_ang_surface
                    & +fw_inb_surface
                    & ~divertor_region
                    & ~div_surf_region
                )
            )

        elif theta_tan < 0:
            cells["fw_outb_surf_cells"].append(
                openmc.Cell(
                    region=-fw_outb_inner_surface_offset
                    & +fw_outb_inner_surface
                    & +fw_ang_surface
                    & -last_fw_ang_surface
                    & +divertor_chop_surface
                    & +fw_inb_surface
                    & ~divertor_region
                    & ~div_surf_region
                )
            )

        # Adding cell properties
        cells["fw_outb_surf_cells"][-1].volume = vf.calc_outb_surf_vol(
            z_upper,
            z_lower,
            dummy_maj_r,
            dummy_min_r,
            vert_semi_axis,
            fw_surf_score_depth,
        )
        cells["fw_outb_surf_cells"][-1].fill = material_lib["eurofer_mat"]
        cells["fw_outb_surf_cells"][-1].name = "Outboard FW Surface " + str(div_surf_i)

        last_fw_ang_surface = fw_ang_surface
        last_theta_tan = theta_tan
        z_lower = z_upper

    # Top regions inside major radius - splitting into two (approximately): 0.6 is arbitrary
    theta_tan = 0.65 * divertor_width / vert_semi_axis
    fw_ang_surface = openmc.ZCone(
        x0=0.0, y0=0.0, z0=major_r / theta_tan, r2=theta_tan**2
    )

    cells["fw_outb_surf_cells"].append(
        openmc.Cell(
            region=-fw_outb_inner_surface_offset
            & +fw_outb_inner_surface
            & -cyl_at_maj_r_sur
            & +divertor_chop_surface
            & -fw_ang_surface
            & +fw_inb_surface
            & ~divertor_region
            & ~div_surf_region
        )
    )
    cells["fw_outb_surf_cells"].append(
        openmc.Cell(
            region=-fw_outb_inner_surface_offset
            & +fw_outb_inner_surface
            & -cyl_at_maj_r_sur
            & +divertor_chop_surface
            & +fw_ang_surface
            & +fw_inb_surface
            & ~divertor_region
            & ~div_surf_region
        )
    )
    cells["fw_outb_surf_cells"][-2].name = "Outboard FW Surface " + str(div_surf_i + 1)
    cells["fw_outb_surf_cells"][-1].name = "Outboard FW Surface " + str(div_surf_i + 2)

    cells["fw_outb_surf_cells"][-2].fill = material_lib["eurofer_mat"]
    cells["fw_outb_surf_cells"][-1].fill = material_lib["eurofer_mat"]

    r_inner = inner_plasma_r
    r_outer, z = vf.get_ellipse_rz_from_theta(
        dummy_min_r,
        dummy_maj_r,
        major_r,
        math.pi + math.atan(theta_tan),
        vert_semi_axis,
    )

    # print("\ntheta_rad", math.pi + math.atan(theta_tan))
    # print("\nradii", r_inner, r_outer, major_r, "\n")

    cells["fw_outb_surf_cells"][-2].volume = vf.calc_outb_inner_surf_vol(
        r_outer, r_inner, dummy_maj_r, dummy_min_r, vert_semi_axis, fw_surf_score_depth
    )

    r_inner = r_outer
    r_outer = major_r
    cells["fw_outb_surf_cells"][-1].volume = vf.calc_outb_inner_surf_vol(
        r_outer, r_inner, dummy_maj_r, dummy_min_r, vert_semi_axis, fw_surf_score_depth
    )

    #################################################
    ### Inboard Surface Cells
    #################################################

    cells["fw_inb_surf_cells"] = []
    regions_in_inb_fw = 9
    fw_inb_z_increment = 2.0 * maxz / regions_in_inb_fw
    last_fw_surface = openmc.ZPlane(z0=-maxz)

    for region_surf_i in range(1, regions_in_inb_fw + 1):

        region_surface = openmc.ZPlane(z0=-maxz + region_surf_i * fw_inb_z_increment)
        inb_surf_cell = openmc.Cell(
            region=-fw_inb_surface
            & +fw_inb_surface_offset
            & -region_surface
            & +last_fw_surface,
            fill=material_lib["eurofer_mat"],
            name="Inboard FW Surface " + str(region_surf_i),
        )
        cells["fw_inb_surf_cells"].append(inb_surf_cell)

        cells["fw_inb_surf_cells"][-1].volume = (
            fw_inb_z_increment
            * math.pi
            * (inner_plasma_r**2 - (inner_plasma_r - fw_surf_score_depth) ** 2)
        )

        last_fw_surface = region_surface

    #################################################
    ### Creating Universe
    #################################################

    # Note that the order in the universe list doesn't define the order in the output,
    # it is defined by the order in which each cell variable is created
    universe = openmc.Universe(
        cells=[
            cells["bore_cell"],          # Cell 1
            cells["tf_coil_cell"],       # Cell 2
            cells["vv_inb_cell"],        # Cell 3
            cells["manifold_inb_cell"],  # Cell 4
            cells["bz_inb_cell"],        # Cell 5
            cells["fw_inb_cell"],        # Cell 6
            cells["divertor_cell"],      # Cell 7
            cells["div_surf_cell"],      # Cell 8
            cells["inner_vessel_cell"],  # Cell 9
            cells["fw_cell"],            # Cell 10
            cells["bz_cell"],            # Cell 11
            cells["manifold_cell"],      # Cell 12
            cells["vv_cell"],            # Cell 13
            cells["outer_vessel_cell1"], # Cell 14
            cells["outer_vessel_cell2"], # Cell 15
            cells["outer_vessel_cell3"], # Cell 16
        ]
        + cells["fw_outb_surf_cells"]    # Cells 17 - 28  
        + cells["fw_inb_surf_cells"]     # Cells 30 - 36
    )

    geometry = openmc.Geometry(universe)
    geometry.export_to_xml()

    return cells, universe
