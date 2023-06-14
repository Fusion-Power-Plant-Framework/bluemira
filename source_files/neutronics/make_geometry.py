"""Make the entire tokamak from scratch using user-provided variables."""
import math
import copy
import openmc

import volume_functions as vf
import numpy as np

cells = {}
surfaces = {}

# Setting the thickness of divertor below the first wall 
div_clearance = 49.

# ------------------------------------------------------------------------------------

def check_geometry(tokamak_geometry):
    
    # Some basic geometry checks
    
    if tokamak_geometry.elong < 1.0:
        raise ValueError("Elongation must be at least 1.0")

    # TODO update this
    inboard_build = (
        tokamak_geometry.minor_r
        + tokamak_geometry.inb_fw_thick
        + tokamak_geometry.inb_bz_thick
        + tokamak_geometry.inb_mnfld_thick
        + tokamak_geometry.inb_vv_thick
        + tokamak_geometry.tf_thick
        + tokamak_geometry.inb_gap
    )

    if inboard_build > tokamak_geometry.major_r:
        raise ValueError("The inboard build does not fit within the major radius. Increase the major radius.")

# ------------------------------------------------------------------------------------

def normalizeVec(bisX,  bisY):
    
    # Normalises a vector

    length = ( bisX**2 +  bisY**2 )**0.5

    return bisX / length, bisY / length
        
# ------------------------------------------------------------------------------------

def get_cone_eqn_from_two_points(p1, p2):
    
    # Gets the equation of the OpenMC cone surface from two points
    # Assumes x0 = 0 and y0 = 0
    
    # print("p1, p2:", p1, p2)
    
    if p2[2] > p1[2]:
        r1 = p1[0]
        z1 = p1[2]
        r2 = p2[0]
        z2 = p2[2]
    else:
        r1 = p2[0]
        z1 = p2[2]
        r2 = p1[0]
        z2 = p1[2]
        
    # print("r1, r2, z1, z2:", r1, r2, z1, z2)
    
    a = r1**2 - r2**2
    b = 2 * (z1 * r2**2 - z2 * r1**2)
    c = r1**2 * z2**2 - r2**2 * z1**2
    
    # print("a, b, c:", a, b, c)
    
    z0 = ( - b + ( b**2 - 4 * a * c )**0.5 ) / (2 * a)
    r02 = r1**2 / ( z1 - z0 )**2
    
    # print("z0, r02:", z0, r02)
    
    return z0, r02

# ------------------------------------------------------------------------------------

def makeOffsetPoly(oldX, oldY, offset, outer_ccw ):
    
    # Makes a larger polygon with the same angle from a specified offset
    # Not mathematically robust for all polygons
    
    num_points = len(oldX)
    
    newX = []
    newY = []

    for curr in range(num_points):
        prev = (curr + num_points - 1) % num_points
        next = (curr + 1) % num_points

        vnX =  oldX[next] - oldX[curr]
        vnY =  oldY[next] - oldY[curr]
        vnnX, vnnY = normalizeVec(vnX,vnY)
        nnnX = vnnY
        nnnY = -vnnX

        vpX =  oldX[curr] - oldX[prev]
        vpY =  oldY[curr] - oldY[prev]
        vpnX, vpnY = normalizeVec(vpX,vpY)
        npnX = vpnY * outer_ccw
        npnY = -vpnX * outer_ccw

        bisX = (nnnX + npnX) * outer_ccw
        bisY = (nnnY + npnY) * outer_ccw

        bisnX, bisnY = normalizeVec(bisX,  bisY)
        bislen = offset / np.sqrt((1 + nnnX*npnX + nnnY*npnY)/2)

        newX.append(oldX[curr] + bislen * bisnX)
        newY.append(oldY[curr] + bislen * bisnY)
        
    return (newX, newY)
        
# ------------------------------------------------------------------------------------

def offset_points(points, offset_cm):
    
    # Calls makeOffsetPoly with points in the format it expects to get the points of an offset polygon

    old_rs = []
    old_zs = []
    
    for point in points:
        old_rs.append( point[0] )
        old_zs.append( point[2] )    
    
    new_rs, new_zs = makeOffsetPoly(old_rs, old_zs, offset_cm, 1)

    layer_points = []
    for i, point in enumerate(points):
        layer_points.append( (new_rs[i], 0., new_zs[i]) ) 
        
    return np.array( layer_points )

# ------------------------------------------------------------------------------------

def shift_points(points, shift_cm):
    
    # Moves all radii of points outwards by shift_cm

    points[:, 0] += shift_cm
    
    return points

# ------------------------------------------------------------------------------------

def elongate(points, adjust_elong):
    
    # Adjusts the elongation of the points

    points[:, 2] *= adjust_elong
    
    return points

# ------------------------------------------------------------------------------------

def stretch_r(points, tokamak_geometry, stretch_r_val):
    
    # Moves the points in the r dimension away from the major radius by extra_r_cm
    
    tokamak_geometry.major_r
    
    points[:, 0] = (points[:, 0] - tokamak_geometry.major_r) * stretch_r_val + tokamak_geometry.major_r
    
    return points

# ------------------------------------------------------------------------------------

def get_min_r_of_points(points):
    
    # Adjusts the elongation of the points

    min_r = np.amin(points, axis=0)[0]
    
    return min_r

# ------------------------------------------------------------------------------------

def get_min_max_z_r_of_points(points):
    
    # Adjusts the elongation of the points

    min_z = np.amin(points, axis=0)[2]
    max_z = np.amax(points, axis=0)[2]
    max_r = np.amax(points, axis=0)[0]
    
    return min_z, max_z, max_r

# ------------------------------------------------------------------------------------

def create_inboard_layer(prefix_for_layer,
                         prefix_for_layer_behind,
                         layer_points,
                         num_inboard_points,
                         layer_name,
                         material_lib,
                        ):
    
    # Creates a layer of inboard cells for scoring 
    
    cells[ prefix_for_layer+ "_cells" ] = []
    surfaces[ prefix_for_layer+ "_surfs" ] = {}
    surfaces[ prefix_for_layer+ "_surfs" ]["cones"] = []  # runs bottom to top
    surfaces[ prefix_for_layer+ "_surfs" ]["planes"] = [] # runs bottom to top
    
    # Generating bottom plane
    region_bot = openmc.ZPlane(
        z0 = layer_points[ -1 ][2]
    )
    surfaces[ prefix_for_layer+ "_surfs"]["planes"].append(region_bot)
    
    for inb_i in range(1, num_inboard_points):
        
        # Making surfaces
        inb_z0, inb_r2 = get_cone_eqn_from_two_points( layer_points[-inb_i], layer_points[-inb_i -1] )
        
        region_cone_surface = openmc.ZCone(
            x0=0.0, y0=0.0, z0=inb_z0, r2=inb_r2
        )
        surfaces[ prefix_for_layer+ "_surfs"]["cones"].append( region_cone_surface )
        
        region_top = openmc.ZPlane(
            z0 = layer_points[-inb_i -1][2]
        )
        surfaces[ prefix_for_layer+ "_surfs"]["planes"].append(region_top)

        inb_cell = openmc.Cell(
            region= -surfaces[ prefix_for_layer+ "_surfs"]["cones"][-1],  
            fill=material_lib[ prefix_for_layer+ "_mat"],
            name=layer_name + " " + str(inb_i),
        )
        
        # Different top surface for top region
        if inb_i == num_inboard_points - 1:          # if top segment
            inb_cell.region = inb_cell.region & -surfaces["meeting_cone"]
        else:
            inb_cell.region = inb_cell.region & -surfaces[ prefix_for_layer+ "_surfs"]["planes"][-1]  # Recently appended top surface  

        # Different bottom surface for bottom region
        if inb_i == 1:          # if bottom segment
            inb_cell.region = inb_cell.region & +surfaces["divertor_surfs"]["chop_surf"]
        elif inb_i == 2:
            inb_cell.region = inb_cell.region & \
                              +surfaces[ prefix_for_layer+ "_surfs"]["planes"][-2] & \
                              +surfaces["divertor_surfs"]["chop_surf"]   
        else:
            inb_cell.region = inb_cell.region & +surfaces[ prefix_for_layer+ "_surfs"]["planes"][-2] 
            
        
        # Adding outside breeder zone surfaces
        if inb_i == 3:  # Middle section
            inb_cell.region = inb_cell.region \
                              & +surfaces[ prefix_for_layer_behind+ "_surfs"]["cones"][inb_i-2] \
                              & +surfaces[ prefix_for_layer_behind+ "_surfs"]["cones"][inb_i-1] \
                              & +surfaces[ prefix_for_layer_behind+ "_surfs"]["cones"][inb_i] 
        elif inb_i < 3:
            inb_cell.region = inb_cell.region  \
                              & +surfaces[ prefix_for_layer_behind+ "_surfs"]["cones"][inb_i-1] \
                              & +surfaces[ prefix_for_layer_behind+ "_surfs"]["cones"][inb_i] 
        elif inb_i > 3:
            inb_cell.region = inb_cell.region  \
                              & +surfaces[ prefix_for_layer_behind+ "_surfs"]["cones"][inb_i-2] \
                              & +surfaces[ prefix_for_layer_behind+ "_surfs"]["cones"][inb_i-1] 
            
            
        # Calculating volume for first wall - not perfect but very close as wall is thin
        if prefix_for_layer in ["inb_fw", "inb_sf"]:
            
            inb_cell.volume = vf.get_fw_vol(surfaces[ prefix_for_layer+        "_surfs"]["cones"][-1] ,        # outer_cone
                                            surfaces[ prefix_for_layer_behind+ "_surfs"]["cones"][inb_i-1],    # inner_cone 
                                            surfaces[ prefix_for_layer+        "_surfs"]["planes"][-1],        # top
                                            surfaces[ prefix_for_layer+        "_surfs"]["planes"][-2]         # bottom
                                           )
                                          
           
        # Appending to cell list
        cells[ prefix_for_layer+ "_cells"].append(inb_cell)
        
    print("Created", layer_name)

    return

# ------------------------------------------------------------------------------------

def create_outboard_layer(prefix_for_layer,
                         prefix_for_layer_behind,
                         layer_points,
                         num_outboard_points,
                         layer_name,
                         material_lib,
                        ):
    
    # Creates a layer of outboard cells for scoring 

    cells[ prefix_for_layer+ "_cells"] = []
    surfaces[ prefix_for_layer+ "_surfs"] = {}
    surfaces[ prefix_for_layer+ "_surfs"]["cones"] = []    # runs bottom to top
    surfaces[ prefix_for_layer+ "_surfs"]["planes"] = []   # runs bottom to top

    # Generating bottom plane
    region_bot = openmc.ZPlane(
        z0 = layer_points[0][2]
    )
    surfaces[ prefix_for_layer+ "_surfs"]["planes"].append(region_bot)
    
    for outb_i in range(0, num_outboard_points): 
        
        # Making surfaces
        outb_z0, outb_r2 = get_cone_eqn_from_two_points( layer_points[outb_i], layer_points[outb_i+1])
        
        region_cone_surface = openmc.ZCone(
            x0=0.0, y0=0.0, z0=outb_z0, r2=outb_r2
        )
        surfaces[ prefix_for_layer+ "_surfs"]["cones"].append(region_cone_surface)
        
        region_top = openmc.ZPlane(
            z0 = layer_points[outb_i + 1][2]
        )
        surfaces[ prefix_for_layer+ "_surfs"]["planes"].append(region_top)
       
        # Making cell
        outb_cell = openmc.Cell(
            region= +surfaces[ prefix_for_layer+ "_surfs"]["cones"][-1],  
            fill=material_lib[ prefix_for_layer+ "_mat"],
            name= layer_name + " " + str(outb_i),
        )
        
        # Different top surface for top region
        if outb_i == num_outboard_points - 1:          # if top segment
            outb_cell.region = outb_cell.region & +surfaces["meeting_cone"] 
        else:
            outb_cell.region = outb_cell.region & -surfaces[ prefix_for_layer+ "_surfs"]["planes"][-1]  # Recently appended top surface  
            
        # Different bottom surface for bottom region
        if outb_i == 0:          # if bottom segment
            outb_cell.region = outb_cell.region & +surfaces["divertor_surfs"]["outer_cone"]
        else:
            outb_cell.region = outb_cell.region & +surfaces[ prefix_for_layer+ "_surfs"]["planes"][-2]   
        
        
        # Get outer cone surfaces and add to cell region
        outer_cone_surf_1 = surfaces[ prefix_for_layer_behind+ "_surfs"]["cones"][outb_i]
        
        if outb_i < 3:
            outer_cone_surf_2 = surfaces[ prefix_for_layer_behind+ "_surfs"]["cones"][outb_i+1]
        else:
            outer_cone_surf_2 = surfaces[ prefix_for_layer_behind+ "_surfs"]["cones"][outb_i-1]
            
        # Completing region
        outb_cell.region = outb_cell.region & -outer_cone_surf_1 & -outer_cone_surf_2
        
        cells[ prefix_for_layer+ "_cells"].append(outb_cell)  
        
        # Calculating volume - not perfect but very close as wall is thin        
        if prefix_for_layer in ["outb_fw", "outb_sf"]:
            outb_cell.volume = vf.get_fw_vol(surfaces[ prefix_for_layer_behind+ "_surfs"]["cones"][outb_i],   # outer_cone
                                             surfaces[ prefix_for_layer+        "_surfs"]["cones"][-1],       # inner_cone 
                                             surfaces[ prefix_for_layer+        "_surfs"]["planes"][-1],      # top
                                             surfaces[ prefix_for_layer+        "_surfs"]["planes"][-2],      # bottom
                                            )
        
    print("Created", layer_name)

    return

# ------------------------------------------------------------------------------------

def create_divertor(div_points, outer_points, inner_points, material_lib):
    """This creates the divertors cells
    outer_points gives the bottom of the VV
    """
    div_fw_thick = 2.5                            # Divertor first wall thickness
    div_sf_thick = 0.01
    div_points_fw_back = offset_points(div_points,  div_fw_thick)
    div_sf_points      = offset_points(div_points, -div_sf_thick)
    
    # Want whichever is lower - 49cm below the divertor fw or the bottom of the vv
    min_z, dummy1, dummy2 = get_min_max_z_r_of_points( div_points ) 
    
    min_z_w_clearance = min_z - div_clearance
    bot_z = min( min_z_w_clearance, outer_points[0][2] )
        
    surfaces["divertor_bot"] = openmc.ZPlane( 
        z0 = bot_z                                # Want to finish at bottom of outboard vv
    )
    
    surfaces["divertor_inner_r"] = openmc.ZCylinder(
        r = inner_points[-2][0]
    )
    surfaces["divertor_r_chop_in"] = openmc.ZCylinder(
        r = div_points[1][0]
    )
    surfaces["divertor_r_chop_out"] = openmc.ZCylinder(
        r = div_points[3][0]
    )
    surfaces["divertor_outer_r"] = openmc.ZCylinder(
        r = outer_points[0][0]
    )
    
    surfaces["divertor_fw_surfs"] = []
    surfaces["divertor_fw_back_surfs"] = []
    surfaces["divertor_scoring_surfs"] = []
    
    for x in range(0, len(div_points)-1):     
        
        # Divertor surface
        div_z0, div_r2 = get_cone_eqn_from_two_points( div_points[x], div_points[x+1] )
        
        div_fw_surface = openmc.ZCone(
                x0=0.0, y0=0.0, z0=div_z0, r2=div_r2
            )
        surfaces["divertor_fw_surfs"].append( div_fw_surface )
        
         # Divertor back of first wall
        div_z0, div_r2 = get_cone_eqn_from_two_points( div_points_fw_back[x], div_points_fw_back[x+1] )
        
        div_fw_surface = openmc.ZCone(
                x0=0.0, y0=0.0, z0=div_z0, r2=div_r2
            )
        surfaces["divertor_fw_back_surfs"].append( div_fw_surface )
        
         # Scoring surfaces
        div_z0, div_r2 = get_cone_eqn_from_two_points( div_sf_points[x], div_sf_points[x+1] )
        
        div_scoring_surface = openmc.ZCone(
                x0=0.0, y0=0.0, z0=div_z0, r2=div_r2
            )
        surfaces["divertor_scoring_surfs"].append( div_scoring_surface )
        
    surfaces["divertor_fw_back_surfs"][1].z0
    surfaces["div_fw_back_surf_1_mid_z"] = openmc.ZPlane(z0=surfaces["divertor_fw_back_surfs"][1].z0, boundary_type="vacuum")

    # Creating divertor regions
    divertor_region_inner = (
        -surfaces["divertor_surfs"]["top_surface"]
        & -surfaces["divertor_surfs"]["chop_surf"] 
        & -surfaces["divertor_r_chop_in"]
        & (-surfaces["divertor_fw_back_surfs"][0]
         | +surfaces["divertor_fw_back_surfs"][1])
        & +surfaces["divertor_inner_r"] 
        & +surfaces["divertor_bot"]
    )
    divertor_region_mid = (
        +surfaces["divertor_r_chop_in"]
        & -surfaces["divertor_r_chop_out"]
        & +surfaces["divertor_fw_back_surfs"][1]
        & -surfaces["divertor_fw_back_surfs"][2]
        & +surfaces["divertor_bot"]
         | ( -surfaces["divertor_fw_back_surfs"][1]
            & -surfaces["div_fw_back_surf_1_mid_z"]
            & +surfaces["divertor_r_chop_in"]
            & -surfaces["divertor_r_chop_out"]
            & +surfaces["divertor_bot"])
    )
    divertor_region_outer = (
        -surfaces["divertor_surfs"]["top_surface"]
        & +surfaces["divertor_r_chop_out"] 
        & -surfaces["divertor_surfs"]["outer_cone"]
        & +surfaces["divertor_fw_back_surfs"][3] 
        & +surfaces["divertor_bot"]
        & -surfaces["divertor_outer_r"]
    )
    
    
    div_fw_region_inner = (
        -surfaces["divertor_surfs"]["top_surface"]
        & -surfaces["divertor_surfs"]["chop_surf"] 
        & -surfaces["divertor_r_chop_in"]
        & -surfaces["divertor_fw_surfs"][0]
        & +surfaces["divertor_fw_back_surfs"][0]
        & -surfaces["divertor_fw_back_surfs"][1]
        & +surfaces["tf_coil_surface"] 
    )
    div_fw_region_mid = (
        +surfaces["divertor_r_chop_in"]
        & -surfaces["divertor_r_chop_out"]
        & +surfaces["divertor_fw_surfs"][1]
        & -surfaces["divertor_fw_surfs"][2]
        & (-surfaces["divertor_fw_back_surfs"][1]
        | +surfaces["divertor_fw_back_surfs"][2])
    )
    div_fw_region_outer = (
        -surfaces["divertor_surfs"]["top_surface"]
        & +surfaces["divertor_r_chop_out"] 
        & -surfaces["divertor_surfs"]["outer_cone"]
        & +surfaces["divertor_fw_surfs"][3] 
        & -surfaces["divertor_fw_back_surfs"][3]
        & +surfaces["divertor_fw_back_surfs"][2]
    )
    
    
    div_sf_region_inner = (
        -surfaces["divertor_surfs"]["top_surface"]
        & -surfaces["divertor_surfs"]["chop_surf"] 
        & -surfaces["divertor_r_chop_in"]
        & -surfaces["divertor_scoring_surfs"][0]
        & +surfaces["divertor_fw_surfs"][0]
        & -surfaces["divertor_fw_surfs"][1]
        & +surfaces["tf_coil_surface"] 
    )
    div_sf_region_mid = (
        +surfaces["divertor_r_chop_in"]
        & -surfaces["divertor_r_chop_out"]
        & +surfaces["divertor_scoring_surfs"][1]
        & -surfaces["divertor_scoring_surfs"][2]
        & (-surfaces["divertor_fw_surfs"][1]
        | +surfaces["divertor_fw_surfs"][2])
    )
    div_sf_region_outer = (
        -surfaces["divertor_surfs"]["top_surface"]
        & +surfaces["divertor_r_chop_out"]
        & -surfaces["divertor_surfs"]["outer_cone"]
        & +surfaces["divertor_scoring_surfs"][3] 
        & -surfaces["divertor_fw_surfs"][3]
        & +surfaces["divertor_fw_surfs"][2]
    )
    
    # Making divertor cells
    div_inb = openmc.Cell(
        region=divertor_region_inner,
        name="Divertor Inner",
        fill=material_lib["divertor_mat"]
    )
    div_mid =  openmc.Cell(
        region=divertor_region_mid,
        name="Divertor Mid",
        fill=material_lib["divertor_mat"]
    )
    div_out = openmc.Cell(
        region=divertor_region_outer,
        name="Divertor Outer",
        fill=material_lib["divertor_mat"]
    )
    cells["divertor_cells"] = [ div_inb, div_mid, div_out ]
    
    # Making divertor first wall cells
    cells["divertor_fw"] = openmc.Cell(
        region= div_fw_region_inner | div_fw_region_mid | div_fw_region_outer,
        name="Divertor PFC",
        fill=material_lib["div_fw_mat"]
    )
    cells["divertor_fw"].volume = vf.get_div_fw_vol([surfaces["divertor_fw_surfs"][0],        # outer_cones
                                                     surfaces["divertor_fw_back_surfs"][1],
                                                     surfaces["divertor_fw_surfs"][2],
                                                     surfaces["divertor_fw_back_surfs"][3]
                                                    ],                                      
                                                    [surfaces["divertor_fw_back_surfs"][0],   # inner_cones
                                                     surfaces["divertor_fw_surfs"][1],
                                                     surfaces["divertor_fw_back_surfs"][2],
                                                     surfaces["divertor_fw_surfs"][3]
                                                    ],
                                                     list(div_points[:,0])                     # radii
                                                    )
    
    # Making divertor first wall surface cells
    cells["divertor_fw_sf"] = openmc.Cell(
        region=div_sf_region_inner | div_sf_region_mid | div_sf_region_outer,
        name="Divertor PFC Surface",
        fill=material_lib["div_sf_mat"]
    )
    cells["divertor_fw_sf"].volume = vf.get_div_fw_vol([surfaces["divertor_scoring_surfs"][0],   # outer_cones
                                                        surfaces["divertor_fw_surfs"][1],
                                                        surfaces["divertor_scoring_surfs"][2],
                                                        surfaces["divertor_fw_surfs"][3]
                                                       ],                                      
                                                       [surfaces["divertor_fw_surfs"][0],        # inner_cones
                                                        surfaces["divertor_scoring_surfs"][1],
                                                        surfaces["divertor_fw_surfs"][2],
                                                        surfaces["divertor_scoring_surfs"][3]
                                                       ],
                                                       list(div_points[:,0])                     # radii
                                                      )
    
    
    # Region inside the divertor first wall, i.e. part of the plasma chamber
    div_in1_region = (-surfaces["divertor_surfs"]["top_surface"]
                      & +surfaces["divertor_scoring_surfs"][0]
                      & -surfaces["divertor_scoring_surfs"][1]
                      & -surfaces["divertor_scoring_surfs"][3]
                      & -surfaces["divertor_surfs"]["chop_surf"]
                      & +surfaces["divertor_bot"]
                     )
    div_in2_region = (-surfaces["divertor_surfs"]["top_surface"]
                      & +surfaces["divertor_scoring_surfs"][1]
                      & +surfaces["divertor_scoring_surfs"][2]
                      & -surfaces["divertor_scoring_surfs"][3] 
                     )
    
    cells["divertor_inner1"] = openmc.Cell(
        region=div_in1_region,
        name="Divertor Inner 1"
    )
    cells["divertor_inner2"] = openmc.Cell(
        region=div_in2_region,
        name="Divertor Inner 2"
    )
    
    return

# ------------------------------------------------------------------------------------

def create_plasma_chamber():
    
    # Creating the cells that live inside the first wall

    cells["plasma_inner1"] = openmc.Cell(
        region=-surfaces["meeting_r_cyl"]
               & +surfaces["divertor_surfs"]["top_surface"],
        name="Plasma inner 1"
    )
    for inb_sf_surf in surfaces["inb_sf_surfs"]["cones"]:
        cells["plasma_inner1"].region = cells["plasma_inner1"].region & +inb_sf_surf
        
    cells["plasma_inner2"] = openmc.Cell(
        region=-surfaces["meeting_r_cyl"]
               & -surfaces["divertor_surfs"]["top_surface"]
               & +surfaces["divertor_surfs"]["chop_surf"],
        name="Plasma inner 2"
    )
    for inb_sf_surf in surfaces["inb_sf_surfs"]["cones"]:
        cells["plasma_inner2"].region = cells["plasma_inner2"].region & +inb_sf_surf

        
    cells["plasma_outer1"] = openmc.Cell(
        region=+surfaces["meeting_r_cyl"]
               & +surfaces["divertor_surfs"]["top_surface"],
        name="Plasma outer 1"
    )
    for outb_sf_surf in surfaces["outb_sf_surfs"]["cones"]:
        cells["plasma_outer1"].region = cells["plasma_outer1"].region & -outb_sf_surf
        
    cells["plasma_outer2"] = openmc.Cell(
        region=+surfaces["meeting_r_cyl"]
               & -surfaces["divertor_surfs"]["top_surface"]
               & +surfaces["divertor_surfs"]["chop_surf"],
        name="Plasma outer 2"
    )
    for outb_sf_surf in surfaces["outb_sf_surfs"]["cones"]:
        cells["plasma_outer2"].region = cells["plasma_outer2"].region & -outb_sf_surf
        
    return

# ------------------------------------------------------------------------------------

def make_geometry(tokamak_geometry, fw_points, div_points, num_inboard_points, material_lib):
    """
    Create a dictionary of cells 
    Parameters
    ----------
    tokamak_geometry: TokamakGeometry
        TokamakGeometry (child of dataclass) instance containing the attributes:
            elong
            inb_bz_thick
            inb_fw_thick
            inb_gap
            inb_mnfld_thick
            inb_vv_thick
            major_r
            minor_r
            outb_bz_thick
            outb_fw_thick
            outb_mnfld_thick
            outb_vv_thick
            tf_thick
        which are all either floats in cm, or dimensionless.
    fw_points:
        coordinates of sample points representing the blanket, where
        blanket = first wall MINUS divertor
        (Hence I think this variable is poorly named)
    div_points:
        coordinates of sample points representing the divertor
    num_inboard_points:
        number of points in fw points that represents the number of inboard points.
    material_lib: dict
        dictionary of materials {name:openmc.Material} used to create cells.
    """
    # Creates an OpenMC CSG geometry for an EU Demo reactor
    print('fw_points',fw_points)
    print('div_points', div_points)

    minor_r = tokamak_geometry.minor_r
    major_r = tokamak_geometry.major_r
    elong = tokamak_geometry.elong

    inb_fw_thick = tokamak_geometry.inb_fw_thick
    inb_bz_thick = tokamak_geometry.inb_bz_thick
    inb_mnfld_thick = tokamak_geometry.inb_mnfld_thick
    inb_vv_thick = tokamak_geometry.inb_vv_thick
    tf_thick = tokamak_geometry.tf_thick

    outb_fw_thick = tokamak_geometry.outb_fw_thick
    outb_bz_thick = tokamak_geometry.outb_bz_thick
    outb_mnfld_thick = tokamak_geometry.outb_mnfld_thick
    outb_vv_thick = tokamak_geometry.outb_vv_thick

    inner_plasma_r = major_r - minor_r
    outer_plasma_r = major_r + minor_r

    # This is a thin geometry layer to score peak surface values
    fw_surf_score_depth = 0.01
    
    # Of the points in fw_points, this specifies the number that define the outboard
    num_outboard_points = len(fw_points) - num_inboard_points
    
    print('\nNumber of inboard points', num_inboard_points )
    print('Number of outboard points', num_outboard_points,'\n' )
    
    #########################################
    ### Inboard surfaces behind breeder zone
    #########################################
    
    # Getting layer points
    outb_bz_points   =  offset_points(fw_points, outb_fw_thick)
    outb_mani_points =  offset_points(fw_points, outb_fw_thick + outb_bz_thick)
    outb_vv_points   =  offset_points(fw_points, outb_fw_thick + outb_bz_thick + outb_mnfld_thick)
    outer_points     =  offset_points(fw_points, outb_fw_thick + outb_bz_thick + outb_mnfld_thick + outb_vv_thick)
    
    inb_bz_points    =  offset_points(fw_points, inb_fw_thick)
    inb_mani_points  =  offset_points(fw_points, inb_fw_thick + inb_bz_thick)
    inb_vv_points    =  offset_points(fw_points, inb_fw_thick + inb_bz_thick + inb_mnfld_thick)
    inner_points     =  offset_points(fw_points, inb_fw_thick + inb_bz_thick + inb_mnfld_thick + inb_vv_thick)
    
    print( 'outb_bz_points\n',   outb_bz_points )
    print( 'outb_mani_points\n', outb_mani_points )
    
    # Getting surface scoring points
    sf_points   =  offset_points(fw_points, -fw_surf_score_depth)
    
    # Getting tf coil r surfaces
    back_of_inb_vv_r = get_min_r_of_points( inner_points )
    gap_between_vv_tf = tokamak_geometry.inb_gap
    
    surfaces["bore_surface"] = openmc.ZCylinder(
        r = back_of_inb_vv_r - gap_between_vv_tf - tf_thick 
    )
    surfaces["tf_coil_surface"] = openmc.ZCylinder(
        r = back_of_inb_vv_r - gap_between_vv_tf
    )
    
    # Getting tf coil top and bottom surfaces
    div_points_w_clearance = copy.deepcopy(div_points)
    div_points_w_clearance[:,2] -= div_clearance
    
    dummy, max_z, max_r = get_min_max_z_r_of_points( np.concatenate((outer_points, 
                                                                     inner_points[-num_inboard_points:]), axis=0) )
    min_z, dummy, dummy = get_min_max_z_r_of_points( np.concatenate((outer_points, 
                                                                     div_points_w_clearance), axis=0) )
    
    # Setting clearance between the top of the divertor and the container shell
    clearance = 5.0

    surfaces["inb_top"] = openmc.ZPlane(z0=max_z+clearance, boundary_type="vacuum")
    surfaces["inb_bot"] = openmc.ZPlane(z0=min_z-clearance, boundary_type="vacuum")
    
    
    # Making rough divertor surfaces
    div_z0, div_r2 = get_cone_eqn_from_two_points(fw_points[0], fw_points[-1])
    
    surfaces["divertor_surfs"] = {}
    surfaces["divertor_surfs"]["top_surface"] = openmc.ZCone(
          x0=0.0, y0=0.0, z0=div_z0, r2=div_r2
         )
    surfaces["divertor_surfs"]["chop_surf"] = openmc.ZPlane( 
        z0 = max(fw_points[0][2], fw_points[-1][2]) 
    )
    
    div_o_z0, div_o_r2 = get_cone_eqn_from_two_points(outer_points[0], fw_points[0] )
    
    surfaces["divertor_surfs"]["outer_cone"] = openmc.ZCone(
          x0=0.0, y0=0.0, z0=div_o_z0, r2=div_o_r2
         )
    
    #############################################################################################
    #############################################################################################
    ### Making inboard vv, manifold, breeder zone, and first wall
    #############################################################################################
    #############################################################################################
    
    # Meeting point between inboard and outboard
    surfaces["meeting_r_cyl"] = openmc.ZCylinder(
        r = fw_points[-num_inboard_points][0]
    )
    
    # Meeting cone at top between inboard and outboard
    z0, r2 = get_cone_eqn_from_two_points(fw_points[-num_inboard_points], inb_vv_points[-num_inboard_points] )
    surfaces["meeting_cone"] = openmc.ZCone(
        x0=0.0, y0=0.0, z0=z0, r2=r2
    )
    
    # Generating inboard cone surfaces (back of vv)
    surfaces["inner_surfs"] = {}
    surfaces["inner_surfs"]["cones"] = []  # runs bottom to top
    
    for inb_i in range(1, num_inboard_points): 
        inb_z0, inb_r2 = get_cone_eqn_from_two_points(inner_points[-inb_i], inner_points[-inb_i -1] )
        
        cone_surface = openmc.ZCone(
            x0=0.0, y0=0.0, z0=inb_z0, r2=inb_r2
        )
        surfaces["inner_surfs"]["cones"].append(cone_surface)
       
    
    ##################################
    ### Making inboard vacuum vessel
        
    create_inboard_layer("inb_vv", 
                         "inner", 
                         inb_vv_points,
                         num_inboard_points,
                         "Inboard VV" ,
                         material_lib)
        
    ##################################
    ### Making inboard manifold 
    
    create_inboard_layer("inb_mani", 
                         "inb_vv", 
                         inb_mani_points,
                         num_inboard_points,
                         "Inboard Manifold" ,
                         material_lib)
    
    ##################################
    ### Making inboard breeder zone 
    
    create_inboard_layer("inb_bz", 
                         "inb_mani", 
                         inb_bz_points,
                         num_inboard_points,
                         "Inboard BZ" ,
                         material_lib)
        
    ##################################
    ### Making inboard first wall   
    
    create_inboard_layer("inb_fw", 
                         "inb_bz", 
                         fw_points,
                         num_inboard_points,
                         "Inboard FW" ,
                         material_lib)
        
    ##################################
    ### Making inboard scoring pionts  
    
    create_inboard_layer("inb_sf", 
                         "inb_fw", 
                         sf_points,
                         num_inboard_points,
                         "Inboard FW Surface" ,
                         material_lib)
    
    #############################################################################################
    #############################################################################################
    ### Making outboard vv, manifold, breeder zone, and first wall
    #############################################################################################
    #############################################################################################
    
    # Generating outboard cone surfaces (back of vv)
    surfaces["outer_surfs"] = {}
    surfaces["outer_surfs"]["cones"] = []  # runs bottom to top
    
    for outb_i in range(0, num_outboard_points): 
        outb_z0, outb_r2 = get_cone_eqn_from_two_points(outer_points[outb_i], outer_points[outb_i+1])
        
        cone_surface = openmc.ZCone(
            x0=0.0, y0=0.0, z0=outb_z0, r2=outb_r2
        )
        surfaces["outer_surfs"]["cones"].append(cone_surface)
        
    
    ##################################
    ### Making outboard vv
    
    create_outboard_layer("outb_vv",
                         "outer",
                         outb_vv_points,
                         num_outboard_points,
                         "Outboard VV",
                         material_lib)
    
    ##################################
    ### Making outboard manifold
    
    create_outboard_layer("outb_mani",
                         "outb_vv",
                         outb_mani_points,
                         num_outboard_points,
                         "Outboard Manifold",
                         material_lib)
    
    ##################################
    ### Making outboard breeder zone 
    
    create_outboard_layer("outb_bz",
                         "outb_mani",
                         outb_bz_points,
                         num_outboard_points,
                         "Outboard BZ",
                         material_lib)
        
    ##################################
    ### Making outboard first wall
     
    create_outboard_layer("outb_fw",
                         "outb_bz",
                         fw_points,
                         num_outboard_points,
                         "Outboard FW",
                         material_lib)
    
    #######################################
    ### Making outboard first wall surface
     
    create_outboard_layer("outb_sf",
                         "outb_fw",
                         sf_points,
                         num_outboard_points,
                         "Outboard FW Surface",
                         material_lib)

    ######################
    ### Outboard surfaces
    ######################

    # Currently it is not possible to tally on boundary_type='vacuum' surfaces
    clearance_r = 50.
    surfaces["outer_surface_cyl"] = openmc.ZCylinder(
        r = max_r + clearance_r
    )
    
    container_steel_thick = 200.
    surfaces["graveyard_top"] = openmc.ZPlane( 
        z0 = max_z + container_steel_thick
    )
    surfaces["graveyard_bot"] = openmc.ZPlane( 
        z0 = min_z - container_steel_thick
    )
    surfaces["graveyard_cyl"] = openmc.ZCylinder(
        r = max_r + clearance_r + container_steel_thick,
        boundary_type="vacuum"
    ) 
   

    ######################
    ### Cells
    ######################

    ### Inboard cells
    cells["bore_cell"] = openmc.Cell(
        region=-surfaces["bore_surface"] & -surfaces["inb_top"] & +surfaces["inb_bot"], 
        name="Inner bore"
    )
    cells["tf_coil_cell"] = openmc.Cell(
        region=-surfaces["tf_coil_surface"] & +surfaces["bore_surface"] & -surfaces["inb_top"] & +surfaces["inb_bot"], 
        name="TF Coils",
        fill= material_lib["tf_coil_mat"]
    )
    
    ### Divertor
    create_divertor(div_points, outer_points, inner_points, material_lib)
    
    
    ### Plasma chamber
    create_plasma_chamber()
        
    
    ### Container cells
    cells["outer_vessel_cell"] = openmc.Cell(
        region=   -surfaces["outer_surface_cyl"] \
                & -surfaces["inb_top"] \
                & +surfaces["inb_bot"] \
                & +surfaces["tf_coil_surface"] \
                & ~cells["plasma_inner1"].region \
                & ~cells["plasma_inner2"].region \
                & ~cells["plasma_outer1"].region \
                & ~cells["plasma_outer2"].region \
                & ~cells["divertor_inner1"].region \
                & ~cells["divertor_inner2"].region \
                & ~cells["divertor_fw"].region \
                & ~cells["divertor_fw_sf"].region,
        name="Outer VV Container",
    )
    
    for cell in cells["inb_sf_cells"] + \
                cells["inb_fw_cells"] + \
                cells["inb_bz_cells"] + \
                cells["inb_mani_cells"] + \
                cells["inb_vv_cells"] + \
                cells["outb_sf_cells"] + \
                cells["outb_fw_cells"] + \
                cells["outb_bz_cells"] + \
                cells["outb_mani_cells"] + \
                cells["outb_vv_cells"] + \
                cells["divertor_cells"]:
                
        cells["outer_vessel_cell"].region = cells["outer_vessel_cell"].region & ~cell.region
    
    inner_container_region = ( -surfaces["outer_surface_cyl"] 
                               & -surfaces["inb_top"] 
                               & +surfaces["inb_bot"] )
    
    cells["outer_container"] = openmc.Cell(
        region= ~inner_container_region \
                & -surfaces["graveyard_cyl"] \
                & -surfaces["graveyard_top"] \
                & +surfaces["graveyard_bot"],
        name="Container steel",
        fill= material_lib["container_mat"]
    )

    
    #################################################
    ### Creating Universe
    #################################################

    # Note that the order in the universe list doesn't define the order in the output,
    # which is instead defined by the order in which each cell variable is created
    universe = openmc.Universe(
        cells=[
            cells["bore_cell"],          # Cell 
            cells["tf_coil_cell"],       # Cell 
            cells["plasma_inner1"],      # Cell 
            cells["plasma_inner2"],      # Cell 
            cells["plasma_outer1"],      # Cell
            cells["plasma_outer2"],      # Cell 
            cells["divertor_inner1"],    # Cell
            cells["divertor_inner2"],    # Cell
            cells["divertor_fw"],        # Cell
            cells["divertor_fw_sf"],     # Cell
            cells["outer_vessel_cell"],  # Cell 
            cells["outer_container"]    # Cell
        ]
        + cells["inb_sf_cells"]          # Cells
        + cells["inb_fw_cells"]          # Cells 
        + cells["inb_bz_cells"]          # Cells 
        + cells["inb_mani_cells"]        # Cells 
        + cells["inb_vv_cells"]          # Cells 
        
        + cells["outb_sf_cells"]          # Cells
        + cells["outb_fw_cells"]         # Cells  
        + cells["outb_bz_cells"]         # Cells 
        + cells["outb_mani_cells"]       # Cells 
        + cells["outb_vv_cells"]         # Cells 
        
        + cells["divertor_cells"]        # Cells 
    )

    geometry = openmc.Geometry(universe)
    geometry.export_to_xml()

    return cells, universe

    