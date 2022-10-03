import math

pi = math.pi

########## Volume Functions ###################################################################################

def get_ellipse_rz_from_theta(dummy_min_r, dummy_maj_r, maj_r, theta_rad, vert_semi_axis):

    # Finds z where 2D ellipse and divided line intersect
    
    # Line
    # z = (maj_r - r) / tan( theta )
    #
    # Ellipse
    # (r - dummy_maj_r)**2 / dummy_min_r**2 + z**2 / vert_semi_axis**2 = 1 )
    
    print('\nCalc of ellipse')
    print(dummy_min_r, dummy_maj_r, maj_r, theta_rad, vert_semi_axis)
    
    if abs(theta_rad) < 1e-6:
        r = None
        z = - ( vert_semi_axis**2 * (1 - (maj_r - dummy_maj_r)**2 / dummy_min_r**2) )**0.5
        
    elif abs(abs(theta_rad) - pi) < 1e-6:
        r = None
        z =   ( vert_semi_axis**2 * (1 - (maj_r - dummy_maj_r)**2 / dummy_min_r**2) )**0.5 
        
    elif abs(theta_rad) < pi:
    
        tan2t = math.tan(theta_rad)**2
        
        a = dummy_min_r**-2 + vert_semi_axis**-2 / tan2t
        b = -2 * ( dummy_maj_r * dummy_min_r**-2 + maj_r * vert_semi_axis**-2 / tan2t )
        c = maj_r**2 * vert_semi_axis**-2 / tan2t + dummy_maj_r**2 * dummy_min_r**-2 - 1
        
        r = ( -b + ( b**2 - 4 * a * c )**0.5 ) / (2. * a)
        z = (maj_r - r) / math.tan( theta_rad )
        
    elif pi < theta_rad < 2*pi:
    
        tan2t = math.tan(theta_rad)**2
        
        a = dummy_min_r**-2 + vert_semi_axis**-2 / tan2t
        b = -2 * ( dummy_maj_r * dummy_min_r**-2 + maj_r * vert_semi_axis**-2 / tan2t )
        c = maj_r**2 * vert_semi_axis**-2 / tan2t + dummy_maj_r**2 * dummy_min_r**-2 - 1
        
        r = ( -b - ( b**2 - 4 * a * c )**0.5 ) / (2. * a)
        z = (maj_r - r) / math.tan( theta_rad )
        
    return r, z

###########################################################################################################################

def calc_ellipse_integral_term_for_z(z, r0, e_min_r, e_maj_r):

    # Calculates the integral term for z 
    
    # e_min_r is the ellipse minor radius
    # e_maj_r is the ellipse major radius

    return pi * ( r0**2 * z + 
                  r0 * e_min_r * z * ( 1 - z**2 / e_maj_r**2 )**0.5 + 
                  r0 * e_min_r * e_maj_r * math.asin( z / e_maj_r ) +
                  e_min_r**2 * z * ( 1 - z**2 / ( 3 * e_maj_r**2 ) )
                )

###########################################################################################################################

def calc_ellipse_integral_term_for_r(r, r0, e_min_r, e_maj_r):

    # Calculates the integral term for r 
    
    # e_min_r is the ellipse minor radius
    # e_maj_r is the ellipse major radius

    return pi * e_maj_r * ( e_min_r * r0 * math.asin( (r - r0) / e_min_r ) -
                            ( 1 - (r - r0)**2 / e_min_r**2 )**0.5 * 
                            ( 2 * e_min_r**2 - 2 * r**2 + r * r0 + r0**2) / 3 
                          )

###########################################################################################################################

def calc_ellipse_integral_for_z(z_upper, z_lower, r0, ellipse_min_r, ellipse_maj_r):

    # Calculates the integral of an ellipse with respect to z
    # Note this has a minor approximation which is acceptable because the layer is very thin

    return calc_ellipse_integral_term_for_z(z_upper, r0, ellipse_min_r, ellipse_maj_r) - \
           calc_ellipse_integral_term_for_z(z_lower, r0, ellipse_min_r, ellipse_maj_r)

###########################################################################################################################
           
def calc_ellipse_integral_for_r(r_outer, r_inner, r0, ellipse_min_r, ellipse_maj_r):

    # Calculates the integral of an ellipse with respect to r
    # Note this has a minor approximation which is acceptable because the layer is very thin
    
    return calc_ellipse_integral_term_for_r(r_outer, r0, ellipse_min_r, ellipse_maj_r) - \
           calc_ellipse_integral_term_for_r(r_inner, r0, ellipse_min_r, ellipse_maj_r)

###########################################################################################################################

def calc_outb_surf_vol(z_upper, z_lower, r0, ellipse_min_r, ellipse_maj_r, offset):

    # Calculates the outboard surface volume
    
    outer_maj_r = ellipse_maj_r + offset
    outer_min_r = ellipse_min_r + offset

    return calc_ellipse_integral_for_z(z_upper, z_lower, r0, outer_min_r, outer_maj_r) - \
           calc_ellipse_integral_for_z(z_upper, z_lower, r0, ellipse_min_r, ellipse_maj_r)

###########################################################################################################################

def calc_outb_inner_surf_vol(r_outer, r_inner, r0, ellipse_min_r, ellipse_maj_r, offset):

    # Calculates the inboard surface volume
    
    outer_maj_r = ellipse_maj_r + offset
    outer_min_r = ellipse_min_r + offset

    return calc_ellipse_integral_for_r(r_outer, r_inner, r0, outer_min_r, outer_maj_r) - \
           calc_ellipse_integral_for_r(r_outer, r_inner, r0, ellipse_min_r, ellipse_maj_r)     
           