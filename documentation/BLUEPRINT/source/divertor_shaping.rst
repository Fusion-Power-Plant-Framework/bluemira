Divertor Shaping
================

This document refers to the ``DivertorBuilder`` class in ``firstwall.py``.  

Overview
--------
Building the divertor is the last step, after the first wall shaping, 
to obtain the final wall.
While the first wall profile is shaped (``FirstWall`` class), via iterations, 
to minimise the heat flux onto the surfaces, and hence downstream of a heat 
flux calculation (``FluxSurface`` class), the divertor profile shaping can be 
seen as a geometrical optimisation. Such optimisation is obtained via external 
geometrical inputs, and according the equilibrium configuraion. Specifically, 
separatrix and flux lines in the divertor region. 
First wall profile and divertor profile are two different grometries that need 
to be "attached". The shared border between the two geometries, along whichh they
will be attached, is a xz plane passing through the x-point(s). Thus, the 
first wall profile is extended until the x-point, and the divertor profile starts 
from the x-point.

.. figure:: ../images/nova/profiles.png
   :scale: 70%
   :name: fig:profiles
   :align: center

   The two separate first wall profile and divertor profile 


Divertor entrance
-----------------
The divertor profile starts to be drawn from the x-point. 
Hence the divertor entrance has the z coordinate of the x-point. 
The lower threshold of how wide this aperture should be, is decided by the user 
through two inputs:

* xpt_outer_gap: Gap between x-point and outer wall
* xpt_inner_gap: Gap between x-point and inner wall

From the first wall profile optimisation, if the aperture needs be wider to handle 
the heat loads, the abovementioned inputs are overwritten accordingly.

Divertor target plates
----------------------
How long the divertor should be is decided by the divertor target plates.
As first input, the user has to decide the strike points, meaning where the separatrix
has to intersect inner and outer target. The two unique points are set giving:

* outer_strike_r: Outer strike point major radius
* inner_strike_r: Inner strike point major radius

However, these inputs can be constrained, shifting from independent to dependent variables.
This is the case if a "keep out zone" (koz) is given in input. If so, this draws a border 
outside of which the entire first wall profile cannot be extended.

Two functions, namely ``find_strike_points_from_koz`` and ``find_strike_points_from_params`` 
sort the options.

Once identified the strike points, the user can decide how long the plates should be, 
giving:

* tk_outer_target_sol: Outer target length between strike point and SOL side
* tk_outer_target_pfr: Outer target length between strike point and PFR side
* tk_inner_target_sol: Inner target length between strike point and SOL side
* tk_inner_target_pfr: Inner target length between strike point and PFR side

.. figure:: ../images/nova/entrance_strike.png
   :scale: 70%
   :name: fig:entrance_strike
   :align: center

   Schematic of key points to draw the divertor. :math:`E_1` and :math:`E_2` are the divertor entrance end points.
   :math:`S_1` and :math:`S_2` are the strike points. The target plates are within the keep out zone (blue line).

Finally the plates can be tilted. The user can assign an angle between flux line 
and target plate:

* theta_outer_target: Angle between flux line tangent at outer strike point and SOL side of outer target
* theta_inner_target: Angle between flux line tangent at inner strike point and SOL side of inner target

To be noticed, the abovementioned :math:`\theta` is not the glancing angle, 
commonly indicated with :math:`\gamma`, but its component over a poloidal plane.
A further, and better description should accept the actual glancing angle as input.

Divertor legs
-------------
Once top limit (divertor entrance) and bottom limit (divertor targets) are defined, 
these need to be connected, thus the divertor legs need to be drawn.
Firstly, the user has to input where outer leg and inner leg will meet in the 
private flux region (below the x-point), providing "xpt_height", the x-point vertical gap.

The divertor legs are drawn using "guide lines". 
Both outer leg and inner leg have an "internal guide line" and an "external guide line".
The internal guide lines have in common the starting point. This is the "middle point", 
which has same x coordinate of the x-point and z coordinate which is shifted from the 
x-point by the given "xpt_height".
The guide lines reach the relative target end, and close the divertor profile in the
private flux region (PFR) side.
The external guide lines start from relative divertor aperture side, and reach relative 
target end, closing the divertor profile at the scrape-off layer (SOL) side.

.. figure:: ../images/nova/pfr_sol.png
   :name: fig:pfr_sol
   :align: center

   Schematic of the last divertor building step. The red dot (x-point) and the blue dot 
   (middle point) are spaced by "xpt_height". 

The guide lines are not straight lines. They have a curvature that is extrapolated by 
interpolating either the separatrix or a specific flux line function.
The choice of the right curvature aims to not to intercept any flux line of the scrape-off 
layer before it reaches the divertor target.
The inner leg is commonly shorter, and the separatrix and the last flux line in the 
scrape-off layer have similar curvature. Thus the function that describes the separatrix is 
interpolated to draw both "internal guide line" and "external guide line".
The outer leg is commonly longer, especially in a long leg divertor configuration, 
and the separatrix and the last flux line in the scrape-off layer can have divergent curvature. 
Thus, in the latter case, the function that describes the separatrix is interpolated to draw 
the "internal guide line", and the function that describes the last flux line in the scrape-off 
layer is interpolated to draw the"external guide line".