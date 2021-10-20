First wall profile procedure
============================

This document refers to the ``FirstWall`` class in ``firstwall.py``.  

Overview
--------
Provided a plasma equilibrium, the FirstWall class allows to design 
a first wall profile and optimise it, in order to reduce the heat flux values 
within prescribed limits. 
Currently, the first wall profile can be made either for the SN or the DN configuration.

The procedure is analogous for the SN configuration and the DN configuration.
For the DN configuration all reasoning are essentially doubled as, in the ideal and 
unlikely scenario, there is no charged particle flow between Low Field Side (LFS) 
and High Field Side (HFS), thus the power coming from the Outboard Mid-Plane (OMP) 
is responsible for the heat flux on the outer wall, hence its optimisation, 
and the power coming from the Inboard Mid-Plane (IMP) is responsible for the heat flux
on the inner wall, hence its optimisation.

Input
-----
- Equilibrium (e.g. .eqdsk, .geqdsk)
- Type of plasma (e.g. SN, DN)
- First wall geometrical offset, :math:`Delta_{fw}` 
  (Starting offset between plasma and wall. Either a single value 
  or two different values for inboard and outboard)
- Scrape-off layer power decay length, :math:`lambda_{q,nearSOL}` and 
  :math:`lambda_{q,farSOL}` (Either a single couple of valuesor two for inboard and outboard)
- Power crossing the SOL, :math:`P_{SOL,near}` and :math:`P_{SOL,far}`
  (Either a single couple of valuesor two for inboard and outboard)
- Hypothetical power sharing among targets

Output
------
- Heat flux onto the first wall

Procedure
---------
- Load equilibrium file
- Extract key attributes:
  - Last Closed Flux Surface (LCFS)
  - O-point coordinates
  - X-point(s) coordinates
  - Separatrix 
  
The preliminary first wall profile is drawn following some objects referred in the script 
as "guidelines". This guidelines are the flux lines chosen by the designer, according to 
the input :math:`Delta_{fw}` . Either the same offset or two different offset values can be
used at the inboard and outboard.
- SN: Offset the LCFS by :math:`Delta_{fw}` .
- DN: Get the flux lines passing through the points lying on the mid-plane (IMP and OMP), 
  and offsetted by :math:`Delta_{fw}` from the LCFS.
- Cut the obtained flux line(s) below the X-point and/or above the upper X-point

.. figure:: ../images/nova/preliminary_fw_profile.png
   :name: fig:preliminary_fw_profile
   :align: center

   Preliminary first wall profile with a different minimum plasma-wall clearance at inboard and
   outboard midplane.

- Make flux surfaces

The region between the LCFS and the preliminary profile is "filled" with a set of flux surfaces.
Thus, the SOL is discretised by flux surfaces (lines) spaced apart by a given dx. 

.. figure:: ../images/nova/fs_set.png
   :name: fig:fs_set
   :align: center

   SOL discretised by a finite number of flux surfaces

- Find intersections



