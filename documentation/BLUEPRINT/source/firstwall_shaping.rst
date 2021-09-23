First wall profile procedure
============================

This document refers to the ``FirstWall`` class in ``firstwall.py``.  

Overview
--------
Provided a plasma equilibrium, the FirstWall class allows to design 
a first wall profile and optimise it in order to reduce the heat flux values 
within prescribed limits. 
Currently, the first wall profile can be made either for the SN or the DN configuration.

The procedure is going to be explained for the SN case, but it is meant to be dual, 
applicable to the SN and the DN, with the only difference that, in the second case,
all reasoning need to be made twice, for the LFS and the HFS.

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
  
The preliminary first wall profile is drawn following guidelines. This guidelines
are the flux lines chosen by the designer, according to the input :math:`Delta_{fw}` .
Either the same offset or two different values can be used at the inboard and outboard.
- SN: Offset the LCFS by :math:`Delta_{fw}`
- DN: Get the flux line passing through the point lying on the mid-plane, 
  and offsetted by :math:`Delta_{fw}` from the LCFS
- Cut the obtained flux line(s) below the X-point and/or above the upper X-point

 
















Heat Flux Calculation
=====================

This document refers to the ``FluxSurface`` class in ``firstwall.py``.  

Overview
--------
The FluxSurface class is used to pick a flux surface from an equilibrium, and
give it all the attributes to calculate the carried power and, once evoked in 
the FirstWall class, dicretise the SOL and calculate the heat flux onto the 
first wall profile.

Heat Flux Model
---------------
In a tokamak, the plasma is confined on nested closed flux surfaces.  
Of these closed surfaces, the outermost one is called the 
Last Closed Flux Surface (LCFS).
All the flux surfaces inside the LCFS are closed. 
The plasma flows mainly along the field lines lying on the flux surfaces. 
All flux surfaces outside the LCFS intersect the tokamak wall and relevant 
field lines are open. 
The region outside the LCFS is called the Scrape-Off Layer (SOL).
Collisional and turbulent processes lead the plasma, confined in the core, 
to diffuse and outflow into the SOL.

Perpendicular and parallel transport in the SOL results in the exponential 
decay of plasma density and temperature moving away from the LCFS in the 
radial direction.
The exhaust power (:math:`P_{SOL}`) is assumed to enter the SOL at the Outboard
Mid-Plane (OMP - the subscript “u” is used for this location, meaning “upstream”) 
and it separates into two flows, one towards the inner divertor, another to the 
outer divertor.

.. figure:: ../images/nova/SOL_power_sharing.png
   :name: fig:SOL_power_sharing
   :align: center

   Schematic of the model for the SOL power sharing between inner and outer divertors. Illustrated, as an example, a LFS Snowflake Minus divertor.

The heat flux along the field lines in the SOL is usually assumed to decay 
exponentially with the distance from the LCFS at the OMP, :math:`r_u`:

.. math::
   
   q_{\parallel}(r_u) = q_{\parallel,0}e^{-r_u/\lambda_q}
   
Where :math:`q_{\parallel,0}` is the flux at the separatrix, and :math:`\lambda_q` 
is the heatflux decay length in the SOL.

To be more precise, the SOL exhibits two different regions [Nespoli_2017]_:

- A “near” SOL, extending a few mm from the LCFS, characterized by a steep profile of :math:`q_{\parallel}`
  and responsible for the peak heat loads in the divertor region;
- A “far” SOL, typically some cm wide, with a flatter profile of :math:`q_{\parallel}`
  and responsible for most of the heat deposited onto the first wall.

.. figure:: ../images/nova/hf_radial_profile.png
   :scale: 50 %
   :name: fig:hf_radial_profile
   :align: center

   Parallel heat flux radial profile in JET.

The parallel heat flux radial profile :math:`q_{\parallel}` is then better described by a sum of two 
exponentials, associated with the two different regions:

.. math::
   
   q_{\parallel}(r_u) = q_{n}e^{-r_u/\lambda_n} + q_{f}e^{-r_u/\lambda_f}

Where :math:`\lambda_n` and :math:`\lambda_f` are the near and far SOL decay lengths and :math:`q_n` and :math:`q_f` 
are the associated heat flux magnitudes.

According to the above expression, the code calculates the radial profile of the 
poloidal component of the heat flux at the OMP, assuming :math:`P_{SOL}` distributed 
between near and far scrape off layer:

.. math::
   
   q_{p,u}(r_u) = \dfrac{P_{SOL,n}e^{-r_u/\lambda_n}}{2 \pi R(r_u)\lambda_n} + \dfrac{P_{SOL,f}e^{-r_u/\lambda_f}}{2 \pi R(r_u)\lambda_f}

At the OMP, the heat flux parallel to the magnetic field :math:`q_{\parallel,u}` and that parallel to 
the poloidal component of the field :math:`q_{p,u}` are related by :math:`q_{\parallel,u} = q_{p,u}(B_{tot,u}/B_{p,u)}`.

To compute the heat flux at the target location, one must consider that each poloidal 
flux surface has a “width”, evaluated at the outboard mid-plane and indicated here as :math:`dr_u` [Maurizio_2020]_.

.. figure:: ../images/nova/flux_expansion.png
   :scale: 50 %
   :name: fig:flux_expansion
   :align: center

   Description of the SOL scalar coordinate :math:`dr_{u}` , defined at the outboard mid-plane, 
   and its relation to the SOL scalar coordinate dx, defined at the divertor target.


Such flux surface width varies when moving poloidally around the confined plasma or along 
the divertor leg. The ratio of the width at the target and at the OMP
is called target poloidal flux expansion.

.. math::
   
   f_{x,t} = \dfrac{dr_t}{dr_u} = \dfrac{R_{u}B_{p,u}}{R_{t}B_{p,t}}

Where :math:`R_u` and :math:`B_{p,u}` are major radius and poloidal magnetic field at the outboard midplane, 
and :math:`R_t` and :math:`B_{p,t}` are major radius and poloidal magnetic field at the target.

Since the power entering a flux tube at the OMP location is equal to the power that exits 
the same flux tube at the target, :math:`2\pi R_{u} dr_{u} q_{p,u} = 2\pi R_{t} dr_{u} f_{x,t} q_{p,t}` 
the poloidal heat flux component at the target can be calculated as:

.. math::

   q_{p,t} = q_{p,u}\frac{R_u}{R_t}\frac{1}{f_{x,t}}

From the poloidal component, at the target, the perpendicular heat flux component is calculated 
considering the angle between flux surface and target surface:

.. math::

   q_{\perp,𝑡} = q_{p,t}sin\beta_t

.. rubric:: References

.. [Nespoli_2017] NESPOLI, Federico. Scrape-Off Layer physics in limited plasmas in TCV. s.l.: EPFL, 2017

.. [Maurizio_2020] MAURIZIO, Roberto. Investigating Scrape-Off Layer transport in alternative divertor geometries on the TCV tokamak. s.l.: EPFL, 2020.



