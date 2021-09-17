Advection transport
===================

To estimate the heat fluxes due to charged particles travelling along open field lines,
``bluemira`` has a very simple advection transport model, in which a double exponential
decay law is used to model the particles in the scrape-off layer. It assumes fully
attached operation, and as such results for the divertor regions in particular should be
ignored if detached operation is expected. The model is predominantly intended to be
used to calculate the charged particle heat fluxes on the first wall.


Several input parameters are required to perform the analysis:

Two input objects are required to perform the analysis:

* an Equilibrium object, representing the equilibrium state of the plasma and the associated coils
* a geometry object, representing the first wall (i.e. all potentially flux-intercepting surfaces)