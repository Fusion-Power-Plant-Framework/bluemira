First wall profile procedure
============================

This document refers to the ``FirstWall`` class in ``firstwall.py``.  

Overview
--------
Provided a plasma equilibrium, the FirstWall class allows to design 
a first wall profile and optimise it in order to reduce the heat flux values 
within prescribed limits. 
Currently the first wall profile can be made either for the SN or the DN configuration.

Procedure
---------
The procedure is going to be explained for the SN case, but it is meant to be dual, 
applicable to the SN and the DN, with the only difference that, in the second case,
all reasoning need to be made twice, for the LFS and the HFS.