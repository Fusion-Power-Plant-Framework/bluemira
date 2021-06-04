======
MIRApy
======

.. warning::

    This is the documentation of **MIRApy**, the python implementation of MIRA.
    MIRA has been developed in MATLAB and its python "conversion" is still
    at a preliminary stage.
    
    The actual package implements:
        * geometrical class
        * meshing class with gmsh
        * fem interface through fenics/dolfin
        * electromagnetic class (Biot-Savart and Green's function)
        * Grad-Schafranov fixed boundary plasma solution
    
    Still not completed:
        * equilibrium solver: missing implementation of a complete set of constraints (update class equilibrium.py)
        * TF/PF coils optimization:
            * conversion to the new architecture of PF_TF_optimization_procedure.py
        * 3D shape:
            * conversion of old Test3D.py
    
    Not implemented:
        - full pulse implementation    
        - material implementation
        - neutronic class
        - Power cycle
        - ...

.. note::
    Check the code according to PEP8 rules

Contents
========

.. toctree::
   :maxdepth: 2
   
   Introduction <introduction>

   Installation <install>
   freecad

   geometry
   meshing
      
   Module Reference <autoapi/index>
   
Testing and Code coverage
=========================

https://developer.ibm.com/recipes/tutorials/testing-and-code-coverage-with-python/

References
==========
.. [Franza] Development and Validation of a Computational Tool for Fusion 
    Reactors System Analysis". Ph.D. thesis, Karlsruhe Institute of Technology
    (KIT), June 2019. DOI: 10.5445/IR/1000095873,
    https://publikationen.bibliothek.kit.edu/1000095873

.. [Process] M. Kovari, R. Kemp, H. Lux, P. Knight et al. PROCESS: A systems 
    code for fusion power plants - Part 1: Physics. Fusion Engineering and 
    Design, 89:3054–3069, 2014.