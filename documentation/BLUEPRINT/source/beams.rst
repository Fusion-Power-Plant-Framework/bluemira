beams
=====

This module is a simple 3-D beam finite element model, based on the
“Matrix Displacement Method” a.k.a. “Direct Stiffness Method”. It is
suitable for statically indeterminate structures. The notation used in
this module follows the Theory of Matrix Structural Analysis, (mostly
Chapter 6).

This module was written by Matti Coleman, but is based heavily on the
works of:

J. S. Przemieniecki: `Theory of Matrix Structural Analysis (1968) <https://s3.amazonaws.com/academia.edu.documents/44535182/45917260-Theory-of-Matrix-Structural-Analysis-1.pdf?response-content-disposition=inline;%20filename=Theory_of_Matrix_Structural_Analysis.pdf&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWOWYYGZ2Y53UL3A/20191021/us-east-1/s3/aws4_request&X-Amz-Date=20191021T123238Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=b6d1a533f21ca4eb57c4d6d99a23befd1acc7d506ebc05a704b6959288d31ab6>`_,
which actually made that structural matrix course I failed in university
look like it might make sense when using computers.

Simon McIntosh, who nearly made it with the “Structural Solver” (see
finite_element.py in Nova / CCFE-Nova) but moved to ITER before he could
seal the deal.

“Craig” a.k.a. JWock82 (`PyNite <https://github.com/JWock82/PyNite>`_) and Runar Tenfjord a.k.a. tenko
(`feapy <https://github.com/tenko/feapy>`_) whose modules were
useful examples, if not quite what I was looking for.

The author is greatly indebted to the above.

.. figure:: ../images/beams/beams_eiffel.png
   :name: eiffel

Overview
--------


The general idea is:

.. math:: \mathbf{U} = \mathbf{K^{-1}}(\mathbf{P}-\mathbf{Q})

where:

- :math:`\mathbf{U}` is the vector of displacements
- :math:`\mathbf{K}` is the global stiffness matrix
- :math:`\mathbf{P}` is the vector of nodal forces
- :math:`\mathbf{Q}` is the vector of thermal and distributed forces mapped to the nodes

The structure of the module aims to follow closely the flow diagram in
the Theory of Matrix Structural Analysis:

.. figure:: ../images/beams/flow_diagram_matrix_displacement.jpg
   :name: coordinates

Coordinates
-----------

BLUEPRINT global coordinates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Global x, y, z coordinates are used as in the rest of the code base:

.. figure:: ../images/coordinate_system.png
   :name: global_coordinates


Element local coordinates
~~~~~~~~~~~~~~~~~~~~~~~~~

Here, local x, y, z coordinates are as follows:

.. figure:: ../images/beams/beams_coordinate_system.jpg
   :name: local_coordinates


Note that the handed-ness of the coordinate system is the same as the
global one, and does not follow a conventional structural analysis
coordinate system for the directions of the y and z axes (which are
typically inverted, or named z and x, respectively).
