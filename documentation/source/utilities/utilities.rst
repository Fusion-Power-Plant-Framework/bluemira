utilities
=========

Example of toroidal coordinate transform
----------------------------------------

Demonstrate  the conversion between cylindrical and toroidal coordinate systems
using the Bluemira functions `cylindrical_to_toroidal` and `toroidal_to_cylindrical`. We denote toroidal coordinates by (:math:`\tau`, :math:`\sigma`, :math:`\phi`) and cylindrical coordinates by (:math:`R`, :math:`z`, :math:`\phi`).

Imports
    .. code-block:: python

        import matplotlib.pyplot as plt
        import numpy as np

        from bluemira.utilities.tools import (
            cylindrical_to_toroidal,
            toroidal_to_cylindrical,
        )

.. figure:: images/toroidal-coordinates-diagram-wolfram.png
    :name: fig:toroidal-coordinates-diagram-wolfram

This diagram is taken from
`Wolfram MathWorld <https://mathworld.wolfram.com/ToroidalCoordinates.html>`_ and shows a
toroidal coordinate system. It uses (:math:`u`, :math:`v`, :math:`\phi`) whereas we use (:math:`\tau`, :math:`\sigma`,
:math:`\phi`).

In toroidal coordinates, surfaces of constant :math:`\tau` are non-intersecting tori of
different radii, and surfaces of constant :math:`\sigma` are non-concentric spheres of
different radii which intersect the focal ring.


We are working in the poloidal plane, so we set :math:`\phi = 0`, and so are looking at a
bipolar coordinate system. We are transforming about a focus :math:`(R_0, z_0)` in the
poloidal plane.

Here, curves of constant :math:`\tau` are non-intersecting circles of different radii that
surround the focus and curves of constant :math:`\sigma` are non-concentric circles
which intersect at the focus.

To transform from toroidal coordinates to cylindrical coordinates about the focus in
the poloidal plant :math:`(R_0, z_0)`, we have the following relations:

.. math::
    R = R_0 \frac{\sinh\tau}{\cosh\tau - \cos\sigma}\\
    z - z_0 = R_0 \frac{\sin\tau}{\cosh\tau - \cos\sigma}

where we have :math:`0 \le \tau < \infty` and :math:`-\pi < \sigma \le \pi`.

The inverse transformations are given by:

.. math::
    \tau = \ln \frac{d_1}{d_2}

.. math::
    \sigma = \text{sign}(z - z_0) \arccos \frac{d_1^2 + d_2^2 - 4 R_0^2}{2 d_1 d_2}

where we have

.. math::
    d_1^2 = (R + R_0)^2 + (z - z_0)^2\\
    d_2^2 = (R - R_0)^2 + (z - z_0)^2

Converting a unit circle
------------------------
We will start with an example of converting a unit circle in cylindrical coordinates to
toroidal coordinates and then converting back to cylindrical using the Bluemira functions `cylindrical_to_toroidal` and `toroidal_to_cylindrical`.
This unit circle is centered at the point (2,0) in the poloidal plane.

Original circle:

.. figure:: images/original-unit-circle-example.png
    :name: fig:original-unit-circle

Convert this circle to toroidal coordinates:

.. figure:: images/unit-circle-converted-toroidal.png
    :name: fig:unit-circle-converted-toroidal

Convert this back to cylindrical coordinates to recover the original unit circle centered at (2,0) in the poloidal plane:

.. figure:: images/unit-circle-back-to-cylindrical.png
    :name: fig:unit-circle-converted-back-cylindrical

Curves of constant :math:`\tau` and :math:`\sigma`
--------------------------------------------------
When plotting in cylindrical coordinates, curves of constant :math:`\tau` correspond to
non-intersecting circles that surround the focus :math:`(R_0, z_0)`, and curves of constant
:math:`\sigma` correspond to non-concentric circles that intersect at the focus.

1. Curves of constant :math:`\tau` plotted in both cylindrical and toroidal coordinates

Set the focus point to be :math:`(R_0, z_0) = (1,0)`. We plot 6 curves of constant :math:`\tau` in cylindrical coordinates

.. figure:: images/constant-tau-cylindrical.png
    :name: fig:constant-tau-cylindrical

Now convert to toroidal coordinates using `cylindrical_to_toroidal` and plot - here curves of constant :math:`\tau` are straight lines

.. figure:: images/constant-tau-toroidal.png
    :name: fig:constant-tau-toroidal

2. Curves of constant :math:`\sigma` plotted in both cylindrical and toroidal coordinates:

Set the focus point to be :math:`(R_0, z_0) = (1,0)`. We plot 6 curves of constant :math:`\sigma` in cylindrical coordinates

.. figure:: images/constant-sigma-cylindrical.png
    :name: fig:constant-sigma-cylindrical

Now convert to toroidal coordinates using `cylindrical_to_toroidal` and plot - here curves of constant :math:`\sigma` are straight lines

.. figure:: images/constant-sigma-toroidal.png
    :name: fig:constant-sigma-toroidal
