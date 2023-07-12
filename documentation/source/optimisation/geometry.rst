Geometry Optimisation
=====================

Bluemira provides the function
:py:func:`~bluemira.optimisation._geometry.optimise_geometry`
to perform geometry optimisation on
:py:class:`~bluemira.geometry.parameterisations.GeometryParameterisation`\s.

Performing a Geometry Optimisation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As a simple example, say we have a circle,
and we want to minimise its perimeter whilst remaining outside some other wire.
First, we must define the
:py:class:`~bluemira.geometry.parameterisations.GeometryParameterisation`
representing our circle.

.. code-block:: python

    from typing import Dict, Optional, Union

    from bluemira.geometry.parameterisations import GeometryParameterisation
    from bluemira.utilities.opt_variables import OptVariables, BoundedVariable
    from bluemira.geometry.tools import make_circle
    from bluemira.geometry.wire import BluemiraWire


    class Circle(GeometryParameterisation):
        """Geometry parameterisation for a circle in the xz-plane."""

        def __init__(
            self, var_dict: Optional[Dict[str, Union[float, Dict[str, float]]]] = None
        ):
            opt_variables = OptVariables(
                [
                    BoundedVariable("radius", 10, 1e-5, 15),
                    BoundedVariable("centre_x", 0, -10, 10),
                    BoundedVariable("centre_z", 0, 0, 10),
                ]
            )
            opt_variables.adjust_variables(var_dict, strict_bounds=True)
            super().__init__(opt_variables)

        def create_shape(self, label: str = "") -> BluemiraWire:
            """Create the circle."""
            return make_circle(
                self.variables["radius"].value,
                center=(
                    self.variables["centre_x"].value,
                    0,
                    self.variables["centre_z"].value,
                ),
                axis=(0, 1, 0),
                label=label,
            )

.. note::
    For more information on
    :py:class:`~bluemira.geometry.parameterisations.GeometryParameterisation`\s
    see :doc:`here <../geometry/parameterisation>`.

Next step is to create our zone in which our circle must remain outside.
Let's make this is a square,
but note that it could be any closed wire.

.. code-block:: python

    from bluemira.display import plot_2d

    zone = make_polygon()
    zone.close()

    # Now lets create our circle within the shape
    circle = Circle(
        {"radius": {"value": 0.5}, "centre_x": {"value": -5}, "centre_z": {"value": 1}}
    )

    plot_2d([circle.create_shape(), zone])

.. image:: images/semi-circle.svg

As we are trying to maximise the perimeter of our circle,
the objective function will be the negative of the perimeter.

.. code-block:: python

    def objective(geom: Circle) -> float:
        """Objective function for maximising the perimeter of a circle."""
        return -geom.create_shape().length

The ``optimise_geometry`` Function
----------------------------------

Now we have everything we need,
we can use the
:py:func:`~bluemira.optimisation._geometry.optimise.optimise_geometry`
function to run the optimisation.

.. code-block:: python

    from bluemira.optimisation import optimise_geometry

    result = optimise_geometry(
        geom=circle,
        f_objective=objective,
        keep_out_zones=[zone],
        algorithm="SLSQP",
        opt_conditions={"ftol_rel": 1e-8},
    )
