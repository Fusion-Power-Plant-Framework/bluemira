Optimisation
============

Bluemira provides a function to perform optimisations
with non-linear constraints.
It also provides a function to perform optimisations on geometry.

Available Optimisation Algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are several optimisation algorithms that can be used within Bluemira.
Including gradient and non-gradient based.

TODO: list them out.

See the :py:class:`~bluemira.optimisation._algorithm.Algorithm`
enum for a full list.

Performing an Optimisation
^^^^^^^^^^^^^^^^^^^^^^^^^^

Consider the following example to demonstrate how to set up
and perform an optimisation with some non-linear constraints.

Suppose that we wish to find

.. math::

    \min_{x \in \mathbb{R}^2} \sqrt{x_2} \tag{1}

subject to constraints

.. math::

    x_2 \ge 0 \tag{2}

.. math::

    x_2 \ge (a_1x_1 + b_1)^3 \tag{3}

.. math::

    x_2 \ge (a_2 x_1 + b_2)^3 \tag{4}

for parameters
:math:`a_1 = 2`, :math:`b_1 = 0`, :math:`a_2 = -1`, :math:`b_2 = 1`.

This example problem is ripped straight from the
`NLOpt docs <https://nlopt.readthedocs.io/en/latest/NLopt_Tutorial/#example-nonlinearly-constrained-problem>`_.

This problem expects a minimum at :math:`x = ( \frac{1}{3}, \frac{8}{27} )`.

First we must define our objective function.
The objective function must take a single argument: a 1-D ``numpy`` array,
where each element in the array is an optimisation parameter.
The objective function must return a float.

.. code-block:: python

    def f_objective(x: np.ndarray) -> float:
        """Objective function for optimisation to find solution to eqn (1)."""
        return np.sqrt(x[1])

If using a gradient-based optimisation algorithm,
it helps to define an analytical gradient.
If you do not give an analytical gradient,
a numerical one will be estimated.

As with the objective, the gradient function must take a 1-D ``numpy`` array,
containing the optimisation parameters, as its only argument.
It must return a ``numpy`` array of the same length,
where each index :math:`i` contains the partial derivative
:math:`\frac{\partial f}{\partial x_i}`
for the corresponding optimisation parameter.

.. code-block:: python

    def df_objective(x):
        """Gradient of the objective function."""
        return np.array([0.0, 0.5 / np.sqrt(x[1])])

Now we need to define our constraints.
Again, a constraint function must take a single argument,
the array of optimisation parameters.

TODO: talk about form of constraint i.e., le 0 and how to rewrite
the above constraints to match that form.


Notice that constraints 3 and 4 are of a similar form,
so we can write the constraint once,
and pass in the different ``a`` and ``b`` values.

.. code-block:: python
