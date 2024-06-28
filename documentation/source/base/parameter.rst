Parameters and ParameterFrames
------------------------------

:py:class:`Parameter` and :py:class:`ParameterFrame` are the means by which the physical
configuration of a ``bluemira`` analysis are setup.

Parameters
^^^^^^^^^^

A :py:class:`Parameter` object wraps a value and holds metadata relating to that value.
To create a :py:class:`Parameter`, as a minimum, the name, the value and a unit need to be provided
but other information can be added:

    name
        The shorthand name of the :py:class:`Parameter` used for access when used as part of a :py:class:`ParameterFrame`.
    value
        The value of the :py:class:`Parameter`.
    unit
        The unit associated with the value.
    source
        The origin of a :py:class:`Parameter`, for instance the module it was set from.
    description
        A description of the :py:class:`Parameter`. This is a good place for references and other useful information.
    long_name
        A longer more descriptive name for instance 'A' could be the :py:class:`Parameter` name but 'Aspect Ratio' is more helpful for a user.

.. code-block:: pycon

    >>> from bluemira.base.parameter_frame import Parameter
    >>> r_0 = Parameter(
    ...     name="R_0",
    ...     value=5000,
    ...     unit="cm",
    ...     source="Input",
    ...     description="Tokamak major radius",
    ...     long_name="Major Radius",
    ... )
    >>> print(r_0)
    <Parameter(R_0=5000 cm)>

Only the value of the :py:class:`Parameter` can be directly updated after initialisation.
However if you want to change the source the :py:meth:`set_value` method can be used.

.. code-block:: pycon

   >>> r_0.value = 6000
   >>> print(r_0, r_0.source or None)
   <Parameter(R_0=6000 cm)> None
   >>> r_0.set_value(20, "New Input")
   >>> print(r_0, r_0.source or None)
   <Parameter(R_0=20 cm)> New Input

If you want to access the value of the :py:class:`Parameter` in a different unit,
the :py:meth:`value_as` method can be used.

.. code-block:: pycon

    >>> r_0.value_as('m')
    0.2

Any update to a :py:class:`Parameter` value is stored and can be accessed with the :py:meth:`history` method
which can be useful to understand why a :py:class:`Parameter` value changed.

.. code-block:: pycon

    >>> r_0.history()
    [ParameterValue(value=5000, source='Input'),
     ParameterValue(value=6000, source=''),
     ParameterValue(value=20, source='New Input')]


ParameterFrames
^^^^^^^^^^^^^^^

A :py:class:`ParameterFrame` allows Parameters to be grouped together to describe the overall
parameterisation of a particular analysis or class within ``bluemira``.
For this reason you will interact with Parameters via a :py:class:`ParameterFrame` in most cases.

A ParameterFrame is written as a dataclass:

.. code-block:: python

    from dataclasses import dataclass
    from bluemira.base.parameter_frame import Parameter, ParameterFrame

    @dataclass
    class MyParameterFrame(ParameterFrame):
        R_0: Parameter[float]
        A: Parameter[float]

The type of each :py:class:`Parameter` must be specified and adhered to in the initialisation of the :py:class:`ParameterFrame`.
A :py:class:`ParameterFrame` can be initialised from a dictionary,
a json file or another :py:class:`ParameterFrame` (must be a superset of the :py:class:`ParameterFrame` being initialised).

.. code-block:: python

    param_dict = {
        "R_0": {
            "value": 9,
            "unit": "m",
            "source": "Input",
        },
        "A": {
            "value": 3.1,
            "unit": "dimensionless",
            "source": "Input",
        },
    }
    params = MyParameterFrame.from_dict(param_dict)
    param_2 = MyParameterFrame.from_frame(params)

Units
"""""
:py:class:`ParameterFrames` always enforce the same set of standard units :ref:`unit_convention`.
:py:class:`Parameters` within a :py:class:`ParameterFrame` whose units are convertible to one of bluemira's standard units,
have their values and converted to the corresponding standard unit.
This keeps the units used within Bluemira consistent across classes and modules.

For this reason, if your inputs use a non-standard unit,
the value you put into a :py:class:`Parameter` will be different to the one you get out.

.. code-block:: pycon

    >>> param_dict = {
    ...     "R_0": {
    ...         "value": 9,
    ...         "unit": "cm",
    ...         "source": "Input",
    ...     },
    ...     "A": {
    ...         "value": 3.1,
    ...         "unit": "dimensionless",
    ...         "source": "Input",
    ...     },
    ... }
    >>> print(MyParameterFrame.from_dict(param_dict))
    ╒════════╤═════════╤═══════════════╤══════════╤═══════════════╤═════════════╕
    │ name   │   value │ unit          │ source   │ description   │ long_name   │
    ╞════════╪═════════╪═══════════════╪══════════╪═══════════════╪═════════════╡
    │ A      │     3.1 │ dimensionless │ Input    │ N/A           │ N/A         │
    ├────────┼─────────┼───────────────┼──────────┼───────────────┼─────────────┤
    │ R_0    │    0.09 │ m             │ Input    │ N/A           │ N/A         │
    ╘════════╧═════════╧═══════════════╧══════════╧═══════════════╧═════════════╛

Use :py:meth:`Parameter.value_as` to return parameter values in a non-standard unit.
Input values with units listed in :ref:`unit_convention` are not modified.
