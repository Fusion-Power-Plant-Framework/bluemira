Parameters and ParameterFrames
==============================

:py:class:`Parameter` and :py:class:`ParameterFrame` are the means by which the physical
configuration of a ``bluemira`` analysis are setup.

The Parameter class
-------------------

Parameters provide the value as well as a set of supporting metadata, including:

- Unit
- Short Description (Name)
- Long Description (Description)
- Source
- Mappings to external codes that are integrated with the bluemira analysis
- Historical values for the parameter value and the source of each value

.. code-block:: pycon

    >>> from bluemira.base import Parameter, ParameterMapping
    >>> r_0 = Parameter(
    ...     var="r_0",
    ...     name="Major Radius",
    ...     value="8.6",
    ...     unit="m",
    ...     description="The tokamak major radius",
    ...     source="Input",
    ...     mapping={"PROCESS": ParameterMapping("rmajor", read=True, write=False)},
    ... )
    >>> print(r_0)

The Parameter class uses a :py:class:`wrapt.ObjectProxy` to make all access to a
Parameter act as if it is the same type as the value of Parameter (except where
required).

.. code-block:: pycon

    >>> from bluemira.base import Parameter
    >>> p = Parameter(var='var', name='variable', value=5.0, source="Input")
    >>> print(p)
    var = 5.0 (variable)
    >>> isinstance(p, float)
    True
    >>> a = p + 5
    >>> print(a)
    10.0
    >>> p += 5
    >>> print(p)
    var = 10.0 (variable)

If the source is not provided for a Parameter, or provided after a value change, a
warning will be produced.

.. code-block:: pycon

    >>> from bluemira.base import Parameter
    >>> p = Parameter(var='var', name='variable', value=5.0, source="Input")
    >>> p += 5.0
    >>> p *= 2
    +-------------------------------------------------------------------------+
    | WARNING: The source of the value of var not consistently known          |
    +-------------------------------------------------------------------------+

This is resolved by ensuring that the source is always reset after changing a value.

.. code-block:: pycon

    >>> from bluemira.base import Parameter
    >>> p = Parameter(var='var', name='variable', value=5.0, source="Input")
    >>> p += 5.0
    >>> p.source = "After addition"
    >>> p *= 2
    >>> p.source = "After multiplication"

This allows the history of parameters to be consistently traced back through the
analysis.

.. code-block:: pycon

    >>> p.history()
    [(5.0, 'Input'), (10.0, 'After addition'), (20.0, 'After multiplication')]

If the value of a parameter is being reassigned then this needs to be performed directly
on the value attribute.

.. code-block:: pycon

    >>> from bluemira.base import Parameter
    >>> p = Parameter(var='var', name='variable', value=5.0, source="Input")
    >>> p.value = 6.0
    >>> p.source = "new val"
    >>> print(p)
    var = 6.0 (variable)

There are a few extra builtin methods to enable copying, array manipulation and pickling.

.. code-block:: pycon

    >>> import copy
    >>> from bluemira.base import Parameter
    >>> p = Parameter(var='var', name='variable', value=5.0, source="Input")
    >>> a = copy.deepcopy(p)
    >>> a.value = 2.0
    >>> a.source = "new val"
    >>> print(a)
    var = 2.0 (variable)
    >>> print(p)
    var = 5.0 (variable)

.. code-block:: pycon

    >>> import numpy as np
    >>> from bluemira.base import Parameter
    >>> p = Parameter(var='var', name='variable', value=np.array([1, 2, 3]), source="Input")
    >>> p *= 2
    >>> p.source = "multiplied"
    >>> print(p)
    var = [2 4 6] (variable)

.. code-block:: pycon

    >>> import pickle
    >>> from bluemira.base import Parameter
    >>> p = Parameter(var='var', name='variable', value=5.0, source="Input")
    >>> with open("param.pkl", "wb") as f:
    ...     pickle.dump(p, f)
    ...
    >>> with open("param.pkl", "rb") as f:
    ...     new_param = pickle.load(f)
    ...
    >>> print(id(p), id(new_param))

Idioms of the Parameter class
#############################

For very low types (eg `str`) it is not possible to modify how an object is treated:

.. code-block:: pycon

    >>> p = Parameter(var='var', name='var', value='hello')
    >>> print(p)
    hello
    >>> isinstance(p, str)
    True
    >>> repr(p)
    hello
    >>> str.__repr__(p)
    TypeError
    >>> p.join('world')
    helloworld
    >> ''.join(p, 'world')
    TypeError
    >>> ''.join(p.value, 'world')
    'helloworld'


This only affects some situations, the usual culprit is when leaving python for C. So far
this comes down to internal use of :py:func:`__repr__` for example
:py:func:`float.__repr__` or :py:func:`str.__repr__` for type checking. As a general rule
:py:func:`__repr__` shouldn't be used for type checking anyway but occasionally is
internally in python.

The ParameterFrame class
------------------------

The ParameterFrame class follows the 'borg' pattern where state is passed round (on request) but each instance is not the same (therefore not a singleton).
The default state of the frame is stored in :py:attr:`__default_params` and populated with the :py:meth:`set_default_parameters` classmethod.

In turn the default state can then populate :py:attr:`__dict__` (as a copy, but this could be in future be changed to a per reactor class variable).
To update the default Parameter values globally :py:meth:`_force_update_default` can be used which updates the Parameter in all ParameterFrame instances as well as the ParameterFrame class.

The attributes of a ParameterFrame are Parameter objects but the value of the Parameter can be accessed directly as a dictionary or with the `value` attribute as with a singular Parameter.

If a ParameterFrame.param is set to a 2 element tuple the second element is assumed to be its source if it is set to a Parameter (with the same name ONLY) the value and source are taken only.
A dictionary of :py:data:`{"value": .., "source":..}` can also be provided.:

.. code-block:: python

    pf = ParameterFrame(config)
    pf.attr = (1., 'here')
    pf.attr = Parameter(var='attr', name='attr', value=1.)
    pf.attr = {"value":1., "source": 'here'}

The concise json representation returns :py:data:`{"value": .., "source":..}` of each Parameter and the verbose representation returns all the attributes of the Parameter.
