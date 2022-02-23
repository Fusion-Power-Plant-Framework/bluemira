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
    ...     mapping={"PROCESS": ParameterMapping("rmajor", recv=True, send=False)},
    ... )
    >>> print(r_0)

The Parameter class uses a :py:class:`wrapt.ObjectProxy` to make all access to a
Parameter act as if it is the same type as the value of Parameter (except where
required).

.. code-block:: pycon

    >>> from bluemira.base import Parameter
    >>> p = Parameter(var="var", name="variable", value=5.0, source="Input")
    >>> print(p)  # var = 5.0 (variable)
    >>> isinstance(p, float)  # True
    >>> a = p + 5
    >>> print(a)  # 10.0
    >>> p += 5
    >>> print(p)  # var = 10.0 (variable)

If the source is not provided for a Parameter, or provided after a value change, a
warning will be produced.

.. code-block:: pycon

    >>> from bluemira.base import Parameter
    >>> p = Parameter(var="var", name="variable", value=5.0, source="Input")
    >>> p += 5.0
    >>> p *= 2
    +-------------------------------------------------------------------------+
    | WARNING: The source of the value of var not consistently known          |
    +-------------------------------------------------------------------------+

This is resolved by ensuring that the source is always reset after changing a value.

.. code-block:: pycon

    >>> from bluemira.base import Parameter
    >>> p = Parameter(var="var", name="variable", value=5.0, source="Input")
    >>> p += 5.0
    >>> p.source = "After addition"
    >>> p *= 2
    >>> p.source = "After multiplication"

This allows the history of parameters to be consistently traced back through the
analysis.

.. code-block:: pycon

    >>> p.history()
    [(5.0, "Input"), (10.0, "After addition"), (20.0, "After multiplication")]

If the value of a parameter is being reassigned then this needs to be performed directly
on the value attribute.

.. code-block:: pycon

    >>> from bluemira.base import Parameter
    >>> p = Parameter(var="var", name="variable", value=5.0, source="Input")
    >>> p.value = 6.0
    >>> p.source = "new val"
    >>> print(p)  # var = 6.0 (variable)

There are a few extra builtin methods to enable copying, array manipulation and pickling.

.. code-block:: pycon

    >>> import copy
    >>> from bluemira.base import Parameter
    >>> p = Parameter(var="var", name="variable", value=5.0, source="Input")
    >>> a = copy.deepcopy(p)
    >>> a.value = 2.0
    >>> a.source = "new val"
    >>> print(a)  # var = 2.0 (variable)
    >>> print(p)  # var = 5.0 (variable)

.. code-block:: pycon

    >>> import numpy as np
    >>> from bluemira.base import Parameter
    >>> p = Parameter(var="var", name="variable", value=np.array([1, 2, 3]), source="Input")
    >>> p *= 2
    >>> p.source = "multiplied"
    >>> print(p)  # var = [2 4 6] (variable)

.. code-block:: pycon

    >>> import pickle
    >>> from bluemira.base import Parameter
    >>> p = Parameter(var="var", name="variable", value=5.0, source="Input")
    >>> with open("param.pkl", "wb") as f:
    ...     pickle.dump(p, f)
    ...
    >>> with open("param.pkl", "rb") as f:
    ...     new_param = pickle.load(f)
    ...
    >>> print(id(p), id(new_param))

Idioms of the Parameter class
#############################

For very low types (eg :py:class:`str`) it is not possible to modify how an object is treated:

.. code-block:: pycon

    >>> p = Parameter(var="var", name="var", value="hello")
    >>> p  # 'hello'
    >>> print(p)  # 'var = hello (var)'
    >>> isinstance(p, str)  # True
    >>> repr(p)  # "'hello'"
    >>> str.__repr__(p)  # TypeError
    >>> "".join([p, "world"])  # TypeError
    >>> "".join([p.value, "world"])  # 'helloworld'


This only affects some situations, the usual culprit is when leaving python for C. So far
this comes down to internal use of :py:func:`__repr__` for example
:py:func:`float.__repr__` or :py:func:`str.__repr__` for type checking. As a general rule
:py:func:`__repr__` shouldn't be used for type checking anyway but occasionally is
internally in python.

Another situation is when passing parameters into a low level library such as a function wrapped with Numba. At this time it may work but behaviour in a function wrapped with Numba has had some odd side effects. A solution could be using Numba's low level api however we are awaiting Numba v1 before implementation.

In all of the above cases you can use the :py:attr:`value` attribute to access the raw value of the parameter.

The ParameterMapping class
--------------------------

ParameterMapping is used to create a connection between ``bluemira`` parameters and parameters on any external program. At its most basic level it is a key-value mapping between two variable names. On top of the mapping, how the parameter value flows between ``bluemira`` and the external program is modified by the :py:attr:`send` and :py:attr:`recv` attributes.

:py:attr:`send`
    true - set bluemira parameter value as input to external code

    false - use default value as input to external code

:py:attr:`recv`
    true - set external code result to the new value of the bluemira parameter

    false - keep the original bluemira parameter value ignoring the external value

The ParameterFrame class
------------------------

ParameterFrames allow Parameters to be grouped together to describe the overall
parameterisation of a particular analysis or other class within ``bluemira``. For this
reason you will interact with Parameters via ParameterFrames in most cases.

A ParameterFrame can be constructed either from a list of records (with each matching
the constructor argument order for Parameter), a dictionary, or a json file.
ParameterFrames can be converted to json in either a verbose format, including all the
attributes on the Parameter, or in a concise format, just mapping Parameters to their
value and source. This allows template ParameterFrames to be created using the verbose
form and then adjusted for a specific analysis via the concise form.

.. code-block:: pycon

    >>> from bluemira.base import ParameterFrame, ParameterMapping
    >>> record_list = [
    ...     ["R_0", "Major radius", 9, "m", None, "Input", {"PROCESS": ParameterMapping("rmajor", True, False)}],
    ...     ["A", "Plasma aspect ratio", 3.1, "dimensionless", None, "Input", {"PROCESS": ParameterMapping("aspect", True, True)}],
    ... ]
    >>> params = ParameterFrame(record_list)
    >>> print(params)

.. code-block:: pycon

    >>> from bluemira.base import ParameterFrame, ParameterMapping
    >>> param_dict = {
    ...     "R_0": {
    ...         "name": "Major radius",
    ...         "value": 9,
    ...         "unit": "m",
    ...         "description": None,
    ...         "source": "Input",
    ...         "mapping": {
    ...             "PROCESS": ParameterMapping("rmajor", True, False)
    ...         }
    ...     },
    ...     "A": {
    ...         "name": "Plasma aspect ratio",
    ...         "value": 3.1,
    ...         "unit": "dimensionless",
    ...         "description": None,
    ...         "source": "Input",
    ...         "mapping": {
    ...             "PROCESS": ParameterMapping("aspect", True, True)
    ...         }
    ...     },
    ... }
    >>> params = ParameterFrame.from_dict(param_dict)
    >>> params.to_json("params_verbose.json", verbose=True)
    >>> params_new = ParameterFrame.from_json("params_verbose.json")
    >>> print(params)
    >>> print(params_new)

The attributes of a ParameterFrame are Parameter objects, and so the attributes on the
Parameter can be accessed directly. It is also possible to access the values of
Parameters can be as if the ParameterFrame were a dictionary.

.. code-block:: pycon

    >>> print(params_new["R_0"])
    >>> print(params_new.R_0)
    >>> print(params_new.R_0.source)
    >>> params_new.R_0 = 8.6
    >>> params_new.R_0.source = "Update"
    >>> params_new.to_json("params_concise.json")

If a ParameterFrame.param is set to a 2 element tuple the second element is assumed to be
its source if it is set to a Parameter (with the same name ONLY) the value and source are
taken only.
A dictionary of :py:data:`{"value": .., "source":..}` can also be provided.:

.. code-block:: pycon

    >>> from bluemira.base import Parameter
    >>> params_new.R_0 = (9.2, "Here")
    >>> params_new.A = Parameter(var="A", name="Plasma aspect ratio", value=3.2, source="There")
    >>> params_new.A = {
    >>>     "value": 3.1,
    >>>     "source": "Here",
    >>> }
    >>> print(params_new)
    >>> print(params_new.R_0.history(), params_new.A.history())
    >>> params_new.to_json("params_concise.json")


As an analysis progresses, various values within the ParameterFrame will be updated from
different sources. This is handled in bulk by updating Parameters based on their keyword
name, which can be done either directly or via an external json source.

.. note::

    Keywords must match the current Parameters contained within the ParameterFrame in
    order to update the corresponding value.

.. code-block:: pycon

    >>> params.update_kw_parameters({"R_0": 9.3}, source="New Value")
    >>> print(params)
    >>> print(params.R_0.history())
    >>> params.set_values_from_json("params_concise.json", source="Load Data")
    >>> print(params)
    >>> print(params.R_0.history(), params.A.history())

Handling Default Parameters
###########################

As noted in the previous section, ParameterFrames require some knowledge of the available
Parameters to be used, otherwise keyword names will deviate between different analysis
stages. This is facilitated by storing a default set of available parameters on the
ParameterFrame class for a particular analysis.

The ParameterFrame class follows the 'borg' pattern where state is passed round (on
request) but each instance is not the same (therefore not a singleton).
The default state of the frame is stored in :py:attr:`__default_params` and populated
with the :py:meth:`set_default_parameters` classmethod.

In turn the default state can then populate :py:attr:`__dict__` (as a copy, but this
could be in future be changed to a per analysis class variable).
To update the default Parameter values globally :py:meth:`_force_update_default` can be
used which updates the Parameter in all ParameterFrame instances as well as the
ParameterFrame class.
