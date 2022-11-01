Designers and Builders
----------------------

Designers and Builders are the main structure in ``bluemira`` for creating a Component.

The two main classes are:

* :py:class:`bluemira.base.designer.Designer`: the Designer base class
* :py:class:`bluemira.base.builder.Builder`: the Builder base class

Designers
^^^^^^^^^

`Designers` solve the minimal design problem of a `Component`. The minimal design problem could
result in, for instance, a geometric wire or some relevant values that allow the `Builder` to build
the `Component` CAD. A Designer is optional as some `Components` can be completely built with
user input and calculations from previous component design stages.
Optimisation problems should be run within a `Designer`.

A basic `Designer` only requires a `run` method and a `param_cls` attribute to store the
`ParameterFrame` class reference. Once initialised the `Designer` is run with its `execute` method.
The below is for illustrative purposes and is overkill, in this instance you would just have a `Builder`.

.. code-block:: python

    from dataclasses import dataclass

    from bluemira.base.designer import Designer
    from bluemira.base.parameter_frame import Parameter, ParameterFrame

    @dataclass
    class DesignerPF(ParameterFrame):
        A: Parameter[float]


    class MyDesigner(Designer)

        param_cls = DesignerPF

        def run(self) -> float:
            return  self.params.A.value


To initialise a `Designer` you need any `ParameterFrame` instance that is a superset of `param_cls`
and optionally a `build_config` dictionary which contains configuration options for the `Designer`.
It is possible to execute a `Designer` in different ways depending on requirements or software
availability. If another method such as `mock` or `read` is defined and `run_mode` is specified in
the `build_config` the `execute` method will call the specified method.

.. code-block:: python

    build_config = {"run_mode": "mock"}

    class MyOtherDesigner(Designer):

        param_cls = DesignerPF

        def run(self) -> float:
            return self.params.A.value

        def mock(self) -> float:
            return self.params.A.value ** 2

Builders
^^^^^^^^

The minimal design problem output if required along with user input to the `Builder` is all
the information needed to build the CAD for the `Component`.
The `build` method of the `Builder` usually builds the xz, xy and xyz views of a `Component`,
creates the component tree and sets up the `ComponentManager` although what the build method does
is up to the Reactor Designer.

Like a `Designer` a `Builder` requires a `param_cls` attribute and is initialised with a `ParameterFrame`
instance that is a superset of `param_cls` and optionally the `Designer.execute()` output and a
`build_config` dictionary.
The below is an example of a standard `Builder` structure and once initialised the `build` method is
called to create the `Component`.

.. code-block:: python

    from dataclasses import dataclass

    from bluemira.base.builder import Builder, ComponentManager
    from bluemira.base.parameter_frame import Parameter, ParameterFrame

    @dataclass
    class BuilderPF(ParameterFrame):
        R_0: Parameter[float]

    class MyComponentManager(ComponentManager):
        """Blank Component Manager"""


    class MyBuilder(Builder):

        param_cls = BuilderPF

        def build(self) -> MyComponentManager
            return MyComponentManager(
                self.component_tree(
                    xz=[self.build_xz()],
                    xy=[self.build_xy()],
                    xyz=[self.build_xyz()],
                )
            )

        def build_xz(self):
            # return A 2D geometry PhysicalComponent

        def build_xy(self):
            # return A 2D geometry PhysicalComponent

        def build_xyz(self):
            # return A 3D geometry PhysicalComponent
