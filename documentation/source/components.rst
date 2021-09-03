components
==========

The ``components`` module defines the fundamental building blocks of a bluemira design
scenario. The module defines various types of ``Component`` classes that can be used
to represent different parts of a design:

- :py:class:`bluemira.components.base.Component` is the base class and defines the
  framework on which other components can be defined within bluemira. This class must not
  not be used directly, rather one of the below subclasses should be used to represent
  parts of the design.
- :py:class:`bluemira.components.base.PhysicalComponent` defines the physical parts of a
  reactor, such as blanket layers, vessel shells, or ports. As implementations of
  :py:class:`bluemira.components.base.PhysicalComponent` correspond to a physical object,
  instances of that class can be defined with a shape and a material.
- :py:class:`bluemira.components.base.MagneticComponent` defines the magnetic parts of a
  reactor, such as poloidal or toroidal field coils. These have a shape and material and
  additionally define a conductor to provide the current-carrying filament.
- :py:class:`bluemira.components.base.GroupingComponent` allows other types of component
  to be grouped together into super-structures. For example, a blanket first wall,
  breeding zone, manifold, and back supporting structure may be grouped into a blanket
  system. As these objects are not physical, they do not have a corresponding shape or
  material composition.
