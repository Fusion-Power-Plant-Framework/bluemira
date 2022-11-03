Components
==========

``Components`` are the fundamental building blocks of a bluemira design. Various types of
``Component`` classes are defined that can be used to represent different parts of a
design:

- :py:class:`bluemira.base.components.Component` is the base class and defines the
  framework on which other components can be defined within bluemira. When used directly
  it allows other types of component to be grouped together into super-structures. For
  example, a blanket first wall, breeding zone, manifold, and back supporting structure
  may be grouped into a blanket system. As these objects are not physical, they do not
  have a corresponding shape or material composition.
- :py:class:`bluemira.base.components.PhysicalComponent` defines the physical parts of a
  reactor, such as blanket layers, vessel shells, or ports. As implementations of
  :py:class:`bluemira.base.components.PhysicalComponent` correspond to a physical object,
  instances of that class can be defined with a shape and a material.
- :py:class:`bluemira.base.components.MagneticComponent` defines the magnetic parts of a
  reactor, such as poloidal or toroidal field coils. These have a shape and material and
  additionally define a conductor to provide the current-carrying filament.
