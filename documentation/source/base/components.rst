Reactors, Components and Managers
=================================

:py:class:`Components` are the fundamental building blocks of a bluemira design.
A :py:class:`Component` is a physical object or a group of physical objects, organised in a tree structure.
A user may want access to some properties associated with a :py:class:`Component` that are related
but not directly connected to the physical object.
For this reason we have :py:class:`ComponentManagers` which can provide helper functions for common operations or
have extra information stored such as the equilibira in a plasma :py:class:`ComponentManager`.

The Component class
-------------------

Two types of :py:class:`Component` classes are defined that can be used to represent different parts of a design:

- :py:class:`bluemira.base.components.Component` is the base class and defines the
  framework on which other components can be defined within bluemira.
  When used directly it allows other types of component to be grouped together into super-structures.
  For example, a blanket first wall, breeding zone, manifold, and back supporting structure may be grouped into a blanket system.
  As these objects are not physical, they do not have a corresponding shape or material composition.
- :py:class:`bluemira.base.components.PhysicalComponent` defines the physical parts of a
  reactor, such as blanket layers, vessel shells, or ports.
  As implementations of :py:class:`bluemira.base.components.PhysicalComponent` correspond to a physical object,
  instances of that class can be defined with a shape and a material.

A :py:class:`Component` can be used as shown below:

.. code-block:: pycon

    >>> reactor = Component("My Reactor")
    >>> plasma = PhysicalComponent("Plasma", plasma_shape)
    >>> blanket = Component("Blanket")
    >>> blanket.add_child(PhysicalComponent("First Wall", wall_shape))
    >>> blanket.add_child(PhysicalComponent("Breeding Zone", breeding_zone_shape))
    >>> reactor.add_child(plasma)
    >>> reactor.add_child(blanket)
    >>> reactor.tree()
    My Reactor (Component)
    ├── Plasma (PhysicalComponent)
    └── Blanket (Component)
        ├── First Wall (PhysicalComponent)
        └── Breeding Zone (PhysicalComponent)

ComponentManagers
-----------------

:py:class:`ComponentManagers` are designed to be created by the :ref:`Reactor Designer <how to use>`.
The aim is to make it easier to access logically associated properties of a :py:class:`Component` that may not be directly connected to the physical object.
It also can contain helper methods to ease access of specific sections of geometry,
for instance the separatrix of a plasma.

.. code-block:: python

    from bluemira.base.components import Component
    from bluemira.base.builder import ComponentManager

    class Plasma(ComponentManager):
        def lcfs(self):
            return (
                self.component
                .get_component("xz")
                .get_component('LCFS')
                .shape.boundary[0]
            )

A :py:class:`ComponentManager` should be how a :py:class:`Component` is used after creation within the top level of the reactor design.

Reactor
-------
