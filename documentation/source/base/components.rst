Reactors, Components and Managers
=================================

:py:class:`~bluemira.base.components.Component`\s are the fundamental building blocks of a bluemira design.
A :py:class:`~bluemira.base.components.Component` is a physical object or a group of physical objects, organised in a tree structure.
A user may want access to some properties associated with a :py:class:`~bluemira.base.components.Component` that are related
but not directly connected to the physical object.
For this reason we have :py:class:`~bluemira.base.reactor.ComponentManager`\s which can provide helper functions for common operations or
have extra information stored such as the equilibira in a plasma :py:class:`~bluemira.base.reactor.ComponentManager`.
To manage a set of :py:class:`~bluemira.base.reactor.ComponentManager`\s a :py:class:`~bluemira.base.reactor.Reactor` class is used which represents the full reactor object.

The Component class
-------------------

Two types of :py:class:`~bluemira.base.components.Component` classes are defined that can be used to represent different parts of a design:

- :py:class:`bluemira.base.components.Component` is the base class and defines the
  framework on which other components can be defined within bluemira.
  When used directly it allows other types of component to be grouped together into super-structures.
  For example, a blanket first wall, breeding zone, manifold, and back supporting structure may be grouped into a blanket system.
  As these objects are not physical, they do not have a corresponding shape or material composition.
- :py:class:`bluemira.base.components.PhysicalComponent` defines the physical parts of a
  reactor, such as blanket layers, vessel shells, or ports.
  As implementations of :py:class:`bluemira.base.components.PhysicalComponent` correspond to a physical object,
  instances of that class can be defined with a shape and a material.

A :py:class:`~bluemira.base.components.Component` can be used as shown below:

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

A component object as above doesn't expose much functionality or flexibility and is designed to
exist as a container for CAD structures. The `ComponentManagers`_ and `Reactor`_ objects discussed below are
designed to give the :ref:`Reactor Designer <how to use>` this flexibility.

ComponentManagers
-----------------

:py:class:`~bluemira.base.reactor.ComponentManager`\s are designed to be created by the :ref:`Reactor Designer <how to use>`.
The aim is to make it easier to access logically associated properties of a :py:class:`~bluemira.base.components.Component` that may not be directly connected to the physical object.
It also can contain helper methods to ease access of specific sections of geometry,
for instance the separatrix of a plasma.

.. code-block:: python

    from bluemira.base.components import Component
    from bluemira.base.reactor import ComponentManager

    class Plasma(ComponentManager):
        def lcfs(self):
            return (
                self.component
                .get_component("xz")
                .get_component('LCFS')
                .shape.boundary[0]
            )

A :py:class:`~bluemira.base.reactor.ComponentManager` should be how a :py:class:`~bluemira.base.components.Component` is used after creation within the top level of the reactor design.

Reactor
-------

:py:class:`~bluemira.base.reactor.Reactor`\s are again designed to be created by the :ref:`Reactor Designer <how to use>`.
This object is the complete reactor and is a container that allows easy access to any part of it.
Methods on the :py:class:`~bluemira.base.reactor.Reactor` object have access to all parts of the reactor
enabling functionality that needs to interact with multiple :py:class:`~bluemira.base.reactor.ComponentManager`\s.

.. code-block:: python

    from bluemira.base.reactor import Reactor

    class MyReactor(Reactor):
        '''An example of how to declare a reactor structure.'''

        plasma: MyPlasma
        tf_coils: MyTfCoils

        def get_ripple(self):
            '''Calculate the ripple in the TF coils.'''

    reactor = MyReactor("My Reactor")
    reactor.plasma = build_plasma()
    reactor.tf_coils = build_tf_coils()
    reactor.show_cad()

A :py:class:`~bluemira.base.reactor.Reactor` interacts dynamically with :py:class:`~bluemira.base.reactor.ComponentManager`\s.
All the default methods on :py:class:`~bluemira.base.reactor.Reactor` such as :py:meth:`show_cad` will act
on the currently available :py:class:`~bluemira.base.reactor.ComponentManager`\s ignoring unavailable parts
of the reactor. If a :py:class:`~bluemira.base.components.Component` is directly added to a :py:class:`~bluemira.base.reactor.Reactor`
and not wrapped in a :py:class:`~bluemira.base.reactor.ComponentManager`\s it will be ignored by the :py:class:`~bluemira.base.reactor.Reactor` methods.
