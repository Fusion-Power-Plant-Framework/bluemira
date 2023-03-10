Introduction
============

``Bluemira`` is a Python 3 framework for designing tokamak nuclear fusion reactors. It
has been developed from the original ``BLUEPRINT`` and ``MIRA`` codes, written by M.
Coleman and S. McIntosh, and F. Franza, respectively.

The overarching aim of the ``bluemira`` framework is to facilitate reactor design using
different levels of fidelity at different stages in the design process. The general idea
is that there is nothing particularly wrong with how tokamak fusion reactors are designed
today, except that it takes months to reach an initial design point. In ``bluemira``,
typical tokamak fusion reactor design activities are parameterised, automated, combined
together, and exposed to the user, enabling reactor designs to be generated in a matter
of minutes.

The goal is for ``bluemira`` to one day function as a multi-fidelity systems code for
tokamaks.

A wide variety of modules useful for the design of tokamaks are available, with more on
the way:

*  Interface to the 0/1-D ``PROCESS`` systems code
*  Interface to the 1.5-D ``PLASMOD`` transport and fixed boundary equilibrium solver
*  2-D fixed and free boundary equilibrium solvers
*  3-D magnetostatics solvers
*  3-D geometry and CAD
*  Simplified dynamic tritium fuel cycle model

Many typical tokamak fusion reactor design optimisation problems are also implemented,
leveraging the functionality of the above modules.

.. _how to use:

How to use ``Bluemira``
-----------------------

``Bluemira`` is designed to be used by three different types of user.

  Modeller
    A Modeller will execute a reactor build workflow (created by a '`Reactor Designer`_'), to carry out studies on a reactor design. Modellers will need to know about the parameters of a design, and how to manipulate JSON files to modify those parameters.

  _`Reactor Designer`
    A Reactor Designer will use ``Bluemira`` as a framework to create a design for a reactor.
    To design a reactor, the design workflow strategy needs to be considered and codified.
    Using Designer and Builder objects, each component of the reactor can be created and collected into a full reactor design, or used individually.

  Developers
    A developer of ``Bluemira`` will need to understand the program to a much more detailed level than a reactor designer. They will be involved with adding new features to ``Bluemira`` as well as helping a Reactor Designer or a Modeller to add a new feature or customisation option.

High level Architecture
^^^^^^^^^^^^^^^^^^^^^^^
The below figure shows the high level workflow to create a component for a reactor. To read more about each individual section click on the links in the figure.

.. graphviz:: base/design_build.dot
    :align: center

Roadmap
-------

``Bluemira`` is still in the early phases of development, and presently contains less
functionality than ``BLUEPRINT`` or ``MIRA`` did. We are very actively working to reach
a point where ``bluemira`` is greater than the sum of its parts. If you are interested in
using ``bluemira`` as an "early adopter", please do get in touch with us so that we can
help you.


.. figure:: bluemira-roadmap.svg
    :name: fig:bluemira-roadmap

    ``Bluemira`` development roadmap for 2023/24
