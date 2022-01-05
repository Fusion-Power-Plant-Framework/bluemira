Introduction
============

``Bluemira`` is a Python 3 framework for designing tokamak nuclear fusion reactors. It
has been developed from the original ``BLUEPRINT`` code, written by M. Coleman and 
S. McIntosh, and the ``MIRA`` code, written by F. Franza.

The overarching aim of the ``bluemira`` framework is to facilitate reactor design using 
different levels of fidelity at different stages in the design process. The general idea
is that there is nothing particularly wrong with how tokamak fusion reactors are designed
today, except that it takes months to reach an initial design point. In ``bluemira``,
typical tokamak fusion reactor design activities are parameterised, automated, combined
together, and exposed to the user, enabling reactor designs to be generated in a matter 
of minutes.

A wide variety of modules useful for the design of tokamaks are available, with more on
the way:
* Interface to the 0/1-D ``PROCESS`` systems code
* Interface to the 1.5-D ``PLASMOD`` transport and fixed boundary equilibrium solver
* 2-D fixed and free boundary equilibrium solvers
* 3-D magnetostatics solvers
* 3-D geometry and CAD
* Simplified dynamic fuel cycle model

Many typical tokamak fusion reactor design optimisation problems are also implemented,
leveraging the functionality of the above modules.

``Bluemira`` is still in the early phases of development, and presently contains less
functionality than ``BLUEPRINT`` or ``MIRA`` did. We are very actively working to reach
a point where ``bluemira`` is greater than the sum of its parts. If you are interested in
using ``bluemira`` as an ''early adopter'', please do get in touch with us so that we can
help you.
