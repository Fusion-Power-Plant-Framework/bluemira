equilibria
==========

A 2-D finite difference free boundary equilibrium solver and
optimisation package.

.. toctree::
    :maxdepth: 3

    equilibria_theory
    equilibria_practice


This module was written by Matti Coleman, but is based heavily on the
works of:

- Young Mu Jeon - NFRI, Daejeon, Korea: `Development of a free-boundary tokamak equilibrium solver for advanced study of tokamak equilibria <https://arxiv.org/pdf/1503.03135.pdf>`_, Journal of the Korean Physical Society, Vol. 67, No. 5, September 2015, pp. 843-853. The calculation logic, profile constraints, unconstrained optimisation approach, and solution strategy elegantly detailed in this paper are recreated in this module.
- Ben Dudson - York, United Kingdom: Who’s beautifully organised `freegs <https://github.com/bendudson/freegs>`_ code provided much of the software and OO logic behind the implementation of this module.
- Simon McIntosh, UKAEA/ITER Organisation, United Kingdom: Who’s Nova code (private repo) provided much of the inspiration behind the optimisation algorithms, and who’s calculation of the force constraints and force Jacobian have enabled a constrained optimisation approach to be developed.

The author is greatly indebted to the above.

This module was extended by James Cook to include zonal position optimisations.
