Developing bluemira
===================

Developing
----------

Please see the `guidelines for contributing to bluemira <https://github.com/Fusion-Power-Plant-Framework/bluemira/blob/main/CONTRIBUTING.md>`_.


Building Documentation
----------------------

The documentation is built with Sphinx. In addition the API documentation is created by `sphinx-autoapi`. There is an additional dependency on Graphviz to build the docs which can be installed with::

    apt-get update && apt-get install -y graphviz

To build the html documentation execute this command:

.. code-block:: bash

    sphinx-build documentation/source documentation/build

New pages can be added by adding them to the `documentation/source` directory and adding a
link in the `toctree` section of `index.rst`.

The index page lives at `documentation/build/index.html`

Any warnings issued by sphinx will need to be fixed before any documentation changes are merged. This includes API documentation within the code. There are some edges cases such as repeat definitions of global variables (eg. `try...except` statements on import of optional external codes) that can be difficult to fix. In that case please talk to a core developer about adding an exception.

