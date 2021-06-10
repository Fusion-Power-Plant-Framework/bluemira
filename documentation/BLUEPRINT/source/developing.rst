Developing BLUEPRINT
====================

Developing
----------

For working on BLUEPRINT you will also need to install the development dependencies:

.. code-block:: bash

    pip install -r requirements-develop.txt
    pre-commit install

Please see the `guidelines for contributing to BLUEPRINT <https://gitlab.com/CoronelBuendia/BLUEPRINT/-/blob/develop/CONTRIBUTING.md>`_.

Please note that commits are automatically passed through the ``black`` code auto-formatter for consistency. Make sure to install `pre-commit` in the BLUEPRINT directory as indicated in the installation instructions, in order to activate auto-formatting.

Tests are run with ``pytest``. To run the unit tests:

.. code-block:: bash

    pytest -m "not longrun"

To run just the smoke test:

.. code-block:: bash

    pytest -m smoketest

.. note::

    If you don't have PROCESS installed then you'll need to edit ``tests/test_reactor.py``.

    Change the line ``"process_mode": "read"`` to ``"process_mode": "mock"``.

To run all the tests:

.. code-block:: bash

    pytest

`pytest` also supports running individual test files and selecting tests by name
or using a regular expression. See:

* https://docs.pytest.org/en/latest/usage.html

After a test run an html report of the results can be found in ``report.html``.
A test coverage report can be found in the ``htmlcov`` directory.


Plotting Flag
^^^^^^^^^^^^^

By default tests are run without showing plots. Showing the plots requires manual intervention to close the plot
windows. Pass the ``--plotting-on`` flag to pytest to show plots during a test run:

.. code-block:: bash

    pytest --plotting-on -m "not longrun"


Building Documentation
----------------------

The documentation is built with Sphinx. In addition the API documentation is created by ``sphinx-autoapi``. There is an additional dependency on Graphviz to build the docs which can be installed with::

    apt-get update && apt-get install -y graphviz

To build the html documentation execute this command:

.. code-block:: bash

    sphinx-build documentation/source documentation/build

New pages can be added by adding them to the ``documentation/source`` directory and adding a
link in the ``toctree`` section of ``index.rst``.

The index page lives at ``documentation/build/index.html``. To locally view the website you can either open the ``index.html`` file in a web browser or you can run a temporary http server in the ``build`` directory::

    cd documentation/build
    python -m http.server 9999

The port number (9999) can be customised as you see fit and the website can be accessed at `localhost:9999 <http://localhost:9999>`_.

Any warnings issued by sphinx will need to be fixed before any documentation changes are merged. This includes API documentation within the code. There are some edges cases such as repeat definitions of global variables (eg. ``try...except`` statements on import of optional external codes) that can be difficult to fix. In that case please talk to a core developer about adding an exception.

Warning's from ``autoapi`` are a bit cryptic. The line numbers on the warnings are associated with the file::

   documentation/build/_sources/autoapi/BLUEPRINT/<sub package>/<module>/index.rst.txt

which is the generated .rst file the documentation is generated from. Using this file you can see where the error originates from.
For example with the module ``BLUEPRINT.equilibria.constants`` the generated rst will be located at::

   documentation/build/_sources/autoapi/BLUEPRINT/equilibria/constants/index.rst.txt

Remember that function docstrings is rst therefore follows all of its idioms.

Some useful links:

* `rst syntax <https://thomas-cokelaer.info/tutorials/sphinx/rest_syntax.html>`_
* `Sphinx getting started <https://www.sphinx-doc.org/en/master/usage/quickstart.html>`_
* `Symbols <https://docutils.sourceforge.io/docs/ref/rst/definitions.html>`_
* `Demo documentation <https://sphinx-rtd-theme.readthedocs.io/en/stable/demo/structure.html>`_
