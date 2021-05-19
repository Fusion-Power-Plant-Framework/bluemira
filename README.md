# BLUEMIRA
BLUEMIRA is an integrated inter-disciplinary design tool for future fusion reactors. It incorporates several modules, some of which rely on other codes, to carry out a range of typical conceptual fusion reactor design activities.

BLUEMIRA was created from two similar codes: BLUEPRINT and MIRA (see [publications](#Publications)).
  

## Modules
BLUEMIRA is comprised of various modules: 

* **base [WIP]**: a collection of utilities which make up the software backbone of the BLUEMIRA framework
* **beams**: a simple 3-D frame finite element model for structural analysis
* **cad [WIP]**: a CAD toolbox and generator, which builds CAD artefacts for a given reactor.
* **costs**: an ultra-crude proof-of-principle cost toolbox
* **equilibria [WIP]**: an ab initio equilibrium solver, used for generating reference equilibria for a given reactor, and optimising poloidal field systems.
* **fuelcycle**: a simplified dynamic Monte Carlo tritium fuel cycle model, used to calculate tritium start-up inventories, tritium release rates, and reactor doubling times.
* **geometry [WIP]**: a set of tools and objects used for the creation and manipulation of 2-D geometry
* **materials [WIP]**: a materials library, used for material properties in FE analyses and neutronics.
* **magnetostatics [WIP]**: a magnetostatics utility library.
* **neutronics [WIP]**: interface to build and run neutronics codes on reactor designs
* **nova**: a collection of tools for equilibria and the optimisation of the toroidal field coil shape. A modest fork of the Nova module, written by S. McIntosh.
* **syscodes**: an interface library to run external modules:
	* The 0/1-D power reactor optimisation code [PROCESS](https://www.sciencedirect.com/science/article/pii/S0920379614005961)
	* The 1.5-D transport and equilibrium code [PLASMOD](https://www.sciencedirect.com/science/article/pii/S0920379618302801), by E. Fable.
* **systems**: the home of reactor system objects and a number of simplified 0/2-D models for reactor cross-section design and analysis.
* **utilities**: collection of utility toolboxes for file IO, plotting, databases, machine learning, and misc.  

## Publications [](#Publications)

The BLUEMIRA code and its modules are described in the following publications:
* TBD

As BLUEMIRA is the result of an integration between two historical codes, some previous publications may also be of interest.

The BLUEPRINT code and its modules are described in the following publications:

* [BLUEPRINT: A novel approach to fusion reactor design, Coleman and McIntosh, *Fusion Engineering and Design* v **139** pp 26-38 (2019)](https://www.sciencedirect.com/science/article/pii/S0920379618308019)
* [DEMO tritium fuel cycle: performance, parameter explorations, and design space constraints, Coleman, Hörstensmeyer and Cismondi, *Fusion Engineering and Design* v **141** pp79-90 (2019)](https://www.sciencedirect.com/science/article/pii/S092037961930167X)
* [The design and optimisation of tokamak poloidal field systems in the BLUEPRINT framework, Coleman and McIntosh, *Fusion Engineering and Design* v **154** 111544 (2020)](https://doi.org/10.1016/j.fusengdes.2020.111544)
* [High-speed generation of neutronics-ready CAD models for DEMO design, Coleman, Shimwell, Davis, and McIntosh, *Fusion Engineering and Design* v **160** 112043 (2020)](https://www.sciencedirect.com/science/article/pii/S0920379620305913)

The MIRA code and its modules are described in the following publications:
* [On the implementation of new technology modules for fusion reactor systems codes, Franza, Boccaccinni, Fischer, Gade, and Heller, *Fusion Engineering and Design*, v **98-99** pp 1767-1770 (2015)](http://www.sciencedirect.com/science/article/pii/S0920379615001933)
* [Development of an advanced magnetic equilibrium model for fusion reactor system codes, Franza, Landman, and Petschanyi, *Fusion Engineering and Design*, v **136** pp 309-313 (2018)](http://www.sciencedirect.com/science/article/pii/S0920379618301157)
* [Development and Validation of a Computational Tool for Fusion Reactors' System Analysis, Franza, Ph.D thesis, Karlsruher Institut f&uuml;r Technologie (2019)](https://publikationen.bibliothek.kit.edu/1000095873)

## Terms of use

BLUEMIRA is [released under the LGPLv2.1 license](LICENSE.md), but we also ask that you show professional courtesy when using this code:

1. Since you are benefiting from work on BLUEMIRA, we ask that you submit any improvements you make to the code to us by submitting a pull request to this repository. If you are looking to develop aspects of this code or add parts to it, please contact us beforehand so that we can help/advise if necessary.

2. If you use BLUEMIRA results in a paper or professional publication, we ask that you discuss your work and results with one of the BLUEMIRA authors first so that we can check them. It is not necessary to list BLUEMIRA developers as co-authors. If one or more of the BLUEMIRA team are involved in preparing results or functionality used in your work, then those members should appear as co-authors.

3. If you use BLUEMIRA or any of its modules in your work, please help ensure that all the authors get the credit they deserve by citing the relevant BLUEMIRA, BLUEPRINT, and MIRA papers.

For a full list of existing publications, see [here](#Publications).

## Code of conduct

See [Code of Conduct](CODE_OF_CONDUCT.md)

## Installation

TBD
  
## Developing

For working on BLUEMIRA you will also need to install the development dependencies and the git pre-commit hook:  

```shell
python -m pip install --no-cache-dir  -e .'[dev]'
pre-commit install -f
```

Please see the [guidelines for contributing to BLUEMIRA](CONTRIBUTING.md).

In addition to the dependencies for installation, the following tools are useful for developing BLUEMIRA:
* Sphinx
* Black

  

Tests are run with `pytest`. To run the unit tests:

```shell
pytest
```

---

If you don't have PROCESS installed then you'll need to edit `tests/test_reactor.py`.

Change the line `"process_mode": "read"` to `"process_mode": "mock"`.

---


To run all the tests:

```shell
pytest --reactor --longrun --plotting-on
```

`pytest` also supports running individual test files and selecting tests by name or using a regular expression. See the [pytest documentation](https://docs.pytest.org/en/latest/usage.html).

After a test run an html report of the results can be found in `report.html`. A test coverage report can be found in the `htmlcov` directory.

## Building Documentation

The documentation is built with Sphinx. In addition the API documentation is created by `sphinx-apidoc`.

To build the html documentation execute this command:
```shell
sphinx-build documentation/source documentation/build
```

New pages can be added by adding them to the `documentation/source` directory and adding a link in the `toctree` section of `index.rst`.  The index page lives at `documentation/build/index.html`. API documentation will be automatically generated from the BLUEMIRA source code.



