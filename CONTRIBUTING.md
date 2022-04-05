# Contributing to ``bluemira``

When contributing to this repository, please first discuss the change you wish to make
via issue, email, or any other method with the owners of this repository before making a
change.

## Code of Conduct

This project and everyone participating in it is governed by the  Contributor Covenant
Code of Conduct. By participating, you are expected to uphold this code. Please report
unacceptable behavior to [matti.coleman@ukaea.uk](mailto:matti.coleman@ukaea.uk) and/or [ivan.maione@kit.edu](mailto:ivan.maione@kit.edu).

## Submitting an issue

In order to help us address your issue effectively, we ask that you follow this
procedure when creating an issue:

* Check that the issue is not already described elsewhere in [Issues
  ](https://github.com/Fusion-Power-Plant-Framework/bluemira/issues)
* Write a fairly complete description of the bug/feature/problem in the issue, using
  the provided templates and adding the relevant tags
* Tag the ``bluemira`` module(s) the issue concerns
* If the issue is linked to a [Project](https://github.com/Fusion-Power-Plant-Framework/bluemira/projects), please tag it

## Submitting a bug report

``Bluemira`` is software in development and is therefore likely to contain bugs. If you
discover bugs, please follow this procedure:

* Raise an issue using the bug template with a `bug` flag
* Include a way to reproduce the bug in the issue, let us know the expected result and
  what actually happens

## Submitting a pull request

Please discuss any feature ideas you have with the developers before submitting them, as
you may not be aware of parallel development work taking place, or implementation
decisions / subtleties which are relevant. The ideal workflow for submitting a pull
request is as follows:

* Discuss the feature with the developers first
* Submit an issue documenting the intent of the feature you wish to develop
* Make a branch of the `develop` branch which bears a similar name as your issue (e.g.
  `new_feature`)
* Develop your feature(s) in your `new_feature` branch
* Discuss any problems that arise in the associated issue, perhaps documenting your
  progress
* Finally, as the author of the `new_feature` branch, you must submit a pull request
  onto the `develop` branch
  * Link the relevant issue(s) and project(s)
  * Add an assignee, who will be responsible for reviewing and merging the PR. This
    should be the person you feel has the most relevant technical background and/or the
    most familiar with the underlying issue or code.
  * The reviewers will be automatically selected based on which module(s) of the code your pull request affects.

The merge request will be reviewed by the core development team before potentially being accepted
.

## Python style guide

``Bluemira`` is an object-oriented code and is designed to run relatively fast. We use
objects to represent physical entities, as well as for certain solvers where it is
useful to persist the state of some stored calculations in order to save run-time. When
writing objects, make sure all attributes are initialised in the ``__init__`` method,
and that all methods and variables that are not user-facing are made protected with a
leading underscore: ``_protected_method``.

We try to keep functions as single-purposes as possible, with as few arguments and
keyword arguments as possible. Similarly, with classes, we try to keep the number of
attributes and methods relatively low.

Python can run relatively slowly when compared with compiled languages, and in key areas
we use a couple of tricks to speed things up:
* Using ``__slots__`` on classes to save memory
* Using ``numba`` just-in-time compilation to speed up certain bottleneck calculations

Please try and follow these relatively loose guidelines when developing ``bluemira``. We
also recommend you use an integrated development environment with appropriate code
linting to improve the code you contribute.

``Bluemira`` is strictly auto-formatted using the [black
](https://pypi.org/project/black/) module. This is an opinionated subset of the
[Python PEP8 style guide](https://www.python.org/dev/peps/pep-0008/).

If you don't like how ``black`` formats your code, join the club... the loss of
aesthetics is the price to pay for uniformity and consistency! The point of using it is
that all code should more or less look the same, regardless of who writes it.

Please read up on this if you need more details. For ``bluemira``, ``black`` has the
potential to generate very ugly code, especially if:

* you are playing code golf
* you are writing data tables of some form
* you are coding out long mathematical formulae

For the first case, it is probably best you don't do this... If ``black`` breaks your
one-line or chain calls into multiple lines, consider breaking these down for
readability.

In some cases, it is nicer to turn off the auto-formatting. You can do this as follows:
```python
# fmt: off
default_params = [
['shaf_shift', 'Shafranov shift of plasma (geometric=>magnetic)', 0.5, 'm', None, 'Input'],
['lambda_q', 'Scrape-off layer power decay length', 0.003, 'm', None, 'Input']
]
# fmt: on
```

On top of ``black``, which only handles formatting, code in bluemira is also checked for
quality using [flake8](https://flake8.pycqa.org/en/latest/) and various extensions to it
. This is more a question of code style, which black doesn't cover in full.

Code that is committed to a branch is automatically checked for quality using pre-commit
. Violations detected by ``flake8`` are printed to the console for information. It does
not prevent you from pushing code, but it will fail in the continuous integration.
Please address issues raised by ``flake8`` prior to pushing your code.

When writing code for ``bluemira``, please adhere to the following Python naming
conventions:

* `ALL_CAPITALS_SNAKE_CASE` for global constants

* `CapitalisedWords` for classes

* `lower_case_snake_case` for functions and methods

* `lower_case_snake_case` for arguments and variables

Abbreviations are allowed to be in capitals, e.g.:

* `show_CAD`
* `build_TF_coils`

A full list of accepted abbreviations can be found in the `.flake8` file under
`ignore_names`.

We try to stick to descriptive `lower_case_snake_case` argument and local variable names with the general rule that names less than three characters should not be used. This is not a hard rule, and there are some notable exceptions:
* `i`, `j`, `k` for integers (e.g. looping, indexing)
* `m`, `n` for integers (e.g. array sizes)
* `x`, `y`, `z`, `r` for floats or arrays describing coordinates
* Where it makes sense to reflect mathematical notation (e.g. `R_0`)

## Documentation style guide

Please write all written documentation in `.rst.` format.

Please write all Python docstrings in `numpydoc` style, see details [here](
https://numpydoc.readthedocs.io/en/latest/format.html). An example is shown below:

```python
def IPB98y2(I_p, B_t, P_sep, n19, R_0, A, kappa):
	"""
	ITER IPB98(y, 2) Confinement time scaling [2]

	Parameters
	----------
	I_p: float
		Plasma current [MA]
	B_t: float
		Toroidal field at R_0 [T]
	P_sep: float
		Separatrix power [MW]
	n19: float
		Line average plasma density [10^19 1/m^3]
	R_0: float
		Major radius [m]
	A: float
		Aspect ratio
	kappa: float
		Plasma elongation

	Returns
	-------
	tau_E: float
		The energy confinement time [s]


	Notes
	-----
	[2] ITER Physics Expert Group, Nucl. Fus. 39, 12, <https://iopscience.iop.org/article/10.1088/0029-5515/39/12/302/pdf>

	\t:math:`\\tau_{E}=0.0562I_p^{0.93}B_t^{0.15}P_{sep}^{-0.69}n^{0.41}M^{0.19}R_0^{1.97}A^{-0.57}\\kappa^{0.78}`
	"""
```

The API documentation is built automatically using `sphinx` which reads docstrings and
processes them accordingly.

## Releases and Packaging

Release versions of ``bluemira`` are generated from git tags, and [versioneer](
https://github.com/python-versioneer/python-versioneer) is used so that the most recent
tag is dynamically pulled into the ``bluemira`` itself to set `__version__` correctly.
[Useage of versioneer for releases](
https://github.com/python-versioneer/python-versioneer/blob/master/INSTALL.md#post-installation-usage)
is quite straightforward, and that link also gives details on how ``versioneer`` is
installed, should it be necessary to do that again. If this fails and you haven't
touched `versioneer.py` or `bluemira/_version.py`, then it is possible
that ``versioneer`` needs to be updated:
<https://github.com/python-versioneer/python-versioneer#updating-versioneer>
