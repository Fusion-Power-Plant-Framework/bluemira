Release Strategy
================

Schedule
--------

A new version of Bluemira is tagged every 6 weeks; marking the end of a
development cycle. A scheduled GitHub Actions workflow opens a PR to
(fast-forward) merge the ``develop`` branch into ``main``. Once merged,
a new `release is manually created <#creating-a-release>`__.

When a release is generated, a pull request is automatically opened to
merge the ``develop-dependencies`` branch into ``develop``. This
updates, and fixes, the project's dependencies ready for the next
release cycle.

Creating a Release
------------------

Releases are tagged, and signed, by a GitHub Actions workflow. This
workflow is manually triggered by a repository maintainer, who specifies
a version number. The workflow performs something similar to the below.

.. code:: console

   git checkout main
   git tag -as v${VERSION} -m "Release v${VERSION}"
   git push --tags

Manually specifying a version number gives full flexibility to
maintainers in when to increment major/minor/patch/tweak version
numbers.

Versioning
----------

Bluemira's versioning strategy follows the scheme laid out in
`PEP440 <https://peps.python.org/pep-0440/>`__, but always uses 3
version numbers (major, minor, patch) as in semantic versioning. As per
the PEP, the version can optionally contain a suffix specifying a
pre-release (``.(a|b|rc)N``), post release (``.postN``), or development
release (``.devN``) number.

Release Notes
-------------

TBC.

Options:

-  Manually update a 'release-notes' file on every PR.
-  Look into generated notes using
   `towncrier <https://github.com/twisted/towncrier>`__ Python package.
-  Write them manually.
-  Use git commit messages.
