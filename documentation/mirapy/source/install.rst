
============
Installation
============

===========================
Spyder developed enviroment
===========================

The Scientific Python Development Environment Spyder will be installed
into the new enviroment.

Open the "Project" menu and create a new project from an existing directory
(i.e. mirapy folder). To make the example working, use the PYTHONPATH Manager
dialog box to add the path "mirapy/src".

============================
Pyscaffold (docs generation)
============================
The mirapy project has been created using the Pyscaffold project template...

Docs generation (html and pdf):

    >>> python setup.py docs
    >>> sphinx-build -b latex docs/ build/sphinx/latex/
    or (from docs folder)
    >>> make latexpdf

Run tests:
   
    >>> python setup.py tests
