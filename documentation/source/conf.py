# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
"""Configuration file for the Sphinx documentation builder."""
import os
import sys

from docutils import nodes, statemachine
from docutils.parsers.rst import Directive

sys.path.insert(0, os.path.abspath("../../"))  # noqa :E402

from bluemira._version import get_versions  # noqa :E402


def setup(app):
    """Setup function for sphinx"""
    # https://stackoverflow.com/questions/14110790/numbered-math-equations-in-restructuredtext
    app.add_css_file("css/custom.css")
    app.add_directive("params", ParamsDirective)
    app.connect("autoapi-skip-member", SkipAlreadyDocumented())


# To use markdown instead of rst see:
# https://www.sphinx-doc.org/en/master/usage/markdown.html

# To pull in docstrings we use autoapi
# https://sphinx-autoapi.readthedocs.io/

# -- Project information -----------------------------------------------------

project = "bluemira"
copyright = (
    "2021, M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris, D. Short"
)
author = "M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris, D. Short, UKAEA & contributors"

# The full version, including alpha/beta/rc tags
release = get_versions()["version"]


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx_rtd_theme",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_theme_options = {
    "canonical_url": "",
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    # Toc options
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

html_css_files = ["css/custom.css"]

numfig = True

# --- Configuration for sphinx-autoapi ---
extensions.append("sphinx.ext.inheritance_diagram")
extensions.append("autoapi.extension")

autoapi_type = "python"
autoapi_dirs = ["../../bluemira"]
autoapi_keep_files = False
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-inheritance-diagram",
    "show-module-summary",
    "special-members",
]


class ParamsDirective(Directive):
    """
    Generates the default parameters table for the given analysis module and class.
    """

    has_content = True

    def run(self):
        """
        Run the directive.
        """
        tab_width = self.options.get("tab-width", self.state.document.settings.tab_width)
        source = self.state_machine.input_lines.source(
            self.lineno - self.state_machine.input_offset - 1
        )

        try:
            import importlib

            analysis_module_name = self.content[0]
            analysis_class_name = self.content[1]

            analysis_module = importlib.import_module(analysis_module_name)
            analysis_class = getattr(analysis_module, analysis_class_name)

            text = analysis_class.default_params.tabulator(tablefmt="rst")
            lines = statemachine.string2lines(text, tab_width, convert_whitespace=True)
            self.state_machine.insert_input(lines, source)
            return []
        except Exception:
            return [
                nodes.error(
                    None,
                    nodes.paragraph(
                        text="Unable to generate parameter documentation at %s:%d:"
                        % (os.path.basename(source), self.lineno)
                    ),
                    nodes.paragraph(text=str(sys.exc_info()[1])),
                )
            ]


class SkipAlreadyDocumented:
    """
    Skip already documented items for autoapi.

    For use with global variables that are defined twice
    for instance in try..except import expressions and similar situations
    """

    def __init__(self):
        lis = [
            "bluemira.codes.process.api.ENABLED",
            "bluemira.codes.process.api.PROCESS_DICT",
        ]

        self.dict = {i: 0 for i in lis}

    def __call__(self, app, what, name, obj, skip, options):
        """autoapi-skip-member definition"""
        if name in self.dict:
            # Skip first occurrence
            if self.dict[name] < 1:
                skip = True
            else:
                skip = False
            self.dict[name] += 1
        return skip
