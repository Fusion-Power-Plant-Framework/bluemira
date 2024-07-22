# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
"""Configuration file for the Sphinx documentation builder."""
import os
import sys
from importlib.metadata import version as get_version

from docutils import nodes, statemachine
from docutils.parsers.rst import Directive


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
copyright = "2021-present, M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris, D. Short"
author = "M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris, D. Short, UKAEA & contributors"

release: str = get_version(project)
version: str = release.split("+")[0]

if version.startswith("0.1"):
    release = "develop"
    version = "develop"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx_rtd_theme",
    "sphinx_copybutton",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["**/*.md"]

suppress_warnings = []

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

_html_version = version

html_title = f"bluemira {_html_version} documentation"

numfig = True


# --- Configuration for sphinx-autoapi ---
autodoc_typehints = "both"

extensions.append("sphinx.ext.inheritance_diagram")
extensions.append("autoapi.extension")

autoapi_type = "python"
autoapi_dirs = ["../../bluemira"]
autoapi_keep_files = True
autoapi_options = [
    "members",
    "undoc-members",
    "private-members",
    "show-inheritance",
    "show-inheritance-diagram",
    "show-module-summary",
    "special-members",
]

# --- Configuration for plotting ---
extensions.append("matplotlib.sphinxext.plot_directive")
plot_formats = ["svg"]
plot_html_show_formats = False
plot_html_show_source_link = False

# --- Configuration for graphviz ---
extensions.append("sphinx.ext.graphviz")
graphviz_output_format = "svg"

# --- Configuration for myst-nb ---
extensions.append("myst_nb")
nb_execution_mode = "off"
nb_custom_formats = {
    ".ex.py": ["jupytext.reads", {"fmt": "py:percent"}],
}
myst_enable_extensions = ["amsmath", "dollarmath"]


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
        skip_list = [
            "bluemira.codes.process.api.ENABLED",
            "bluemira.codes.process.api.PROCESS_DICT",
            "bluemira.codes._polyscope.DefaultDisplayOptions.colour",
            "bluemira.codes._freecadapi.DefaultDisplayOptions.colour",
            "bluemira.geometry.optimisation._optimise.optimise_geometry",
            "bluemira.balance_of_plant.plotting.BalanceOfPlantPlotter.plot_options",
            "bluemira.balance_of_plant.steady_state.BalanceOfPlantModel.params",
            # possibly bug in autoapi with these ones
            "bluemira.base.builder.Builder.build",
            "bluemira.base.components.Component.name",
            "bluemira.base.components.PhysicalComponent.shape",
            "bluemira.base.components.PhysicalComponent.material",
            "bluemira.base.components.MagneticComponent.conductor",
            "bluemira.base.file.FileManager._reactor_name",
            "bluemira.base.file.FileManager._reference_data_root",
            "bluemira.base.file.FileManager._generated_data_root",
            "bluemira.base.parameter_frame._parameter.Parameter.value",
            "bluemira.base.reactor_config.ReactorConfig.config_data",
            "bluemira.codes.interface.CodesSolver.params",
            "bluemira.codes.openmc.solver.OpenMCNeutronicsSolver.params",
            "bluemira.codes.openmc.solver.OpenMCNeutronicsSolver.source",
            "bluemira.codes.openmc.solver.OpenMCNeutronicsSolver.tally_function",
            "bluemira.equilibria.coils._coil.Coil.x",
            "bluemira.equilibria.coils._coil.Coil.z",
            "bluemira.equilibria.coils._coil.Coil.ctype",
            "bluemira.equilibria.coils._coil.Coil.dx",
            "bluemira.equilibria.coils._coil.Coil.dz",
            "bluemira.equilibria.coils._coil.Coil.current",
            "bluemira.equilibria.coils._coil.Coil.j_max",
            "bluemira.equilibria.coils._coil.Coil.b_max",
            "bluemira.equilibria.coils._coil.Coil.discretisation",
            "bluemira.equilibria.coils._grouping.CoilSet.control",
            "bluemira.equilibria.optimisation.problem._breakdown.BreakdownCOP.bounds",
            "bluemira.equilibria.plotting.EquilibriumPlotter.eq",
            "bluemira.equilibria.run.PulsedCoilsetDesign.bd_settings",
            "bluemira.equilibria.run.PulsedCoilsetDesign.eq_settings",
            "bluemira.equilibria.run.OptimisedPulsedCoilsetDesign.pos_settings",
            "bluemira.fuel_cycle.timeline.Timeline.t",
            "bluemira.fuel_cycle.timeline.Timeline.ft",
            "bluemira.fuel_cycle.timeline.Timeline.DD_rate",
            "bluemira.fuel_cycle.timeline.Timeline.DT_rate",
            "bluemira.fuel_cycle.timeline.Timeline.bci",
            "bluemira.fuel_cycle.timeline.Timeline.phases",
            "bluemira.magnetostatics.baseclass.SourceGroup.sources",
            "bluemira.magnetostatics.baseclass.SourceGroup._points",
            "bluemira.mesh.meshing.Mesh.meshfile",
            "bluemira.optimisation._nlopt.optimiser.NloptOptimiser.opt_conditions",
            "bluemira.optimisation._nlopt.optimiser.NloptOptimiser.opt_parameters",
            "bluemira.radiation_transport.neutronics.neutronics_axisymmetric.PreCellStage.old_vv_wire",
            "bluemira.radiation_transport.neutronics.neutronics_axisymmetric.PreCellStage.in_wire",
            "bluemira.radiation_transport.neutronics.neutronics_axisymmetric.PreCellStage.vv_wire",
            "bluemira.radiation_transport.neutronics.neutronics_axisymmetric.PreCellStage.ex_wire",
            "bluemira.radiation_transport.neutronics.neutronics_axisymmetric.PreCellStage.blanket",
            "bluemira.structural.model.FiniteElementModel.geometry",
            "bluemira.structural.model.FiniteElementModel.load_case",
            "bluemira.structural.model.FiniteElementModel.n_fixed_dofs",
            "bluemira.structural.model.FiniteElementModel.fixed_dofs",
            "bluemira.structural.model.FiniteElementModel.fixed_dof_ids",
        ]

        self.skip_dict = {i: 0 for i in skip_list}

    def __call__(self, app, what, name, obj, skip, options):
        """autoapi-skip-member definition"""
        if name in self.skip_dict:
            # Skip first occurrence
            if self.skip_dict[name] < 1:
                skip = True
            self.skip_dict[name] += 1
        return skip


# autoapi inheritance diagram hack
import sphinx.ext.inheritance_diagram as inheritance_diagram  # noqa: E402

_old_html_visit_inheritance_diagram = inheritance_diagram.html_visit_inheritance_diagram


def html_visit_inheritance_diagram(self, node):
    """
    Hacks the uri of the inheritance diagram if its an autoapi diagram

    By default the refuri link is to ../../<expected file>.html whereas
    the actual file lives at autoapi/bluemira/<expected file>.html.

    refuri is used for parent classes outside of the current file.

    The original function appends a further ../ to the refuri which has the
    effect of returning to the root directory of the built documentation.

    I havent found a method to set the default expected path and it seems to
    be a slight incompatibility between autoapi and inheritance-diagram

    This replaces the wrong path with a corrected path to the autoapi folder,
    otherwise we get 404 links from the diagram links.
    """
    current_filename = self.builder.current_docname + self.builder.out_suffix
    if "autoapi" in current_filename:
        for n in node:
            refuri = n.get("refuri")
            if refuri is not None:
                n["refuri"] = f"autoapi/bluemira/{refuri[6:]}"

    return _old_html_visit_inheritance_diagram(self, node)


inheritance_diagram.html_visit_inheritance_diagram = html_visit_inheritance_diagram
