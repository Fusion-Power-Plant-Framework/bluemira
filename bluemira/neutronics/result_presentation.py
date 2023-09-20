"""Functions to present the results prettily
(Including both printed/logged texts and images)
"""
from typing import Any, Callable, Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
import openmc

from bluemira.base.constants import raw_uc
from bluemira.neutronics.constants import S_TO_YEAR, DPACoefficients
from bluemira.neutronics.params import TokamakGeometry


def print_df_decorator_with_title_string(
    title_string,
) -> Callable[[Callable[[bool], Any]], Callable[[bool], Any]]:
    """
    Decorator for methods inside OpenMCResult,
        so that the method has an added option to print the dataframe before exiting.
    Parameter
    ---------
    title_string: bool, default=True
    """

    def print_dataframe_decorator(method) -> Callable[[bool], Any]:
        def dataframe_method_wrapper(self, print_df: bool = True) -> Any:
            method_output = method(self)
            if print_df:
                if hasattr(method_output, "to_string"):  # duck-typing
                    output_str = method_output.to_string()
                else:
                    output_str = str(method_output)
                print("\n{}\n".format(title_string) + output_str)
            return method_output

        return dataframe_method_wrapper

    return print_dataframe_decorator


def get_percent_err(row):
    """
    Calculate a percentage error to the required row,
    assuming  cells had been filled out.

    Parameters
    ----------
    row: pd.Series object
        It should have the "mean" and "std. dev."
        row['mean']: scalar
        row['std. dev.']: scalar

    Returns
    -------
    fractional_error: scalar
    """
    # Returns to an OpenMC results dataframe that is the
    # percentage stochastic uncertainty in the result

    return row["std. dev."] / row["mean"] * 100.0


class PoloidalXSPlot(object):
    """Context manager so that we can save the plot as soon as we exit.
    Using the 'with' statement (i.e. in the syntax of context manager in python)
    also improves readability, as the save_name is written at the top of the indented
    block, so it's obvious what's the indented block plotting.
    """

    def __init__(self, save_name, title=None):
        self.save_name = save_name
        self.ax = plt.subplot()
        self.ax.axis("equal")
        if title:
            self.ax.set_title(title)

    def __enter__(self):
        """Return the initialized matplotlib axes object"""
        return self.ax

    def __exit__(self, exception_type, value, traceback):
        """Save and close upon exit."""
        plt.savefig(self.save_name)
        # self.ax.cla() # not necessary to clear axes or clear figure
        # self.ax.figure.clf()
        plt.close()


class OpenMCResult:
    """
    Class that looks opens up the openmc universe from the statepoint file,
        so that the dataframes containing the relevant results
        can be generated and reformatted by its methods.
    """

    def __init__(
        self,
        universe: openmc.Universe,
        src_rate: Union[int, float],
        statepoint_file: str = "statepoint.2.h5",
    ):
        self.universe = universe
        self.src_rate = src_rate
        self.statepoint_file = statepoint_file
        # Create cell and material name dictionaries to allow easy mapping to dataframe
        self.cell_names = {}
        self.mat_names = {}
        for cell_id, _cell in self.universe.cells.items():
            self.cell_names[cell_id] = _cell.name
            if _cell.fill is not None:  # if this cell is not a void
                self.mat_names[_cell.fill.id] = _cell.fill.name

        # Creating cell volume dictionary to allow easy mapping to dataframe
        self.cell_vols = {}
        for cell_id in self.universe.cells:
            self.cell_vols[cell_id] = self.universe.cells[cell_id].volume
        # Loads up the output file from the simulation
        self.statepoint = openmc.StatePoint(self.statepoint_file)

    def _load_dataframe_from_statepoint(self, tally_name: str):  # -> pd.DataFrame
        return self.statepoint.get_tally(name=tally_name).get_pandas_dataframe()

    @print_df_decorator_with_title_string("TBR")
    def load_tbr(self):  # -> Tuple[pd.Series, pd.Series]
        """Load the TBR value and uncertainty."""
        self.tbr_df = self._load_dataframe_from_statepoint("TBR")
        self.tbr = "{:.2f}".format(self.tbr_df["mean"].sum())
        self.tbr_e = "{:.2f}".format(self.tbr_df["std. dev."].sum())
        return self.tbr, self.tbr_e

    @print_df_decorator_with_title_string("Heating (MW)")
    def load_heating_in_MW(self):  # -> pd.DataFrame
        """Load the heating (sorted by material) dataframe"""
        heating_df = self._load_dataframe_from_statepoint("MW heating")
        # additional columns
        heating_df["material_name"] = heating_df["material"].map(self.mat_names)
        heating_df["%err."] = heating_df.apply(get_percent_err, axis=1).apply(
            lambda x: "%.1f" % x
        )
        # DataFrame columns rearrangement
        heating_df = heating_df[
            [
                "material",
                "material_name",
                "nuclide",
                "score",
                "mean",  # 'mean' units are MW
                "std. dev.",
                "%err.",
            ]
        ]  # rearrange dataframe into this desired order
        self.heating_df = heating_df
        return self.heating_df

    @print_df_decorator_with_title_string("Neutron Wall Load (eV)")
    def load_neutron_wall_loading(self):  # -> pd.DataFrame
        """Load the neutron wall load dataframe"""
        dfa_coefs = DPACoefficients()  # default assumes iron (Fe) is used.
        n_wl_df = self._load_dataframe_from_statepoint("neutron wall load")
        # additional columns
        n_wl_df["cell_name"] = n_wl_df["cell"].map(self.cell_names)
        n_wl_df["vol(cc)"] = n_wl_df["cell"].map(self.cell_vols)
        n_wl_df["dpa/fpy"] = (
            n_wl_df["mean"]
            * dfa_coefs.displacements_per_damage_eV
            / (n_wl_df["vol(cc)"] * dfa_coefs.atoms_per_cc)
            / S_TO_YEAR
            * self.src_rate
        )
        n_wl_df["dpa/fpy"] = n_wl_df["dpa/fpy"].apply(lambda x: "%.1f" % x)
        n_wl_df["%err."] = n_wl_df.apply(get_percent_err, axis=1).apply(
            lambda x: "%.1f" % x
        )
        n_wl_df = n_wl_df.drop(
            n_wl_df[~n_wl_df["cell_name"].str.contains("Surface")].index
        )  # ~ invert operator = doesnt contain
        # DataFrame columns rearrangement
        n_wl_df = n_wl_df.reindex([12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        n_wl_df = n_wl_df[
            [
                "cell",
                "cell_name",
                "particle",
                "nuclide",
                "score",
                "vol(cc)",
                "mean",  # 'mean' units are eV per source particle
                "std. dev.",
                "dpa/fpy",
                "%err.",
            ]
        ]

        self.neutron_wall_load = n_wl_df
        return self.neutron_wall_load

    @print_df_decorator_with_title_string("Photon Heat Flux MW m-2")
    def load_photon_heat_flux(self):  # -> pd.DataFrame
        """Load the photon heaat flux dataframe"""
        p_hf_df = self._load_dataframe_from_statepoint("photon heat flux")
        # additional columns
        p_hf_df["cell_name"] = p_hf_df["cell"].map(self.cell_names)
        p_hf_df["vol(cc)"] = p_hf_df["cell"].map(self.cell_vols)
        _MW_per_cm_2 = p_hf_df["mean"] / p_hf_df["vol(cc)"]
        p_hf_df["MW_m-2"] = raw_uc(_MW_per_cm_2.to_numpy(), "1/cm^2", "1/m^2")
        p_hf_df["%err."] = p_hf_df.apply(get_percent_err, axis=1).apply(
            lambda x: "%.1f" % x
        )
        # Scaling first wall results by factor to surface results
        surface_total = p_hf_df.loc[
            p_hf_df["cell_name"].str.contains("FW Surface"), "MW_m-2"
        ].sum()
        cell_total = p_hf_df.loc[
            ~p_hf_df["cell_name"].str.contains("FW Surface|PFC"), "MW_m-2"
        ].sum()  # ~ invert operator = doesnt contain
        _surface_factor = surface_total / cell_total
        p_hf_df["MW_m-2"] = np.where(
            ~p_hf_df["cell_name"].str.contains("FW Surface|PFC"),
            p_hf_df["MW_m-2"] * _surface_factor,
            p_hf_df["MW_m-2"],
        )
        # DataFrame columns rearrangement
        p_hf_df = p_hf_df.drop(
            p_hf_df[p_hf_df["cell_name"].str.contains("FW Surface")].index
        )
        p_hf_df = p_hf_df.drop(p_hf_df[p_hf_df["cell_name"] == "Divertor PFC"].index)
        p_hf_df = p_hf_df.replace(
            "FW", "FW Surface", regex=True
        )  # df.replace('Py','Python with ', regex=True)
        p_hf_df = p_hf_df[
            [
                "cell",
                "cell_name",
                "particle",
                "nuclide",
                "score",
                "vol(cc)",
                "mean",  # 'mean' units are MW cm
                "std. dev.",
                "MW_m-2",
                "%err.",
            ]
        ]

        self.photon_heat_flux = p_hf_df
        return self.photon_heat_flux

    def summarize(self, print_dfs):
        """Run all of the results formatting method available in this class."""
        # change to logging here?
        self.load_tbr(print_dfs)
        self.load_heating_in_MW(print_dfs)
        self.load_neutron_wall_loading(print_dfs)
        self.load_photon_heat_flux(print_dfs)


def geometry_plotter(
    cells_and_cell_lists: Dict[str, Union[List[openmc.Cell], openmc.Cell]],
    tokamak_geometry: TokamakGeometry,
) -> None:
    """
    Uses the OpenMC plotter to produce an image of the modelled geometry

    Parameters
    ----------
    cells_and_cell_lists:
        dictionary where each item is either a single openmc.Cell,
            or a list of openmc.Cell.
    tokamak_geometry : TokamakGeometry

    Returns
    -------
    Saves the plots to png files.
    Saves the plots to xml files.

    """
    # Assigning colours for plots
    cell_color_assignment = {
        cells_and_cell_lists["tf_coil_cell"]: "brown",
        cells_and_cell_lists["plasma_inner1"]: "dimgrey",
        cells_and_cell_lists["plasma_inner2"]: "grey",
        cells_and_cell_lists["plasma_outer1"]: "darkgrey",
        cells_and_cell_lists["plasma_outer2"]: "dimgrey",
        cells_and_cell_lists["divertor_inner1"]: "grey",
        cells_and_cell_lists["divertor_inner2"]: "dimgrey",
        cells_and_cell_lists["outer_vessel_cell"]: "white",
        cells_and_cell_lists["inb_vv_cells"][0]: "red",
        cells_and_cell_lists["outb_vv_cells"][1]: "orange",
        cells_and_cell_lists["outb_vv_cells"][2]: "yellow",
    }

    mat_color_assignment = {
        cells_and_cell_lists["bore_cell"]: "blue",
        cells_and_cell_lists["tf_coil_cell"]: "brown",
        cells_and_cell_lists["plasma_inner1"]: "white",
        cells_and_cell_lists["plasma_inner2"]: "white",
        cells_and_cell_lists["plasma_outer1"]: "white",
        cells_and_cell_lists["plasma_outer2"]: "white",
        cells_and_cell_lists["divertor_inner1"]: "white",
        cells_and_cell_lists["divertor_inner2"]: "white",
        cells_and_cell_lists["divertor_fw"]: "red",
        cells_and_cell_lists["outer_vessel_cell"]: "white",
        cells_and_cell_lists["outer_container"]: "darkgrey",
    }

    def color_cells(prefixed_cell_type, color):
        for i in range(len(cells_and_cell_lists[prefixed_cell_type + "_cells"])):
            mat_color_assignment[
                cells_and_cell_lists[prefixed_cell_type + "_cells"][i]
            ] = color

    # first wall: red
    color_cells("outb_fw", "red")
    color_cells("inb_fw", "red")
    # breeding zone: yellow
    color_cells("outb_bz", "yellow")
    color_cells("inb_bz", "yellow")
    # manifold: green
    color_cells("outb_mani", "green")
    color_cells("inb_mani", "green")
    # vacuum vessel: grey
    color_cells("outb_vv", "grey")
    color_cells("inb_vv", "grey")
    # divertor: cyan
    color_cells("divertor", "cyan")

    plot_width = 2 * (
        tokamak_geometry.major_r
        + tokamak_geometry.minor_r * tokamak_geometry.elong
        + tokamak_geometry.outb_fw_thick
        + tokamak_geometry.outb_bz_thick
        + tokamak_geometry.outb_mnfld_thick
        + tokamak_geometry.outb_vv_thick
        + 200.0  # margin
    )

    plot_list = []
    for _, basis in enumerate(["xz", "xy", "yz"]):
        plot = openmc.Plot()
        plot.basis = basis
        plot.pixels = [400, 400]
        plot.width = (plot_width, plot_width)
        if basis == "yz":
            plot.colors = cell_color_assignment
        else:
            plot.colors = mat_color_assignment
        plot.filename = f"./plots_{basis}"

        plot_list.append(plot)

    openmc.Plots(plot_list).export_to_xml()
    openmc.plot_geometry()
