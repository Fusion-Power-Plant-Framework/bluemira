# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
#
# bluemira is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# bluemira is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with bluemira; if not, see <https://www.gnu.org/licenses/>.
"""Functions to present the results prettily
(Including both printed/logged texts and images)
"""
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
import openmc
from tabulate import tabulate

from bluemira.base.constants import S_TO_YR, raw_uc
from bluemira.neutronics.constants import DPACoefficients
from bluemira.neutronics.params import TokamakGeometry


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
    try:
        return row["std. dev."] / row["mean"] * 100.0
    except ZeroDivisionError:
        return np.nan


class PoloidalXSPlot:
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
        src_rate: float,
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

    @property
    def tbr(self):
        """TBR"""
        self._load_tbr()
        return self._tbr, self._tbr_e

    @property
    def heating(self):
        """Heating"""
        self._load_heating()
        return self._heating_df

    @property
    def neutron_wall_load(self):
        """Neutron wall load (eV)"""
        self._load_neutron_wall_loading()
        return self._neutron_wall_load

    @property
    def photon_heat_flux(self):
        self._load_photon_heat_flux()
        return self._photon_heat_flux

    def _load_dataframe_from_statepoint(self, tally_name: str):
        return self.statepoint.get_tally(name=tally_name).get_pandas_dataframe()

    @staticmethod
    def _convert_dict_contents(dataset: Dict[str, Dict[int, List[Union[str, float]]]]):
        for k, v in dataset.items():
            vals = list(v.values())
            dataset[k] = vals if isinstance(vals[0], str) else np.array(vals)
        return dataset

    def _load_tbr(self):
        """Load the TBR value and uncertainty."""
        self.tbr_df = self._load_dataframe_from_statepoint("TBR")
        self._tbr = self.tbr_df["mean"].sum()
        self._tbr_e = self.tbr_df["std. dev."].sum()

    def _load_heating(self):
        """Load the heating (sorted by material) dataframe"""
        heating_df = self._load_dataframe_from_statepoint("MW heating")
        # additional columns
        heating_df["material_name"] = heating_df["material"].map(self.mat_names)
        heating_df["%err."] = heating_df.apply(get_percent_err, axis=1)
        # DataFrame columns rearrangement
        # rearrange dataframe into this desired order
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
        ]
        hdf = heating_df.to_dict()
        hdf["mean"] = raw_uc(heating_df["mean"].to_numpy(), "MW", "W")
        hdf["std. dev."] = raw_uc(heating_df["std. dev."].to_numpy(), "MW", "W")
        self._heating_df = self._convert_dict_contents(heating_df.to_dict())

    def _load_neutron_wall_loading(self):
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
            / S_TO_YR
            * self.src_rate
        )
        n_wl_df["%err."] = n_wl_df.apply(get_percent_err, axis=1)
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

        self._neutron_wall_load = self._convert_dict_contents(n_wl_df.to_dict())

    def _load_photon_heat_flux(self):
        """Load the photon heaat flux dataframe"""
        p_hf_df = self._load_dataframe_from_statepoint("photon heat flux")
        # additional columns
        p_hf_df["cell_name"] = p_hf_df["cell"].map(self.cell_names)
        p_hf_df["vol(cc)"] = p_hf_df["cell"].map(self.cell_vols)
        p_hf_df["MW_m-2"] = raw_uc(
            (p_hf_df["mean"] / p_hf_df["vol(cc)"]).to_numpy(), "1/cm^2", "1/m^2"
        )
        p_hf_df["%err."] = p_hf_df.apply(get_percent_err, axis=1)
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

        self._photon_heat_flux = self._convert_dict_contents(p_hf_df.to_dict())

    def __str__(self):
        """String representation"""
        tbr, err = self.tbr
        ret_str = f"TBR\n{tbr:.3f}±{err:.3f}"
        for title, data in zip(
            ("Heating (W)", "Neutron Wall Load (eV)", "Photon Heat Flux MW m-2"),
            (
                self.heating,
                self.neutron_wall_load,
                self.photon_heat_flux,
            ),
        ):
            ret_str = f"{ret_str}\n{title}\n{self._tabulate(data)}"
        return ret_str

    @staticmethod
    def _tabulate(
        records: Dict[str, Union[str, float]],
        tablefmt: str = "fancy_grid",
        floatfmt: str = ".3g",
    ) -> str:
        return tabulate(
            zip(*records.values()),
            headers=records.keys(),
            tablefmt=tablefmt,
            showindex=False,
            numalign="right",
            floatfmt=floatfmt,
        )


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
