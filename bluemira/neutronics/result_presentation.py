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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import openmc
from tabulate import tabulate

from bluemira.base.constants import raw_uc
from bluemira.base.look_and_feel import bluemira_debug
from bluemira.neutronics.constants import DPACoefficients
from bluemira.neutronics.params import TokamakGeometry


def get_percent_err(row):
    """
    Calculate a percentage error to the required row,
    assuming cells had been filled out.

    Parameters
    ----------
    row: pd.Series object
        It should have the "mean" and "std. dev."
        row['mean']: scalar
        row['std. dev.']: scalar

    Returns
    -------
    fractional_error: scalar

    Usage
    -----
    dataframe.apply(get_percent_err),
    where dataframe must have one row named "std. dev." and another named "mean".
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
        self.ax.set_xlabel("r (m)")
        self.ax.set_ylabel("z (m)")
        if title:
            self.ax.set_title(title)

        # monkey patch on two methods that automatically convert the coordinates to [m].
        def _monkey_patch_plot_cm(x, y, *arg, **kwargs):
            """Line plot coodinates (given in cm) in meters."""
            return self.ax.plot(
                raw_uc(x, "cm", "m"), raw_uc(y, "cm", "m"), *arg, **kwargs
            )

        def _monkey_patch_scatter_cm(x, y, *arg, **kwargs):
            """Scatter plot coodinates (given in cm) in meters."""
            return self.ax.scatter(
                raw_uc(x, "cm", "m"), raw_uc(y, "cm", "m"), *arg, **kwargs
            )

        self.ax.plot_cm = _monkey_patch_plot_cm
        self.ax.scatter_cm = _monkey_patch_scatter_cm

    def __enter__(self):
        """Return the initialized matplotlib axes object"""
        return self.ax

    def __exit__(self, exception_type, value, traceback):
        """Save and close upon exit."""
        plt.savefig(self.save_name)
        # self.ax.cla() # not necessary to clear axes or clear figure
        # self.ax.figure.clf()
        plt.close()


@dataclass
class OpenMCResult:
    """
    Class that looks opens up the openmc universe from the statepoint file,
        so that the dataframes containing the relevant results
        can be generated and reformatted by its methods.
    """

    tbr: float
    tbr_err: float
    heating: Dict
    neutron_wall_load: Dict
    """Neutron wall load (eV)"""

    photon_heat_flux: Dict
    """Photon heat flux"""

    universe: openmc.Universe
    src_rate: float
    statepoint: openmc.StatePoint
    statepoint_file: str
    cell_names: Dict
    cell_vols: Dict  # [m^3]
    mat_names: Dict

    volume_file: str
    stochastic_cell_volumes: Optional[Dict[int, float]] = None
    volume_state: Optional[openmc.VolumeCalculation] = None

    @classmethod
    def from_run(
        cls,
        universe: openmc.Universe,
        src_rate: float,
        statepoint_file: str = "statepoint.2.h5",
        volume_file: str = "volume_1.h5",
    ):
        """Create results class from run statepoint"""
        # Create cell and material name dictionaries to allow easy mapping to dataframe
        cell_names = {}
        mat_names = {}
        for cell_id, _cell in universe.cells.items():
            cell_names[cell_id] = _cell.name
            if _cell.fill is not None:  # if this cell is not a void
                mat_names[_cell.fill.id] = _cell.fill.name

        # Creating cell volume dictionary to allow easy mapping to dataframe
        cell_vols = {}
        for cell_id in universe.cells:
            try:
                cell_vols[cell_id] = raw_uc(
                    universe.cells[cell_id].volume, "cm^3", "m^3"
                )
            except TypeError:
                cell_vols[cell_id] = universe.cells[
                    cell_id
                ].volume  # catch the None's or na.
            # provided by openmc in cm^3, but we want to save it in m^3
        # Loads up the output file from the simulation
        statepoint = openmc.StatePoint(statepoint_file)
        tbr, tbr_err = cls._load_tbr(statepoint)
        volume_state, st_cell_volumes = cls._load_volume_calculation(
            volume_file, cell_names
        )

        return cls(
            universe=universe,
            src_rate=src_rate,
            statepoint_file=statepoint_file,
            statepoint=statepoint,
            cell_names=cell_names,
            cell_vols=cell_vols,
            mat_names=mat_names,
            tbr=tbr,
            tbr_err=tbr_err,
            heating=cls._load_heating(statepoint, mat_names, src_rate),
            neutron_wall_load=cls._load_neutron_wall_loading(
                statepoint, cell_names, cell_vols, src_rate
            ),
            photon_heat_flux=cls._load_photon_heat_flux(
                statepoint, cell_names, cell_vols, src_rate
            ),
            stochastic_cell_volumes=st_cell_volumes,
            volume_state=volume_state,
            volume_file=volume_file,
        )

    @staticmethod
    def _load_volume_calculation(volume_file, cell_names):
        if Path(volume_file).is_file():
            vol_results = openmc.VolumeCalculation.from_hdf5("volume_1.h5")
            vols = vol_results.volumes
            ids = list(vols.keys())
            cell_volumes = {
                "cell": ids,
                "cell_names": [cell_names[i] for i in ids],
                "Stochasitic Volumes": list(raw_uc(list(vols.values()), "cm^3", "m^3")),
            }

        else:
            bluemira_debug("No volume file found")
            vol_results = None
            cell_volumes = None

        return vol_results, cell_volumes

    @staticmethod
    def _load_dataframe_from_statepoint(statepoint, tally_name: str):
        return statepoint.get_tally(name=tally_name).get_pandas_dataframe()

    @staticmethod
    def _convert_dict_contents(dataset: Dict[str, Dict[int, List[Union[str, float]]]]):
        for k, v in dataset.items():
            vals = list(v.values()) if isinstance(v, dict) else v
            dataset[k] = vals if isinstance(vals[0], str) else np.array(vals)
        return dataset

    @classmethod
    def _load_tbr(cls, statepoint):
        """Load the TBR value and uncertainty."""
        tbr_df = cls._load_dataframe_from_statepoint(statepoint, "TBR")
        return tbr_df["mean"].sum(), tbr_df["std. dev."].sum()

    @classmethod
    def _load_heating(cls, statepoint, mat_names, src_rate):
        """Load the heating (sorted by material) dataframe"""
        # mean and std. dev. are given in eV per source particle,
        # so we don't need to show them to the user.
        heating_df = cls._load_dataframe_from_statepoint(statepoint, "material heating")
        heating_df["material_name"] = heating_df["material"].map(mat_names)
        heating_df["mean(W)"] = raw_uc(
            heating_df["mean"].to_numpy() * src_rate, "eV/s", "W"
        )
        heating_df["err."] = raw_uc(
            heating_df["std. dev."].to_numpy() * src_rate, "eV/s", "W"
        )
        heating_df["%err."] = heating_df.apply(get_percent_err, axis=1)
        # rearrange dataframe into this desired order
        heating_df = heating_df[
            [
                "material",
                "material_name",
                "nuclide",
                "score",
                "mean(W)",
                "err.",
                "%err.",
            ]
        ]
        hdf = heating_df.to_dict()
        return cls._convert_dict_contents(hdf)

    @classmethod
    def _load_neutron_wall_loading(cls, statepoint, cell_names, cell_vols, src_rate):
        """Load the neutron wall load dataframe"""
        dfa_coefs = DPACoefficients()  # default assumes iron (Fe) is used.
        n_wl_df = cls._load_dataframe_from_statepoint(statepoint, "neutron wall load")
        n_wl_df["cell_name"] = n_wl_df["cell"].map(cell_names)
        n_wl_df["vol (m^3)"] = n_wl_df["cell"].map(cell_vols)
        total_displacements_per_second = (
            n_wl_df["mean"] * dfa_coefs.displacements_per_damage_eV * src_rate
        )  # "mean" has units "eV per source particle"
        # total number of atomic displacements per second in the cell.
        num_atoms_in_cell = n_wl_df["vol (m^3)"] * raw_uc(
            dfa_coefs.atoms_per_cc, "1/cm^3", "1/m^3"
        )
        n_wl_df["dpa/fpy"] = raw_uc(
            total_displacements_per_second.to_numpy() / num_atoms_in_cell.to_numpy(),
            "1/s",
            "1/year",
        )

        n_wl_df["%err."] = n_wl_df.apply(get_percent_err, axis=1)
        # keep only the surface cells:
        n_wl_df = n_wl_df.drop(
            n_wl_df[~n_wl_df["cell_name"].str.contains("Surface")].index
        )
        # DataFrame columns rearrangement
        n_wl_df = n_wl_df.reindex([12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        n_wl_df = n_wl_df[
            [
                "cell",
                "cell_name",
                "particle",
                "nuclide",
                "score",
                "vol (m^3)",
                "dpa/fpy",
                "%err.",
            ]
        ]

        return cls._convert_dict_contents(n_wl_df.to_dict())

    @classmethod
    def _load_photon_heat_flux(cls, statepoint, cell_names, cell_vols, src_rate):
        """Load the photon heaat flux dataframe"""
        p_hf_df = cls._load_dataframe_from_statepoint(statepoint, "photon heating")
        p_hf_df["cell_name"] = p_hf_df["cell"].map(cell_names)

        p_hf_df["%err."] = p_hf_df.apply(get_percent_err, axis=1)
        p_hf_df["vol (m^3)"] = p_hf_df["cell"].map(cell_vols)
        p_hf_df["heating (W)"] = photon_heating = raw_uc(
            p_hf_df["mean"].to_numpy() * src_rate, "eV/s", "W"
        )
        p_hf_df["heating std.dev."] = photon_heating_stddev = raw_uc(
            p_hf_df["std. dev."].to_numpy() * src_rate, "eV/s", "W"
        )
        p_hf_df["vol. heating (W/m3)"] = photon_heating / p_hf_df["vol (m^3)"]
        p_hf_df["vol. heating std.dev."] = photon_heating_stddev / p_hf_df["vol (m^3)"]

        # Scaling first wall results by factor to surface results
        surface_total = p_hf_df.loc[
            p_hf_df["cell_name"].str.contains("FW Surface"), "heating (W)"
        ].sum()
        cell_total = p_hf_df.loc[
            ~p_hf_df["cell_name"].str.contains("FW Surface|PFC"), "heating (W)"
        ].sum()
        _surface_factor = surface_total / cell_total
        # in-place modification
        p_hf_df["vol. heating (W/m3)"] = np.where(
            ~p_hf_df["cell_name"].str.contains(
                "FW Surface|PFC"
            ),  # modify the matching entries,
            p_hf_df["heating (W)"] * _surface_factor,
            p_hf_df["heating (W)"],  # otherwise leave it unchanged.
        )
        p_hf_df["vol. heating (W/m3)"] = np.where(
            ~p_hf_df["cell_name"].str.contains("FW Surface|PFC"),
            p_hf_df["heating std.dev."] * _surface_factor,
            p_hf_df["heating std.dev."],
        )
        p_hf_df["vol. heating (W/m3)"] = np.where(
            ~p_hf_df["cell_name"].str.contains("FW Surface|PFC"),
            p_hf_df["vol. heating (W/m3)"] * _surface_factor,
            p_hf_df["vol. heating (W/m3)"],
        )
        p_hf_df["vol. heating (W/m3)"] = np.where(
            ~p_hf_df["cell_name"].str.contains("FW Surface|PFC"),
            p_hf_df["vol. heating std.dev."] * _surface_factor,
            p_hf_df["vol. heating std.dev."],
        )
        # DataFrame columns rearrangement
        p_hf_df = p_hf_df.drop(
            p_hf_df[p_hf_df["cell_name"].str.contains("FW Surface")].index
        )
        p_hf_df = p_hf_df.drop(p_hf_df[p_hf_df["cell_name"] == "Divertor PFC"].index)
        p_hf_df = p_hf_df.replace(
            "FW", "FW Surface", regex=True
        )  # expand the word again
        p_hf_df = p_hf_df[
            [
                "cell",
                "cell_name",
                "particle",
                "nuclide",
                "score",
                "vol (m^3)",
                "heating (W)",
                "heating std.dev.",
                "vol. heating (W/m3)",
                "vol. heating std.dev.",
                "%err.",
            ]
        ]
        p_hf_dict = p_hf_df.to_dict()
        return cls._convert_dict_contents(p_hf_dict)

    def __str__(self):
        """String representation"""
        ret_str = f"TBR\n{self.tbr:.3f}±{self.tbr_err:.3f}"
        for title, data in zip(
            ("Heating (W)", "Neutron Wall Load (eV)", "Photon Heat Flux (W m)"),
            (
                self.heating,
                self.neutron_wall_load,
                self.photon_heat_flux,
            ),
        ):
            ret_str = f"{ret_str}\n{title}\n{self._tabulate(data)}"

        if self.stochastic_cell_volumes is not None:
            ret_str += f"\nStochastic Cell Volumes (m^3) \n{self._tabulate(self.stochastic_cell_volumes)}"
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
    cells: Dict[str, Union[List[openmc.Cell], openmc.Cell]],
    tokamak_geometry: TokamakGeometry,
) -> None:
    """
    Uses the OpenMC plotter to produce an image of the modelled geometry

    Parameters
    ----------
    cells:
        dictionary where each item is either a single openmc.Cell,
            or a list of openmc.Cell.
    tokamak_geometry: TokamakGeometry

    Returns
    -------
    Saves the plots to png files.
    Saves the plots to xml files.

    """
    # Assigning colours for plots
    cell_color_assignment = {
        cells.tf_coil: "brown",
        cells.plasma.inner1: "dimgrey",
        cells.plasma.inner2: "grey",
        cells.plasma.outer1: "darkgrey",
        cells.plasma.outer2: "dimgrey",
        cells.divertor.inner1: "grey",
        cells.divertor.inner2: "dimgrey",
        cells.outer_vessel: "white",
        cells.inboard.vv[0]: "red",
        cells.outboard.vv[1]: "orange",
        cells.outboard.vv[2]: "yellow",
    }

    mat_color_assignment = {
        cells.bore: "blue",
        cells.tf_coil: "brown",
        cells.plasma.inner1: "white",
        cells.plasma.inner2: "white",
        cells.plasma.outer1: "white",
        cells.plasma.outer2: "white",
        cells.divertor.inner1: "white",
        cells.divertor.inner2: "white",
        cells.divertor.fw: "red",
        cells.outer_vessel: "white",
        cells.outer_container: "darkgrey",
    }

    def color_cells(cell, ctype, color):
        for c in getattr(getattr(cells, cell), ctype):
            mat_color_assignment[c] = color

    # first wall: red
    color_cells("outboard", "fw", "red")
    color_cells("inboard", "fw", "red")
    # breeding zone: yellow
    color_cells("outboard", "bz", "yellow")
    color_cells("inboard", "bz", "yellow")
    # manifold: green
    color_cells("outboard", "mani", "green")
    color_cells("inboard", "mani", "green")
    # vacuum vessel: grey
    color_cells("outboard", "vv", "grey")
    color_cells("inboard", "vv", "grey")
    # divertor: cyan
    color_cells("divertor", "regions", "cyan")

    plot_width = 2 * (
        tokamak_geometry.cgs.major_r
        + tokamak_geometry.cgs.minor_r * tokamak_geometry.cgs.elong
        + tokamak_geometry.cgs.outb_fw_thick
        + tokamak_geometry.cgs.outb_bz_thick
        + tokamak_geometry.cgs.outb_mnfld_thick
        + tokamak_geometry.cgs.outb_vv_thick
        + 200.0  # margin
    )

    plot_list = []
    for _, basis in enumerate(("xz", "xy", "yz")):
        plot = openmc.Plot()
        plot.basis = basis
        plot.pixels = [400, 400]
        plot.width = (plot_width, plot_width)
        if basis == "yz":
            plot.colors = cell_color_assignment
        else:
            plot.colors = mat_color_assignment
        plot.filename = f"./out_plots_{basis}"

        plot_list.append(plot)

    openmc.Plots(plot_list).export_to_xml()
    openmc.plot_geometry(output=False)
