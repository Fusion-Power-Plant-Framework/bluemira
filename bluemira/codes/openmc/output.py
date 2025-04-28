# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Functions to present the results prettily
(Including both printed/logged texts and images)
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openmc
from tabulate import tabulate

from bluemira.base.constants import raw_uc
from bluemira.base.look_and_feel import bluemira_debug
from bluemira.radiation_transport.neutronics.constants import DPACoefficients


def get_percent_err(row):
    """
    Calculate a percentage error to the required row,
    assuming cells had been filled out.

    Parameters
    ----------
    row: pd.Series object
        It should have the "mean" and "std. dev."
        row['mean']: float
        row['std. dev.']: float

    Returns
    -------
    fractional_error: float

    Usage
    -----
    dataframe.apply(get_percent_err),
    where dataframe must have one row named "std. dev." and another named "mean".
    """
    # if percentage error > 1E7:)
    if np.isclose(row["mean"], 0.0, rtol=0.0, atol=row["std. dev."] / 100000):
        return np.nan
    # else: normal mode of operation: divide std by mean, then multiply by 100.
    return row["std. dev."] / row["mean"] * 100.0


@dataclass
class OpenMCResult:
    """
    Class that looks opens up the openmc universe from the statepoint file,
        so that the dataframes containing the relevant results
        can be generated and reformatted by its methods.
    """

    tbr: float
    tbr_err: float
    heating: dict
    neutron_wall_load: dict
    blanket_power: float
    blanket_power_err: float
    divertor_power: float
    divertor_power_err: float
    vessel_power: float
    vessel_power_err: float
    """Neutron wall load (eV)"""

    photon_heat_flux: dict
    """Photon heat flux"""

    universe: openmc.Universe
    src_rate: float
    statepoint: openmc.StatePoint
    statepoint_file: str
    cell_names: dict
    cell_vols: dict  # [m^3]
    mat_names: dict

    @classmethod
    def from_run(
        cls,
        universe: openmc.Universe,
        src_rate: float,
        statepoint_file: str = "",
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
        # provided by openmc in cm^3, but we want to save it in m^3
        cell_vols = {
            cell_id: raw_uc(universe.cells[cell_id].volume, "cm^3", "m^3")
            if isinstance(universe.cells[cell_id].volume, float)
            else None
            for cell_id in universe.cells
        }
        # Loads up the output file from the simulation
        statepoint = openmc.StatePoint(statepoint_file)
        tbr, tbr_err = cls._load_tbr(statepoint)
        blanket_power, blanket_power_err = cls._load_filter_power_err(
            statepoint, src_rate, "breeding blanket power"
        )
        divertor_power, divertor_power_err = cls._load_filter_power_err(
            statepoint, src_rate, "divertor power"
        )
        vessel_power, vessel_power_err = cls._load_filter_power_err(
            statepoint, src_rate, "vacuum vessel power"
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
            blanket_power=blanket_power,
            blanket_power_err=blanket_power_err,
            divertor_power=divertor_power,
            divertor_power_err=divertor_power_err,
            vessel_power=vessel_power,
            vessel_power_err=vessel_power_err,
            neutron_wall_load=cls._load_neutron_wall_loading(
                statepoint, cell_names, cell_vols, src_rate
            ),
            photon_heat_flux=cls._load_photon_heat_flux(
                statepoint, cell_names, cell_vols, src_rate
            ),
        )

    @staticmethod
    def _load_volume_calculation_from_file(
        volume_file_path: Path, cell_names: list[str]
    ):
        """
        Load the volume file to record as volume information.

        Parameters
        ----------
        volume_file_path

        Cell_names
            indicative names to print.
        """
        if volume_file_path.is_file():
            vol_results = openmc.VolumeCalculation.from_hdf5(volume_file_path)
            vols = vol_results.volumes
            ids = list(vols.keys())
            cell_volumes = {
                "cell": ids,
                "cell_names": [cell_names[i] for i in ids],
                "Stochastic Volumes": list(raw_uc(list(vols.values()), "cm^3", "m^3")),
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
    def _convert_dict_contents(dataset: dict[str, dict[int, list[str | float]]]):
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
    def _load_filter_power_err(
        cls, statepoint, src_rate: float, filter_name: str
    ) -> tuple[float, float]:
        df = cls._load_dataframe_from_statepoint(statepoint, filter_name)
        power = raw_uc(df["mean"].to_numpy() * src_rate, "eV/s", "W")
        error = raw_uc(df["std. dev."].to_numpy() * src_rate, "eV/s", "W")
        return power.sum(), 100 * (error / power).mean()

    @classmethod
    def _load_heating(cls, statepoint, mat_names, src_rate):
        """Load the heating (sorted by material) dataframe"""
        # mean and std. dev. are given in eV per source particle,
        # so we don't need to show them to the user.
        heating_df = cls._load_dataframe_from_statepoint(statepoint, "Total power")
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
        n_wl_df = cls._load_dataframe_from_statepoint(
            statepoint, "neutron flux in every cell"
        )
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
            p_hf_df["cell_name"].str.contains("Surface"), "heating (W)"
        ].sum()
        cell_total = p_hf_df.loc[
            ~p_hf_df["cell_name"].str.contains("Surface"), "heating (W)"
        ].sum()

        surface_factor = surface_total / cell_total
        # in-place modification
        p_hf_df["vol. heating (W/m3)"] = np.where(
            ~p_hf_df["cell_name"].str.contains(
                "Surface"
            ),  # modify the matching entries,
            p_hf_df["heating (W)"] * surface_factor,
            p_hf_df["heating (W)"],  # otherwise leave it unchanged.
        )
        p_hf_df["vol. heating (W/m3)"] = np.where(
            ~p_hf_df["cell_name"].str.contains("Surface"),
            p_hf_df["heating std.dev."] * surface_factor,
            p_hf_df["heating std.dev."],
        )
        p_hf_df["vol. heating (W/m3)"] = np.where(
            ~p_hf_df["cell_name"].str.contains("Surface"),
            p_hf_df["vol. heating (W/m3)"] * surface_factor,
            p_hf_df["vol. heating (W/m3)"],
        )
        p_hf_df["vol. heating (W/m3)"] = np.where(
            ~p_hf_df["cell_name"].str.contains("Surface"),
            p_hf_df["vol. heating std.dev."] * surface_factor,
            p_hf_df["vol. heating std.dev."],
        )
        # DataFrame columns rearrangement
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
            strict=True,
        ):
            ret_str += f"\n{title}\n{self._tabulate(data)}"

        return ret_str

    @staticmethod
    def _tabulate(
        records: dict[str, str | float],
        tablefmt: str = "fancy_grid",
        floatfmt: str = ".3g",
    ) -> str:
        return tabulate(
            zip(*records.values(), strict=False),
            headers=records.keys(),
            tablefmt=tablefmt,
            showindex=False,
            numalign="right",
            floatfmt=floatfmt,
        )
