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
import pandas as pd
from tabulate import tabulate

from bluemira.base.constants import raw_uc
from bluemira.base.look_and_feel import bluemira_debug
from bluemira.base.parameter_frame._frame import ParameterFrame
from bluemira.base.parameter_frame._parameter import Parameter
from bluemira.codes.openmc.make_csg import CellStage
from bluemira.geometry.tools import revolve_shape
from bluemira.plasma_physics.reactions import n_DT_reactions
from bluemira.radiation_transport.neutronics.make_pre_cell import (
    DivertorPreCellArray,
    PreCellArray,
)
from bluemira.radiation_transport.neutronics.neutronics_axisymmetric import (
    NeutronicsReactor,
)
from bluemira.radiation_transport.neutronics.zero_d_neutronics import (
    ZeroDNeutronicsResult,
)


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


def extract_plasma_facing_areas(
    cell_arrays: CellStage,
    blanket_array: PreCellArray,
    divertor_array: DivertorPreCellArray,
) -> dict[int, str]:
    """
    Extract a dictionary that stores the plasma-facing area of each plasma-facing cell,
    according to the cell id in openmc.
    """
    area_dict = {}
    for blanket_cell_stack, pre_cell in zip(
        cell_arrays.blanket, blanket_array, strict=False
    ):
        plasma_facing_area = revolve_shape(pre_cell.interior_wire, degree=360).area
        area_dict[blanket_cell_stack[0].id] = plasma_facing_area
    for divertor_cell_stack, div_pre_cell in zip(
        cell_arrays.divertor, divertor_array, strict=False
    ):
        plasma_facing_area = revolve_shape(
            div_pre_cell.interior_wire.restore_to_wire(), degree=360
        ).area
        area_dict[divertor_cell_stack[0].id] = plasma_facing_area
    return area_dict


@dataclass
class OpenMCResult:
    """
    Class that looks opens up the openmc universe from the statepoint file,
        so that the dataframes containing the relevant results
        can be generated and reformatted by its methods.
    """

    tbr: float
    tbr_err: float
    e_mult: float
    e_mult_err: float
    heating: dict
    neutron_wall_load: dict
    blanket_power: float
    blanket_power_err: float
    divertor_power: float
    divertor_power_err: float
    vessel_power: float
    vessel_power_err: float
    total_power: float
    total_power_err: float
    mult_power: float
    """Fluxes and neutron wall loads"""
    fluxes: dict
    neutron_wall_load: dict

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
        pre_cell_model: NeutronicsReactor,
        cell_arrays: CellStage,
        P_fus_DT: float,
        statepoint_file: str = "",
    ):
        """Create results class from run statepoint"""
        src_rate = n_DT_reactions(P_fus_DT)
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

        # MC: There is power in the TF + CS, and probably the radiation shield
        # that I am ignoring here. Perhaps worth adding filters for these
        total_power, total_power_err = cls._load_filter_power_err(
            statepoint, src_rate, "total power"
        )

        dt_neuton_power = 0.8 * P_fus_DT
        e_mult = total_power / dt_neuton_power
        e_mult_err = total_power_err / dt_neuton_power
        mult_power = (e_mult - 1.0) * dt_neuton_power
        all_fluxes = cls._load_fluxes(statepoint, cell_names, cell_vols, src_rate)
        cell_plasma_facing_area = extract_plasma_facing_areas(
            cell_arrays, pre_cell_model.blanket, pre_cell_model.divertor
        )

        nwl_df = (
            cls._load_neutron_wall_loading(
                statepoint, cell_names, cell_vols, cell_plasma_facing_area, src_rate
            ),
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
            e_mult=e_mult,
            e_mult_err=e_mult_err,
            heating=cls._load_heating(statepoint, mat_names, src_rate),
            total_power=total_power,
            total_power_err=total_power_err,
            blanket_power=blanket_power,
            blanket_power_err=blanket_power_err,
            divertor_power=divertor_power,
            divertor_power_err=divertor_power_err,
            vessel_power=vessel_power,
            vessel_power_err=vessel_power_err,
            fluxes=all_fluxes,
            neutron_wall_load=nwl_df,
            mult_power=mult_power,
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
    def _load_dataframe_from_statepoint(
        statepoint: openmc.StatePoint, tally_name: str
    ) -> pd.DataFrame:
        return statepoint.get_tally(name=tally_name).get_pandas_dataframe()

    @staticmethod
    def _convert_dict_contents(dataset: dict[str, dict[int, list[str | float]]]):
        for k, v in dataset.items():
            vals = list(v.values()) if isinstance(v, dict) else v
            dataset[k] = vals if isinstance(vals[0], str) else np.array(vals)
        return dataset

    @classmethod
    def _load_tbr(cls, statepoint):
        """
        Load the TBR value and uncertainty.

        Returns
        -------
        mean:
            average TBR, i.e. average (n,Xt) per source particle.
        error:
            absolute error, but since the table is only 1 row long, we can turn the array
            into a float by .sum().
        """
        tbr_df = cls._load_dataframe_from_statepoint(statepoint, "TBR")
        return tbr_df["mean"].sum(), np.sqrt((tbr_df["std. dev."] ** 2).sum())

    @classmethod
    def _load_filter_power_err(
        cls, statepoint, src_rate: float, filter_name: str
    ) -> tuple[float, float]:
        """
        Power is initially loaded as eV/source particle. To convert to Watt, we need the
        source particle rate.

        Parameters
        ----------
        filter_name:
            the literal name that was used in tallying.py to refer to this tally.
        src_rate:
            source particle rate.

        Returns
        -------
        power:
            The total power [W].
        errors:
            The absolute error on the total power [W]. RMS of errors from each cell.
        """
        df = cls._load_dataframe_from_statepoint(statepoint, filter_name)
        powers = raw_uc(df["mean"].to_numpy() * src_rate, "eV/s", "W")
        errors = raw_uc(df["std. dev."].to_numpy() * src_rate, "eV/s", "W")
        return powers.sum(), np.sqrt((errors**2).sum())

    @classmethod
    def _load_heating(cls, statepoint, mat_names, src_rate):
        """Load the heating (sorted by material) dataframe"""
        # mean and std. dev. are given in eV per source particle,
        # so we don't need to show them to the user.
        heating_df = cls._load_dataframe_from_statepoint(statepoint, "total power")
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
                "mean(W)",
                "err.",
                "%err.",
            ]
        ]
        hdf = heating_df.to_dict()
        return cls._convert_dict_contents(hdf)

    @classmethod
    def _load_fluxes(cls, statepoint, cell_names, cell_vols, src_rate):
        """Load the neutron fluxes dataframe."""
        flux_df = cls._load_dataframe_from_statepoint(
            statepoint, "neutron flux in every cell"
        )  # this loads the track length per cell, given in [particle-cm].
        flux_df["cell_name"] = flux_df["cell"].map(cell_names)
        flux_df["vol (m^3)"] = flux_df["cell"].map(cell_vols)
        flux_df["flux (m^-2)"] = (
            raw_uc(flux_df["mean"].to_numpy(), "cm", "m")
            * src_rate
            / flux_df["vol (m^3)"]
        )
        flux_df["%err."] = flux_df.apply(get_percent_err, axis=1)

        flux_df = flux_df[
            [
                "cell",
                "cell_name",
                "particle",
                "nuclide",
                "vol (m^3)",
                "flux (m^-2)",
                "%err.",
            ]
        ]
        return cls._convert_dict_contents(flux_df.to_dict())

    @classmethod
    def _load_neutron_wall_loading(
        cls, statepoint, cell_names, cell_vols, cell_plasma_facing_area, src_rate
    ):
        """
        Notes
        -----
        Note that thinner components will have lower [W/m^2] value, because we
        are measuring ONLY volumetric heating; All photonic heating directly
        from the plasma (e.g. infrared, Brehmsstralung, etc.) are all omitted, because
        this is a neutron simulation code. (But Brehmsstralung generated within the
        material is accounted for.)
        """
        flux = cls._load_dataframe_from_statepoint(statepoint, "neutron flux at PFS")
        flux["vol (m^3)"] = flux["cell"].map(cell_vols)
        flux["flux (m^-2)"] = (
            raw_uc(flux["mean"].to_numpy(), "cm", "m") * src_rate / flux["vol (m^3)"]
        )
        flux["flux %err."] = flux.apply(get_percent_err, axis=1)

        heating = cls._load_dataframe_from_statepoint(
            statepoint, "volumetric heating at PFS"
        )
        heating["cell_name"] = heating["cell"].map(cell_names)
        heating["area (m^2)"] = heating["cell"].map(cell_plasma_facing_area)
        heating["heating (W)"] = raw_uc(
            heating["mean"].to_numpy() * src_rate, "eV/s", "W"
        )
        heating["neutron wall load (W/m^2)"] = (
            heating["heating (W)"] / heating["area (m^2)"]
        )
        heating["heating %err."] = heating.apply(get_percent_err, axis=1)

        nwl = heating[
            [
                "cell",
                "cell_name",
                "nuclide",
                "neutron wall load (W/m^2)",
                "heating %err.",
            ]
        ].merge(
            flux[["cell", "flux (m^-2)", "flux %err."]],
            on="cell",
        )
        # total number of atomic displacements per second in the cell.
        return cls._convert_dict_contents(nwl.to_dict())

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


@dataclass
class NeutronicsOutputParams(ParameterFrame):
    """
    Neutronics output parameters
    """

    e_mult: Parameter[float]
    TBR: Parameter[float]
    P_n_blanket: Parameter[float]
    P_n_divertor: Parameter[float]
    P_n_vessel: Parameter[float]
    P_n_aux: Parameter[float]
    P_n_e_mult: Parameter[float]
    P_n_decay: Parameter[float]
    peak_NWL: Parameter[float]  # noqa: N815
    peak_bb_iron_dpa_rate: Parameter[float]
    peak_vv_iron_dpa_rate: Parameter[float]
    peak_div_cu_dpa_rate: Parameter[float]

    @classmethod
    def from_openmc_csg_result(cls, result: OpenMCResult):
        """
        Produce output parameters from an OpenMC CSG result
        """
        source = "OpenMC CSG"

        auxiliary_power = (
            result.total_power
            - result.blanket_power
            - result.divertor_power
            - result.vessel_power
        )
        return cls(
            Parameter("e_mult", result.e_mult, unit="", source=source),
            Parameter("TBR", result.tbr, unit="", source=source),
            Parameter("P_n_blanket", result.blanket_power, unit="W", source=source),
            Parameter("P_n_divertor", result.divertor_power, unit="W", source=source),
            Parameter("P_n_vessel", result.vessel_power, unit="W", source=source),
            Parameter("P_n_aux", auxiliary_power, unit="W", source=source),
            Parameter("P_n_e_mult", result.mult_power, unit="W", source=source),
            Parameter("P_n_decay", 0.0, unit="W", source=source),
            # TODO @Ocean: Add these  # noqa: TD003
            Parameter("peak_NWL", 0.0, unit="W/m^2", source=source),
            Parameter("peak_bb_iron_dpa_rate", 0.0, unit="dpa/fpy", source=source),
            Parameter("peak_vv_iron_dpa_rate", 0.0, unit="dpa/fpy", source=source),
            Parameter("peak_div_cu_dpa_rate", 0.0, unit="dpa/fpy", source=source),
        )

    @classmethod
    def from_0d_result(cls, result: ZeroDNeutronicsResult):
        """
        Produce output parameters from simplified 0-D neutronics model
        """
        return cls.from_frame(result)
