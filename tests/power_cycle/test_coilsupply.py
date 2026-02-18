# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Verification of Coil Supply System classes with downsampled data associated
with the results for EU-DEMO that is found in the following IDM link:
    - T. Pomella Lobo, Original and processed power profiles in JSON format
      for baseline DEMO1 2017 (for use in Bluemira & PowerFactory), EUROfusion,
      2024. https://idm.euro-fusion.org/?uid=2RRCMT.

The JSON files in "breakdown.zip" and "pulse_full.zip" from the "processed"
folder should be stored in a 'bluemira_private_data' directory, as indicated
below, otherwise the variable 'data_dir' should be adapted.

(common root)
    ├── bluemira
    │   └── tests
    │       └── power_cycle
    │           ├── test_coilsupply.py          (THIS FILE!)
    │           └── test_data
    │               └──config_coilsupply.json
    └── bluemira-private-data
        └── power_cycle
            └── coilsupply_verification
                ├── coilsupply_data_design.json
                ├── coilsupply_data_breakdown.json
                ├── coilsupply_data_pulse_full.json
                ├── coilsupply_data_pulse_semi.json
                └── coilsupply_data_pulse_trim.json

More details about them can be found in the following IDM report and thesis:
    - A. Ferro, Update of the ITER-like design of CS-PF power supplies
      and FDU, EUROfusion, 2022. https://idm.euro-fusion.org/?uid=2Q6988.
    - F. Lunardon, Studies on the reactive power demand in DEMO and
      mitigation strategies, Università degli Studi di Padova, 2018.
      http://hdl.handle.net/20.500.12608/27012.
"""

# %%
from pathlib import Path
from typing import ClassVar

import numpy as np
from matplotlib import (
    colormaps as cmap,
)
from matplotlib import (
    pyplot as plt,
)

from bluemira.base.file import get_bluemira_root
from bluemira.power_cycle.coilsupply import CoilSupplyInputs, CoilSupplySystem
from bluemira.power_cycle.tools import (
    match_domains,
    pp,
    read_json,
    rms_deviation,
    symmetrical_subplot_distribution,
)

script_dir = Path(__file__).resolve().parent
config_dir = script_dir / "test_data"
root_dir = Path(get_bluemira_root())
private_data_dir = root_dir.parent / "bluemira-private-data"
data_dir = private_data_dir / "power_cycle" / "coilsupply_verification"

# Select which pulse data file to use for the pulse verification.
# (full = original, trim = downsampled, semi = original breakdown + downsampled rest)
# data_type = "trim"
# data_type = "semi"
data_type = "full"

verification_dict = {
    "full": {"active": 0.03, "reactive": 0.27},
    "semi": {"active": 0.29, "reactive": 0.29},
    "trim": {"active": 0.78, "reactive": 0.17},
}


class _PlotOptions:
    title_figure = f"coilsupply_verification_{data_type}"
    title_time = "Time (s)"
    title_voltage = "Voltage (V)"
    title_current = "Current (A)"
    title_active = "Active Power (W)"
    title_reactive = "Reactive Power (var)"

    scale_voltage = "k"
    scale_current = "k"
    scale_active = "G"
    scale_reactive = "G"

    colormap_choice = "cool"
    line_width = 2
    ax_left_color = (0.0, 0.0, 1.0)
    ax_right_color = (1.0, 0.0, 0.0)
    default_subplot_size = 4.5
    shade_color_factor = 0.5
    shade_alpha = 0.2
    shade_zorder = -100
    shade_hatch = "/" * 3
    fancy_legend: ClassVar = {
        "loc": "upper center",
        "bbox_to_anchor": (0.5, 1.2),
        "ncol": 6,
        "fancybox": True,
        "shadow": True,
    }
    default_format = "svg"
    show_block = False

    @property
    def _line_thin(self):
        return self.line_width

    @property
    def _line_thick(self):
        return 1 + self.line_width

    def _make_colormap(self, n):
        return cmap[self.colormap_choice].resampled(n)

    def _get_scale_factor(self, scale):
        scale_map = {"k": 1e3, "M": 1e6, "G": 1e9}
        return scale_map.get(scale, 1)

    def _get_param_dict(self, dict_type=None):
        if dict_type == "scale":
            return {
                "voltage": self.scale_voltage,
                "current": self.scale_current,
                "active": self.scale_active,
                "reactive": self.scale_reactive,
            }
        if dict_type == "title":
            return {
                "voltage": self.title_voltage,
                "current": self.title_current,
                "active": self.title_active,
                "reactive": self.title_reactive,
            }
        raise ValueError(f"Unknown param dict type: {dict_type}")

    def _get_param_scale(self, param):
        param_to_scale = self._get_param_dict(dict_type="scale")
        return param_to_scale.get(param, "")

    def _get_param_from_variable(self, variable):
        param_to_scale = self._get_param_dict(dict_type="scale")
        for param in param_to_scale:
            if param in variable:
                return param
        raise ValueError(f"Could not infer parameter from variable: {variable}")

    def _get_param_title(self, param):
        param_to_title = self._get_param_dict(dict_type="title")
        if param not in param_to_title:
            raise ValueError(f"Unknown parameter for title: {param}")
        return param_to_title[param]

    def _scale_data(self, data, param):
        scale = self._get_param_scale(param)
        factor = self._get_scale_factor(scale)
        return np.array(data) / factor

    def _get_scaled_title(self, param):
        base_title = self._get_param_title(param)
        scale = self._get_param_scale(param)
        if (
            scale
            and scale in {"k", "M", "G"}
            and "(" in base_title
            and ")" in base_title
        ):
            param_name = base_title[: base_title.find("(")].strip()
            unit = base_title[base_title.find("(") + 1 : base_title.find(")")].strip()
            return f"{param_name} ({scale}{unit})"
        return base_title

    def _side_color(self, side):
        if side == "left":
            return self.ax_left_color
        if side == "right":
            return self.ax_right_color
        return None

    def _color_yaxis(self, ax, side):
        color = self._side_color(side)
        ax.yaxis.label.set_color(color)
        ax.spines[side].set_color(color)
        ax.tick_params(axis="y", labelcolor=color)

    def _create_twin_ax(self, axs):
        both_axs = {}
        both_axs["left"] = axs.flatten()
        both_axs["right"] = []
        for ax in both_axs["left"]:
            both_axs["right"].append(ax.twinx())
        return both_axs

    def _prepare_ax(
        self,
        ax=None,
        *,
        title=None,
        x_title=None,
        y_title=None,
        side=None,
    ):
        if ax is None:
            ax = plt.axes()

        if side:
            ax.grid(visible=True, axis="x")
            self._color_yaxis(ax, side)
        else:
            ax.grid(visible=True)

        if y_title:
            ax.set_ylabel(y_title)

        if x_title:
            ax.set_xlabel(x_title)
        else:
            ax.set_xlabel(self.title_time)

        if title:
            ax.title.set_text(title)
        return ax

    def _constrained_fig_size(
        self,
        n_rows,
        n_cols,
        row_size=None,
        col_size=None,
    ):
        row_size = self.default_subplot_size if row_size is None else row_size
        col_size = self.default_subplot_size if col_size is None else col_size
        return (col_size * n_cols, row_size * n_rows)

    def _darken_color(self, color, shade=None):
        shade = self.shade_color_factor if shade is None else shade
        return tuple(c * shade for c in color)

    def _shade_background(self, ax, x_vector, x_range, *, hatch=False, label=None):
        ind_in_range = [
            i for i, x in enumerate(x_vector) if x_range[0] <= x <= x_range[1]
        ]
        ind_first = ind_in_range[0]
        ind_last = ind_in_range[-1]
        ax.axvspan(
            x_vector[ind_first],
            x_vector[ind_last],
            facecolor=str(self.shade_color_factor),
            edgecolor="k",
            alpha=self.shade_alpha,
            zorder=self.shade_zorder,
            hatch=self.shade_hatch if hatch else None,
            label=label,
        )
        return ax

    def _save_fig(self, fig, fname, fpath=None, extra_format=None):
        fig_name = f"{self.title_figure}_{fname}"
        save_path = script_dir / "test_figures" / data_type if fpath is None else fpath
        save_path.mkdir(exist_ok=True)
        if extra_format is not None:
            fig.savefig(
                fname=save_path / f"{fig_name}.{extra_format}",
                format=extra_format,
                transparent=True,
            )
        fig.savefig(
            fname=save_path / f"{fig_name}.{self.default_format}",
            format=self.default_format,
            transparent=True,
        )


options = _PlotOptions()

# %% [markdown]
# # Import default Coil Supply System data
#
# The Power Supply for DEMO coils is composed of a main converter device,
# potentially based on the technology of Thyristor Bridges (THY), and
# some auxiliary components:
#   - Protective Make Switch (PMS): isolates the coils from the supply
#                                   in case of component fault
#   - Fast Discharging Unit (FDU): quench protector for the superconducting
#                                  coils
#   - Switiching Network Unit (SNU): devices connected in series to reduce
#                                    voltage consumed during breakdown
#
# We can create the basic arguments to set a Coil Supply System up
# based on the EU-DEMO, using the example inputs stored in the file
# `config_coilsupply.json`.
#

# %%
config_path = config_dir / "config_coilsupply.json"
coilsupply_data = read_json(config_path)

coilsupply_config = coilsupply_data["coilsupply_config"]
corrector_library = coilsupply_data["corrector_library"]
converter_library = coilsupply_data["converter_library"]

# %% [markdown]
# # Import setup data
#
# Design data relevant for EU-DEMO Coil Supply System design, taken from
# IDM, can be found in the `coilsupply_data_design.json` file.

# For this simple example, we import the resistances that characterize
# the SNU for each coil circuit. This data is added to the appropriate
# field in the `corrector_library` variable.
#

# %%
design_path = data_dir / "coilsupply_data_design.json"
design_data = read_json(design_path)

snu_resistances = {}
coil_times = {}
for coil in design_data:
    snu_resistances[coil] = design_data[coil]["SNU_resistance"]
coil_names = list(design_data.keys())

coilsupply_config["coil_names"] = coil_names
corrector_library["SNU"]["resistance_set"] = snu_resistances

# %% [markdown]
# # Initialize CoilSupplySystem
#
# A `CoilSupplySystem` instance can then be initialized.
#
# This is done by first initializing a `CoilSupplyInputs` instance.
# Notice that it pre-processes the argument variables into the required
# formats. For example, scalar inputs for correctors are transformed
# into dictionaries with the appropriate coil names, based on the
# `coil_names` list in the `coilsupply_config` variable.
#
# Then, it is used to initialize the desired `CoilSupplySystem` instance.
#

# %%
coilsupply_inputs = CoilSupplyInputs(
    config=coilsupply_config,
    corrector_library=corrector_library,
    converter_library=converter_library,
)

coilsupply = CoilSupplySystem(coilsupply_inputs)


def display_inputs(coilsupply, summary):
    """Print Coil Supply System inputs."""
    pp(coilsupply.inputs, summary=summary)


def display_subsystems(coilsupply, summary):
    """Print summary of Coil Supply System subsystems."""
    correctors_summary = {
        c.name: {
            "class": type(c),
            "resistance": c.resistance_set,
        }
        for c in coilsupply.correctors
    }
    converters_summary = {
        c.name: {
            "class": type(c),
            "v_bridge_arg": c.max_bridge_voltage,
            "power_loss_arg": c.power_loss_percentages,
        }
        for c in [coilsupply.converter]
    }
    pp(correctors_summary, summary=summary)
    pp(converters_summary, summary=summary)


# %% [markdown]
# # Verification with breakdown data
#
# The IDM data for the behavior of correctors and converters during
# breakdown is stored in the `coilsupply_data_breakdown.json` file. We
# import it to use:
#   - coil voltages and currents as simulation inputs;
#   - SNU and THY voltages as output verification.
#
# Voltages and currents are first re-ordered and later used as arguments
# and data for comparison.
#
# The figure displays how computed voltages for the SNU and THY (in color)
# coincide with the same values in the original data (in black). The
# computation arguments (coil currents, in red, and voltages, in shades
# of blue) are also plotted against the original data (in black).
#

# %%
breakdown_path = data_dir / "coilsupply_data_breakdown.json"
breakdown_data = read_json(breakdown_path)
t_start_breakdown = 500


def prepare_breakdown_verification(breakdown_data, t_start_breakdown):
    """Prepare Coil Supply System breakdown data for verification."""
    breakdown_per_coil = {}
    breakdown_reorder_keys = {
        "SNU_voltages": "SNU_voltage",
        "THY_voltages": "THY_voltage",
        "coil_voltages": "coil_voltage",
        "coil_currents": "coil_current",
        "coil_times": "coil_time",
        "SNU_switches": "coil_time",  # use time vector to build switch
    }
    duration_breakdown = []
    for new_key, old_key in breakdown_reorder_keys.items():
        breakdown_per_coil[new_key] = {}
        for coil in breakdown_data:
            old_value = breakdown_data[coil][old_key]
            if new_key == "SNU_switches":
                old_value = [1 for t in old_value]  # SNU always on
            if new_key == "coil_times":
                duration_breakdown.append(max(old_value))
            breakdown_per_coil[new_key][coil] = old_value
    duration_breakdown = max(duration_breakdown)
    breakdown_wallplug = coilsupply.compute_wallplug_loads(
        breakdown_per_coil["coil_voltages"],
        breakdown_per_coil["coil_currents"],
        breakdown_per_coil["coil_times"],
        {"SNU": breakdown_per_coil["SNU_switches"]},
        verbose=False,
    )
    t_range_breakdown = (
        t_start_breakdown,
        t_start_breakdown + duration_breakdown,
    )
    return breakdown_per_coil, breakdown_wallplug, t_range_breakdown


def plot_breakdown_verification(breakdown_data, t_start_breakdown):
    """Plot Coil Supply System verification for breakdown data."""
    (
        breakdown_per_coil,
        breakdown_wallplug,
        t_range_breakdown,
    ) = prepare_breakdown_verification(breakdown_data, t_start_breakdown)

    n_plots = len(breakdown_wallplug)
    n_rows, n_cols = symmetrical_subplot_distribution(
        n_plots,
        direction="col",
    )

    v_labels_and_styles = {
        "coil_voltages": "-",
        "SNU_voltages": "-.",
        "THY_voltages": ":",
    }
    v_colors = options._make_colormap(n_plots)

    fig, axs = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        layout="tight",
        figsize=options._constrained_fig_size(n_rows, n_cols, 3.5, 10.5),
    )
    for plot_index, coil in enumerate(coil_names, start=0):
        wallplug_info = getattr(breakdown_wallplug, coil)

        ax_left = axs.flat[plot_index]
        for label, style in v_labels_and_styles.items():
            ax_left.plot(
                breakdown_per_coil["coil_times"][coil],
                options._scale_data(breakdown_per_coil[label][coil], "voltage"),
                f"{style}",
                linewidth=options._line_thick,
                color="k",
                label=f"_{label}_verification",
            )
            ax_left.plot(
                breakdown_per_coil["coil_times"][coil],
                options._scale_data(wallplug_info[label], "voltage"),
                f"{style}",
                linewidth=options._line_thin,
                color=v_colors(plot_index),
                label=label,
            )
        ax_left.set_ylabel(options._get_scaled_title("voltage"))
        options._color_yaxis(ax_left, "left")
        ax_left.set_xlabel(options.title_time)
        ax_left.grid()

        ax_right = ax_left.twinx()
        ax_right.set_ylabel(options._get_scaled_title("current"))
        options._color_yaxis(ax_right, "right")
        ax_right.plot(
            breakdown_per_coil["coil_times"][coil],
            options._scale_data(breakdown_per_coil["coil_currents"][coil], "current"),
            "-",
            linewidth=options._line_thick,
            color="k",
            label="coil_currents",
        )
        ax_right.plot(
            breakdown_per_coil["coil_times"][coil],
            options._scale_data(wallplug_info["coil_currents"], "current"),
            "-",
            linewidth=options._line_thin,
            color=options.ax_right_color,
            label="coil_currents",
        )

        ax_left.legend(loc="upper right")
        ax_left.title.set_text(coil)

    fig.suptitle(
        "Coil Supply System Model, Breakdown Verification:\n"
        "original (black) X model (color)",
    )
    return fig, t_range_breakdown


def save_breakdown_verification(breakdown_data, t_start_breakdown, fpath=None):
    """Save Coil Supply System verification plots for breakdown data."""
    fig, t_range_breakdown = plot_breakdown_verification(
        breakdown_data,
        t_start_breakdown,
    )
    options._save_fig(
        fig=fig,
        fname="breakdown_BLUEMIRA",
        fpath=fpath,
        extra_format="png",
    )
    plt.show(block=options.show_block)
    return t_range_breakdown


# %% [markdown]
# # Verification with pulse data
#
# A downsampled version of the IDM data for the total active and reactive
# powers, consumed during a full pulse by the Coil Supply System, is stored
# in the `coilsupply_data_pulse.json` file. We import it to use:
#   - coil voltages and currents as simulation inputs;
#   - active and reactive powers consumed by each coil and the full system
#     as output verification.
#
# Voltages, currents are first re-ordered and later used as arguments.
# Returned power values are summed and compared against the original results.
#
# The figures show how the estaimtes of active and reactive power per coil
# largely coincide with the original data, between the start of breakdown and
# the end of ramp-down. This is represented by the calculation of a normalized
# RMS deviation between the curves within that time period.
#
# The deviation is larger for the active power curve, since its values are
# close to zero for most of the pulse duration, which amplifies the effect of
# small differences.
#
# The deviation is smaller for the reactive power curve, but clear mismatches
# occur where the effects of active control systems can be seen in the original
# data, represented by abrupt steps in values. These are not currently emulated
# by the model.
#

# %%
data_file = f"coilsupply_data_pulse_{data_type}.json"
pulse_path = data_dir / data_file
pulse_data = read_json(pulse_path)
t_end_rampdown = 7890


def prepare_pulse_verification(pulse_data, t_range_breakdown):
    """Prepare Coil Supply System pulse data for verification."""
    t_start_breakdown = t_range_breakdown[0]
    t_end_breakdown = t_range_breakdown[1]

    pulse_totals = pulse_data["power"]
    pulse_per_coil = {}
    pulse_reorder_keys = {
        "coil_times": "coil_time",
        "coil_voltages": "coil_voltage",
        "coil_currents": "coil_current",
        "coil_active": "coil_active",
        "coil_reactive": "coil_reactive",
        "SNU_switches": "coil_time",
    }

    for new_key, old_key in pulse_reorder_keys.items():
        pulse_per_coil[new_key] = {}
        for coil in pulse_data["coils"]:
            old_value = pulse_data["coils"][coil].get(old_key, None)
            if new_key == "SNU_switches":
                t_after_start = [t >= t_start_breakdown for t in old_value]
                t_before_end = [t <= t_end_breakdown for t in old_value]
                new_value = [
                    a and b for a, b in zip(t_after_start, t_before_end, strict=False)
                ]
            else:
                new_value = old_value

            if old_value is not None:
                pulse_per_coil[new_key][coil] = new_value

    pulse_wallplug = coilsupply.compute_wallplug_loads(
        pulse_per_coil["coil_voltages"],
        pulse_per_coil["coil_currents"],
        pulse_per_coil["coil_times"],
        {"SNU": pulse_per_coil["SNU_switches"]},
        verbose=False,
    )

    per_coil_keys_to_plot = pulse_per_coil.keys() - {"coil_times", "SNU_switches"}
    coil_subplots_settings = {}
    for key in per_coil_keys_to_plot:
        coil_subplots_settings[key] = {}
        if key in {"coil_voltages", "THY_voltages"}:
            coil_subplots_settings[key]["fig_ind"] = 1
            coil_subplots_settings[key]["side"] = "left"
            coil_subplots_settings[key]["plot_color"] = "map"
            coil_subplots_settings[key]["variable"] = key
        elif key == "coil_currents":
            coil_subplots_settings[key]["fig_ind"] = 1
            coil_subplots_settings[key]["side"] = "right"
            coil_subplots_settings[key]["plot_color"] = "side"
            coil_subplots_settings[key]["variable"] = key
        elif key == "coil_active":
            coil_subplots_settings[key]["fig_ind"] = 2
            coil_subplots_settings[key]["side"] = "left"
            coil_subplots_settings[key]["plot_color"] = "map"
            coil_subplots_settings[key]["variable"] = "power_active"
        elif key == "coil_reactive":
            coil_subplots_settings[key]["fig_ind"] = 2
            coil_subplots_settings[key]["side"] = "right"
            coil_subplots_settings[key]["plot_color"] = "side"
            coil_subplots_settings[key]["variable"] = "power_reactive"
        else:
            raise ValueError(f"Unknown subplot settings for: {key}")
        variable = coil_subplots_settings[key]["variable"]
        param = options._get_param_from_variable(variable)
        coil_subplots_settings[key]["y_title"] = options._get_scaled_title(param)

    total_subplots_settings = {}
    for key in pulse_totals:
        total_subplots_settings[key] = {}
        if key == "total_active":
            total_subplots_settings[key]["side"] = "left"
            total_subplots_settings[key]["plot_color"] = "side"
            total_subplots_settings[key]["variable"] = "power_active"
        elif key == "total_reactive":
            total_subplots_settings[key]["side"] = "right"
            total_subplots_settings[key]["plot_color"] = "side"
            total_subplots_settings[key]["variable"] = "power_reactive"
        else:
            raise ValueError(f"Unknown subplot settings for: {key}")
        variable = total_subplots_settings[key]["variable"]
        param = options._get_param_from_variable(variable)
        total_subplots_settings[key]["y_title"] = options._get_scaled_title(param)
        total_subplots_settings[key]["sum_time"] = []
        total_subplots_settings[key]["sum_power"] = []

    return (
        per_coil_keys_to_plot,
        pulse_per_coil,
        coil_subplots_settings,
        pulse_totals,
        total_subplots_settings,
        pulse_wallplug,
    )


def plot_pulse_verification(
    pulse_data,
    t_range_breakdown,
    t_end_rampdown,
    *,
    phase_plot=False,
):
    """Plot Coil Supply System verification for pulse data."""
    (
        per_coil_keys_to_plot,
        pulse_per_coil,
        coil_subplots_settings,
        pulse_totals,
        total_subplots_settings,
        pulse_wallplug,
    ) = prepare_pulse_verification(pulse_data, t_range_breakdown)
    rms_range = (t_range_breakdown[0], t_end_rampdown)

    n_coils = len(pulse_wallplug)
    coil_colors = options._make_colormap(n_coils)
    n_plots = n_coils + 1  # last subplot for total powers
    n_rows, n_cols = symmetrical_subplot_distribution(
        n_plots,
        direction="col",
    )
    all_figs = {}
    all_axes = {}

    for key in per_coil_keys_to_plot:
        fig_ind, side, y_title, plot_color, variable = (
            coil_subplots_settings[key][name]
            for name in ("fig_ind", "side", "y_title", "plot_color", "variable")
        )

        param = options._get_param_from_variable(variable)
        if fig_ind not in all_figs:
            fig, axs = plt.subplots(
                nrows=n_rows,
                ncols=n_cols,
                layout="tight",
                sharex=True,
                figsize=options._constrained_fig_size(
                    n_rows,
                    n_cols,
                    3.5,
                    10.5,
                ),
            )
            all_figs[fig_ind] = fig
            all_axes[fig_ind] = options._create_twin_ax(axs)
        fig = all_figs[fig_ind]
        axs = all_axes[fig_ind][side]

        plot_index = 0
        for coil in coil_names:
            ax = options._prepare_ax(axs[plot_index], title=coil, side=side)
            ax_color = options._side_color(side)
            color_verification = (
                ax_color if plot_color == "side" else coil_colors(plot_index)
            )
            color_computation = options._darken_color(color_verification)

            wallplug_info = getattr(pulse_wallplug, coil)
            rms, _ = rms_deviation(
                [pulse_per_coil["coil_times"][coil], pulse_per_coil[key][coil]],
                [pulse_per_coil["coil_times"][coil], wallplug_info[variable]],
                x_range=rms_range,
            )
            ax.plot(
                pulse_per_coil["coil_times"][coil],
                options._scale_data(pulse_per_coil[key][coil], param),
                "-",
                linewidth=options._line_thick,
                color=color_verification,
                label=f"_{key}_verification",
            )
            ax.plot(
                pulse_per_coil["coil_times"][coil],
                options._scale_data(wallplug_info[variable], param),
                "--",
                linewidth=options._line_thin,
                color=color_computation,
                label=f"_{key}_computation",
            )

            if side == "right":
                ax = options._shade_background(
                    ax,
                    x_vector=pulse_per_coil["coil_times"][coil],
                    x_range=rms_range,
                    label="Counts towards RMS deviation",
                )
                ax = options._shade_background(
                    ax,
                    x_vector=pulse_per_coil["coil_times"][coil],
                    x_range=t_range_breakdown,
                    hatch=True,
                    label="Breakdown",
                )
            if y_title is not None:
                ax.set_ylabel(f"{y_title} - RMS dev.: {rms:.2%}")
            ax.grid(visible=True, axis="y", linestyle=":", color=ax_color)

            plot_index += 1

    totals_rms = {}
    for key in pulse_totals:
        side, y_title, plot_color, variable, sum_time, sum_power = (
            total_subplots_settings[key][name]
            for name in (
                "side",
                "y_title",
                "plot_color",
                "variable",
                "sum_time",
                "sum_power",
            )
        )

        param = options._get_param_from_variable(variable)
        for coil in coil_names:
            wallplug_info = getattr(pulse_wallplug, coil)
            sum_time.append(pulse_per_coil["coil_times"][coil])
            sum_power.append(wallplug_info[variable])
        sum_time, sum_power = match_domains(sum_time, sum_power)
        sum_power = np.add.reduce(sum_power)
        total_subplots_settings[key]["sum_time"] = sum_time
        total_subplots_settings[key]["sum_power"] = sum_power

        ax_color = options._side_color(side)
        color_verification = (
            ax_color if plot_color == "side" else coil_colors(plot_index)
        )
        color_computation = options._darken_color(color_verification)

        for fig_ind in all_figs:
            last_ax = all_axes[fig_ind][side][-1]
            last_ax = options._prepare_ax(last_ax, title="Totals", side=side)

            totals_rms[key], _ = rms_deviation(
                [pulse_totals[key]["time"], pulse_totals[key]["power"]],
                [sum_time, sum_power],
                x_range=rms_range,
            )
            last_ax.plot(
                pulse_totals[key]["time"],
                options._scale_data(pulse_totals[key]["power"], param),
                "-",
                linewidth=options._line_thick,
                color=color_verification,
                label=f"_{key}_verification",
            )
            last_ax.plot(
                sum_time,
                options._scale_data(sum_power, param),
                "--",
                linewidth=options._line_thin,
                color=color_computation,
                label=f"_{key}_computation",
            )
            if side == "right":
                last_ax = options._shade_background(
                    last_ax,
                    x_vector=sum_time,
                    x_range=rms_range,
                    label="Counts towards RMS deviation",
                )
                last_ax = options._shade_background(
                    last_ax,
                    x_vector=sum_time,
                    x_range=t_range_breakdown,
                    hatch=True,
                    label="Breakdown",
                )
            last_ax.set_ylabel(f"{y_title} - RMS dev.: {totals_rms[key]:.2%}")
            last_ax.grid(visible=True, axis="y", linestyle=":", color=ax_color)

    for fig in all_figs.values():
        fig.suptitle(
            "Coil Supply System Model, Pulse Verification:\n"
            "original (continuous) X model (dashed)\n"
            "breakdown (hatched region), normalized RMS deviation (shaded region)",
        )

    if phase_plot:
        fig = plt.figure()
        ax = options._prepare_ax(
            ax=None,
            title=None,
            x_title="Vector index [-]",
            y_title="Phase (phi) [°]",
            side=None,
        )
        for coil in coil_names:
            wallplug_info = getattr(pulse_wallplug, coil)
            n_units = wallplug_info["number_of_bridge_units"]
            ax.plot(
                wallplug_info["phase_degrees"],
                label=f"{coil} ({n_units} bridge units)",
            )
        plt.legend()
        all_figs["phase"] = fig
        all_axes["phase"] = ax

    return all_figs, all_axes, totals_rms


def plot_standalone_fig(all_axes, fig_index, subplot_index, y_scale=None):
    """
    Extract single subplot from the verification for pulse data.

    Parameters
    ----------
    all_axes: dict
        Dictionary with all axes, produced by 'plot_pulse_verification'.
    fig_index: int
        Index of figure from which to extract subplot.
        Subplot in figures: V,I (1); P,Q (2).
    subplot_index: int
        Index of subplot to be extracted.
        Subplots: CS coils (0-4); PF coils (5-10); totals (11).
    y_scale: str or None
        If "log", sets both y-axes to log scale.
        If "lin", sets both y-axes to linear scale.
        If None, keeps original axis scales.
    """
    n_coils = len(all_axes[fig_index]["left"]) - 1
    coil_colors = options._make_colormap(n_coils)

    standalone_fig = plt.figure()

    standalone_axes = {"left": plt.axes()}
    standalone_axes["right"] = standalone_axes["left"].twinx()

    if y_scale == "log":
        for ax in standalone_axes.values():
            ax.set_yscale("log")
    elif y_scale == "lin":
        for ax in standalone_axes.values():
            ax.set_yscale("linear")

    for side in ["left", "right"]:
        standalone_ax = standalone_axes[side]
        ax = all_axes[fig_index][side][subplot_index]
        ax_title = ax.title.get_text()
        ax_ylabel = ax.get_ylabel()
        ax_color = options._side_color(side)
        if side == "left":
            color_verification = (
                ax_color if subplot_index == 11 else coil_colors(subplot_index)
            )
        else:
            color_verification = ax_color
        color_computation = options._darken_color(color_verification)
        for c, line in enumerate(ax.get_lines()):
            lt = options._line_thin if c == 1 else options._line_thick
            ls = "--" if c == 1 else "-"
            lc = color_computation if c == 1 else color_verification
            line_data = line.get_data()
            standalone_ax.plot(*line_data, ls, color=lc, linewidth=lt)
        options._color_yaxis(standalone_ax, side)
        standalone_ax.grid(visible=True, axis="x")
        standalone_ax.grid(visible=True, axis="y", linestyle=":", color=ax_color)
        standalone_ax.set_ylabel(ax_ylabel)
    standalone_ax.title.set_text(ax_title)
    standalone_ax.set_xlabel(options.title_time)

    standalone_fig.tight_layout(rect=[0, 0, 1, 0.9])
    standalone_fig.suptitle(
        "Coil Supply System Model, Pulse Verification:\n"
        "original (continuous) X model (dashed)",
    )
    return standalone_fig


def save_pulse_verification(
    pulse_data,
    t_range_breakdown,
    t_end_rampdown,
    standalone_indexes=None,
    standalone_scale=None,
    zoom_time_range=None,
    fpath=None,
):
    """Save Coil Supply System verification plots for pulse data."""

    def fig_name(fig_index):
        if fig_index == 1:
            fig_type = "VI"
        elif fig_index == 2:
            fig_type = "PQ"
        elif fig_index == "phase":
            fig_type = "phase"
        else:
            raise ValueError(f"Unknown figure index: {fig_index}")
        return f"pulse_BLUEMIRA_{fig_type}"

    extra_format = "png"

    figs_normal, axes_normal, totals_rms = plot_pulse_verification(
        pulse_data,
        t_range_breakdown,
        t_end_rampdown,
        phase_plot=True,
    )
    for fig_ind, fig in figs_normal.items():
        options._save_fig(
            fig=fig,
            fname=fig_name(fig_ind),
            fpath=fpath,
            extra_format=extra_format,
        )
    plt.show(block=options.show_block)

    fig_index, subplot_index = standalone_indexes
    if fig_index and subplot_index:
        standalone_fig = plot_standalone_fig(
            axes_normal, fig_index, subplot_index, y_scale=standalone_scale
        )
        data = f"fig{fig_index}_sub{subplot_index}"
        options._save_fig(
            fig=standalone_fig,
            fname=f"{fig_name(fig_ind)}_STANDALONE",
            fpath=fpath,
            extra_format=extra_format,
        )
        plt.show(block=options.show_block)

    if not zoom_time_range:
        zoom_time_range = t_range_breakdown
    figs_zoom, axes_zoom, _ = plot_pulse_verification(
        pulse_data.copy(),
        t_range_breakdown,
        t_end_rampdown,
        phase_plot=False,
    )
    for fig_ind, fig in figs_zoom.items():
        not_phase_fig = isinstance(fig_ind, int)
        if not_phase_fig:
            last_ax = axes_zoom[fig_ind]["left"][-1]
            dx = 0.05 * np.diff(zoom_time_range)
            zoom_limits = [zoom_time_range[0] - dx, zoom_time_range[1] + dx]
            last_ax.set_xlim(zoom_limits)
            options._save_fig(
                fig=fig,
                fname=f"{fig_name(fig_ind)}_ZOOM",
                fpath=fpath,
                extra_format=extra_format,
            )
    plt.show(block=options.show_block)

    return totals_rms


# %% [markdown]
# # Run test
#


def test_CoilSupplySystem(tmp_path, data_type):
    display_inputs(coilsupply, summary=False)
    display_subsystems(coilsupply, summary=True)
    t_range_breakdown = save_breakdown_verification(
        breakdown_data,
        t_start_breakdown,
        fpath=tmp_path,
    )
    totals_rms = save_pulse_verification(
        pulse_data,
        t_range_breakdown,
        t_end_rampdown,
        standalone_indexes=(1, 11),
        standalone_scale="lin",
        zoom_time_range=None,
        fpath=tmp_path,
    )
    assert totals_rms["total_active"] < verification_dict[data_type]["active"]
    assert totals_rms["total_reactive"] < verification_dict[data_type]["reactive"]
    plt.show()


if __name__ == "__main__":
    test_CoilSupplySystem(None, data_type=data_type)
