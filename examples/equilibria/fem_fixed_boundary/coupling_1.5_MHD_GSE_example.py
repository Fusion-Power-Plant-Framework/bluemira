# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% tags=["remove-cell"]
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

"""
An example that shows how to set up the problem for the fixed boundary equilibrium.
"""

# %%
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np

from bluemira.base.file import get_bluemira_path, get_bluemira_root
from bluemira.base.logs import set_log_level
from bluemira.codes import transport_code_solver
from bluemira.equilibria.fem_fixed_boundary.equilibrium import (
    solve_transport_fixed_boundary,
)
from bluemira.equilibria.fem_fixed_boundary.fem_magnetostatic_2D import (
    FemGradShafranovFixedBoundary,
)
from bluemira.equilibria.fem_fixed_boundary.file import save_fixed_boundary_to_file
from bluemira.equilibria.shapes import JohnerLCFS

from typing import Union, Tuple, Optional
from matplotlib.axes._axes import Axes

set_log_level("NOTSET")



class ScalarField:
    """A class to simplify operations on a scalar field"""

    def __init__(self, points: np.ndarray, data: np.ndarray, label: str = ""):
        self._max = None
        self._min = None
        self.data = data
        self.points = points
        self.label = label

    @property
    def points(self) -> np.ndarray:
        return self._points

    @points.setter
    def points(self, arr):
        self._points = arr

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, arr):
        self._data = arr
        # reassign max and min
        self._max = None
        self._min = None

    @property
    def max(self):
        if self._max is None:
            self._max = np.max(self.data)
        return self._max

    @property
    def min(self):
        if self._min is None:
            self._min = np.min(self.data)
        return self._min

    def rescale_min_max(self, new_min, new_max):
        field = deepcopy(self)
        field.data = new_min + (self.data - self.min) * (
            (new_max - new_min) / (self.max - self.min)
        )
        return field

    @property
    def dim(self):
        return self.points.shape[1]

    def transform(self, function: callable):
        field = deepcopy(self)
        field.data = np.array([function(x) for x in self.data])
        return field

    def __repr__(self) -> str:
        """
        The string representation of the instance
        """
        return f"{self.label}: dim = {self.dim}D, [ {self.min} : {self.max} ])"


class ScalarField1D(ScalarField):
    """A class to simplify operations on a scalar field"""

    def interp1D(self, *args, **kwargs):
        return scipy.interpolate.interp1d(*args, **kwargs)

    def plot(
        self,
        ax: Optional[Axes] = None,
        show: bool = True,
    ):
        """
        Plot a 1D scalar field
        """
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(self.points, self.data)
        ax.grid()
        if show:
            plt.show()
        return ax


class ScalarField2D(ScalarField):
    """A class to simplify operations on a scalar field"""

    def plot(
        self,
        levels: int = 20,
        ax: Optional[Axes] = None,
        contour: bool = True,
        tofill: bool = True,
        show: bool = True,
        **kwargs,
    ) -> Tuple[Axes, Union[Axes, None], Union[Axes, None]]:
        """
        Plot a 2D scalar field

        Parameters
        ----------
        levels: int
            Number of contour levels to plot
        axis: Optional[Axis]
            axis onto which to plot
        contour: bool
            Whether or not to plot contour lines
        tofill: bool
            Whether or not to plot filled contours

        Returns
        -------
        axis: Axis
            Matplotlib axis on which the plot ocurred
        """
        x = self.points[:, 0]
        y = self.points[:, 1]
        data = self.data

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        defaults = {"linewidths": 2, "colors": "k"}
        contour_kwargs = {**defaults, **kwargs}

        cntr = None
        cntrf = None

        if contour:
            cntr = ax.tricontour(x, y, data, levels=levels, **contour_kwargs)

        if tofill:
            cntrf = ax.tricontourf(x, y, data, levels=levels, cmap="RdBu_r")
            fig.colorbar(cntrf, ax=ax)

        ax.set_aspect("equal")

        if show:
            plt.show()

        return ax, cntr, cntrf




# %% [markdown]
#
# # Fixed Boundary Equilibrium
# Setup the Plasma shape parameterisation variables

# %%
johner_parameterisation = JohnerLCFS(
    {
        "r_0": {"value": 8.9830e00},
        "a": {"value": 3.1},
        "kappa_u": {"value": 1.6},
        "kappa_l": {"value": 1.75},
        "delta_u": {"value": 0.33},
        "delta_l": {"value": 0.45},
    }
)

# %% [markdown]
# Initialise the transport solver in this case PLASMOD is used

# %%
if plasmod_binary := shutil.which("plasmod"):
    PLASMOD_PATH = os.path.dirname(plasmod_binary)
else:
    PLASMOD_PATH = os.path.join(os.path.dirname(get_bluemira_root()), "plasmod/bin")
binary = os.path.join(PLASMOD_PATH, "plasmod")


source = "Plasmod Example: coupling 1.5 MHD and GSE"
plasmod_params = {
    "A": {"value": johner_parameterisation.variables.a, "unit": "", "source": source},
    "R_0": {
        "value": johner_parameterisation.variables.r_0,
        "unit": "m",
        "source": source,
    },
    "I_p": {"value": 19e6, "unit": "A", "source": source},
    "B_0": {"value": 5.31, "unit": "T", "source": source},
    "V_p": {"value": -2500, "unit": "m^3", "source": source},
    "v_burn": {"value": -1.0e6, "unit": "V", "source": source},
    "kappa_95": {"value": 1.652, "unit": "", "source": source},
    "delta_95": {"value": 0.333, "unit": "", "source": source},
    "delta": {
        "value": (
            johner_parameterisation.variables.delta_l
            + johner_parameterisation.variables.delta_u
        )
        / 2,
        "unit": "",
        "source": source,
    },
    "kappa": {
        "value": (
            johner_parameterisation.variables.kappa_l
            + johner_parameterisation.variables.kappa_u
        )
        / 2,
        "unit": "",
        "source": source,
    },
    "q_95": {"value": 3.25, "unit": "", "source": source},
    "f_ni": {"value": 0, "unit": "", "source": source},
}

problem_settings = {
    "amin": plasmod_params["R_0"]["value"] / plasmod_params["A"]["value"],
    "pfus_req": 2000.0,
    "pheat_max": 100.0,
    "q_control": 50.0,
    "i_impmodel": "PED_FIXED",
    "i_modeltype": "GYROBOHM_2",
    "i_equiltype": "q95_sawtooth",
    "i_pedestal": "SAARELMA",
    "isawt": "FULLY_RELAXED",
}

plasmod_build_config = {
    "problem_settings": problem_settings,
    "mode": "run",
    "binary": binary,
    "directory": get_bluemira_path("", subfolder="generated_data"),
}

plasmod_solver = transport_code_solver(
    params=plasmod_params,
    build_config=plasmod_build_config,
    module="PLASMOD",
)

# %% [markdown]
# Initialise the FEM problem

# %%
fem_GS_fixed_boundary = FemGradShafranovFixedBoundary(
    p_order=2,
    max_iter=30,
    iter_err_max=1e-2,
    relaxation=0.05,
)

# %% [markdown]
# Solve

# %%

mesh_filename = "coupling_MHD_GSE"
directory = get_bluemira_path("", subfolder="generated_data")
mesh_name_msh = mesh_filename + ".msh"

from bluemira.equilibria.fem_fixed_boundary.equilibrium import (
    PlasmaFixedBoundaryParams,
    TransportSolverParams,
    create_plasma_xz_cross_section,
    create_mesh
)
from copy import deepcopy

parameterisation = johner_parameterisation
transport_solver = plasmod_solver
gs_solver = fem_GS_fixed_boundary
lcar_mesh = 0.3


paramet_params = PlasmaFixedBoundaryParams(
    **{
        k: v
        for k, v in zip(
            parameterisation.variables.names, parameterisation.variables.values
        )
        if k in PlasmaFixedBoundaryParams.fields()
    }
)

transport_params = TransportSolverParams.from_frame(deepcopy(transport_solver.params))

lcfs_options = {
    "face": {"lcar": lcar_mesh, "physical_group": "plasma_face"},
    "lcfs": {"lcar": lcar_mesh, "physical_group": "lcfs"},
}

transport_solver.params.update_from_frame(transport_params)
transp_out_params = transport_solver.execute("run")


x = transport_solver.get_profile("x")
pprime = transport_solver.get_profile("pprime")
ffprime = transport_solver.get_profile("ffprime")
q = transport_solver.get_profile("q")
press = transport_solver.get_profile("pressure")
g2 = transport_solver.get_profile("g2")
g3 = transport_solver.get_profile("g3")


plasmod_profiles_for_plot = {
    "x": transport_solver.get_profile("x"),
    "pprime": transport_solver.get_profile("pprime"),
    "ffprime": transport_solver.get_profile("ffprime"),
    "q": transport_solver.get_profile("q"),
    "press": transport_solver.get_profile("pressure"),
    "g2": transport_solver.get_profile("g2"),
    "g3": transport_solver.get_profile("g3"),
}

kappa_95 = 1.652
delta_95 = 0.333

plasma = create_plasma_xz_cross_section(
    parameterisation,
    transport_params,
    paramet_params,
    kappa_95,
    delta_95,
    lcfs_options,
    f"from equilibrium",
)

mesh = create_mesh(
    plasma,
    directory,
    mesh_filename,
    mesh_name_msh,
)

gs_solver.set_mesh(mesh)

# Start loop for convergence of pprime and ffprime
from scipy.interpolate import interp1d

# first iteration with J constant
points = gs_solver.mesh.coordinates()

f_pprime = interp1d(x, pprime, fill_value="extrapolate")
f_ffprime = interp1d(x, ffprime, fill_value="extrapolate")

gs_solver.set_profiles(
    f_pprime,
    f_ffprime,
    transp_out_params.I_p.value,
    transp_out_params.B_0.value,
    transp_out_params.R_0.value,
)

equil = gs_solver.solve()

x2d_0 = np.array([gs_solver.psi_norm_2d(p) for p in points])

n_iter_max = 10

import bluemira.equilibria.fem_fixed_boundary.equilibrium as equilibrium
import bluemira.equilibria.fem_fixed_boundary.utilities as utilities

q_func = interp1d(x, q, fill_value="extrapolate")
p_func = interp1d(x, press, fill_value="extrapolate")

x1d, flux_surfaces = utilities.get_flux_surfaces_from_mesh(
    mesh, gs_solver.psi_norm_2d, nx=50
)
x1d, V_0, g1_0, g2_0, g3_0 = equilibrium.calc_metric_coefficients(
    flux_surfaces, gs_solver.psi, gs_solver.psi_norm_2d, x1d
)

g2_0_fun = interp1d(x1d, g2_0, fill_value="extrapolate")
g3_0_fun = interp1d(x1d, g3_0, fill_value="extrapolate")
V_0_fun = interp1d(x1d, V_0, fill_value="extrapolate")

Psi_ax_0 = gs_solver.psi_ax
Psi_b_0 = gs_solver.psi_b
theta = 0.5

for n_iter in range(n_iter_max):

    Ip, Phi1D, Psi1D, pprime_psi1D_data, F, FFprime = equilibrium.calc_curr_dens_profiles(
        x1d, p_func(x1d), q_func(x1d), g2_0, g3_0, V_0, 0,
        transp_out_params.B_0.value, transp_out_params.R_0.value, Psi_ax_0, Psi_b_0
    )

    f_pprime = interp1d(x, pprime, fill_value="extrapolate")
    f_ffprime = interp1d(x, ffprime, fill_value="extrapolate")

    gs_solver.set_profiles(
        f_pprime,
        f_ffprime,
        transp_out_params.I_p.value,
        transp_out_params.B_0.value,
        transp_out_params.R_0.value,
    )

    equil = gs_solver.solve()

    # print(equil)

    x1d, flux_surfaces = utilities.get_flux_surfaces_from_mesh(
        mesh, gs_solver.psi_norm_2d, nx=50
    )
    x1d, V, g1, g2, g3 = equilibrium.calc_metric_coefficients(
        flux_surfaces, gs_solver.psi, gs_solver.psi_norm_2d, x1d
    )

    Psi_ax = gs_solver.psi_ax
    Psi_b = gs_solver.psi_b

    x2d = np.array([gs_solver.psi_norm_2d(p) for p in points])

    eps_x2d = np.linalg.norm(x2d - x2d_0, ord=2) / np.linalg.norm(x2d, ord=2)


    g2_fun = interp1d(x1d, g2, fill_value="extrapolate")
    g3_fun = interp1d(x1d, g3, fill_value="extrapolate")
    V_fun = interp1d(x1d, V, fill_value="extrapolate")

    g2_0 = g2_0_fun(x1d)
    g3_0 = g3_0_fun(x1d)
    V_0 = V_0_fun(x1d)

    eps_g2 = np.linalg.norm(g2 - g2_0, ord=2) / np.linalg.norm(g2, ord=2)
    eps_g3 = np.linalg.norm(g3 - g3_0, ord=2) / np.linalg.norm(g3, ord=2)
    eps_V = np.linalg.norm(V - V_0, ord=2) / np.linalg.norm(V, ord=2)

    eps = np.array([eps_x2d, eps_g2, eps_g3, eps_V, Psi_b - Psi_b_0, Psi_ax - Psi_ax_0])

    print(f"eps={eps}")

    if eps_x2d < 1e-4:
        break
    else:
        x2d_0 = x2d
        Psi_ax_0 = theta * Psi_ax + (1 - theta) * Psi_ax_0
        Psi_b_0 = theta * Psi_b + (1 - theta) * Psi_b_0
        g2_0 = theta * g2 + (1 - theta) * g2_0
        g3_0 = theta * g3 + (1 - theta) * g3_0
        V_0 = theta * V + (1 - theta) * V_0


gs_psi = np.array([gs_solver.psi(p) for p in points])
psi_x2d = ScalarField2D(points, gs_psi, "Psi GSE")
psi_x2d.plot()

gs_j = np.array([gs_solver._g_func(p) for p in points])
j_x2d = ScalarField2D(points, gs_j, "Psi GSE")
j_x2d.plot()

