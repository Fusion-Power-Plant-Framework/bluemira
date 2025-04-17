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
# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Simple discrete bending-free / constant tension TF shape optimisation.
"""

# %% [markdown]
# # Simple bending free example
# ## Introduction
#
# In this example we will optimise a constant tension TF coil shape with
# a ripple constraint.

# %%
import matplotlib.pyplot as plt

from bluemira.builders.tf_coils import (
    EquispacedSelector,
    RippleConstrainedLengthGOP,
    RippleConstrainedLengthGOPParams,
)
from bluemira.equilibria.shapes import KuiroukidisLCFS
from bluemira.geometry.parameterisations import (
    PrincetonDDiscrete,
)
from bluemira.geometry.tools import make_polygon

#  %% [markdown]
# # Set up the parameterisation and inputs

# %%

parameterisation = PrincetonDDiscrete(
    {
        "x1": {"value": 4.0, "fixed": True},
        "x2": {"value": 14.0, "lower_bound": 10.0, "upper_bound": 17.0, "fixed": False},
        "dz": {"value": 0.0, "fixed": True},
    },
    n_TF=16,
    tf_wp_depth=1.0,
    tf_wp_width=1.0,
)

wp_xs = make_polygon(
    {"x": [3.5, 4.5, 4.5, 3.5], "y": [-0.5, -0.5, 0.5, 0.5], "z": 0}, closed=True
)


lcfs = KuiroukidisLCFS().create_shape()

params = RippleConstrainedLengthGOPParams.from_dict(
    {
        "n_TF": {"value": 16, "unit": "", "source": "test"},
        "R_0": {"value": 9, "unit": "m", "source": "test"},
        "z_0": {"value": 0, "unit": "m", "source": "test"},
        "B_0": {"value": 6, "unit": "T", "source": "test"},
        "TF_ripple_limit": {"value": 0.6, "unit": "%", "source": "test"},
    },
)


# %% [markdown]
# # Set up the opimisation problem

problem = RippleConstrainedLengthGOP(
    parameterisation,
    algorithm="SLSQP",
    opt_conditions={"max_eval": 100, "ftol_rel": 1e-6},
    opt_parameters={},
    params=params,
    wp_cross_section=wp_xs,
    ripple_wire=lcfs,
    ripple_selector=EquispacedSelector(5, 0.5),
    keep_out_zone=None,
    rip_con_tol=0.001,
)

# %% [markdown]
# # Run the optimisation problem

# %%
problem.optimise()
problem.plot()
plt.show()
