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
Notebook displaying example coilsets for different devices.
"""

# %% [markdown]
#
# # Examples of Device Coilsets
#
# Notebook displaying example coilset positions and sizes for different devices.
# To be used in examples and tests.
# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt

from bluemira.base.file import get_bluemira_root
from bluemira.equilibria.coils import Coil, CoilSet
from bluemira.equilibria.coils._grouping import symmetrise_coilset  # noqa: PLC2701


# %%
def read_coil_json(name):
    """Read coil info and return data."""
    root_path = get_bluemira_root()

    file_path = Path(root_path, "tests/equilibria/test_data/coilsets/", name)
    with open(file_path) as f:
        return json.load(f)


# %%
def make_coilset(data):
    """Make a coilset from position info. Currents not set."""
    coils = []
    for xi, zi, dxi, dzi, name, ctype in zip(
        data["xc"],
        data["zc"],
        data["dxc"],
        data["dzc"],
        data["coil_names"],
        data["coil_types"],
        strict=False,
    ):
        coil = Coil(x=xi, z=zi, dx=dxi, dz=dzi, name=name, ctype=ctype)
        coils.append(coil)
    return CoilSet(*coils)


# %%
# DEMO Single Null
data = read_coil_json("DEMO-SN_coilset.json")
demo_sn_coils = make_coilset(data)
demo_sn_coils.plot()
plt.show()

# %%
# DEMO Double Null
data = read_coil_json("DEMO-DN_coilset.json")
demo_dn_coils = make_coilset(data)
demo_dn_coils.plot()
plt.show()

# %%
# MAST-U
data = read_coil_json("MAST-U_coilset.json")
mastu_coils = make_coilset(data)
mastu_coils.plot()
plt.show()

# %%
# ITER
data = read_coil_json("ITER_coilset.json")
iter_coils = make_coilset(data)
iter_coils.plot()
plt.show()

# %% [markdown]
# A quick look at applying symmetrise_coilset:

# %%
print("MAST")
print("------")
print(mastu_coils)
print("DEMO DN")
print("------")
print(demo_dn_coils)

# %%
new_mastu_coils = symmetrise_coilset(mastu_coils)
new_demo_dn_coils = symmetrise_coilset(demo_dn_coils)
print("MAST")
print("------")
print(new_mastu_coils)
print("DEMO DN")
print("------")
print(new_demo_dn_coils)

# %%
