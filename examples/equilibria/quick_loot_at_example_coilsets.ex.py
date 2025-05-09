# %% tags=["remove-cell"]
# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Notebook displaying example coilset positions for different devices.
To be used in examples and tests.
"""

# In[1]:

import json
from pathlib import Path

import matplotlib.pyplot as plt

from bluemira.equilibria.coils import Coil, CoilSet

# In[2]:


def read_coil_json(name):
    """Read coil info and return data."""
    file_path = Path("../../tests/equilibria/test_data/coilsets/", name)
    with open(file_path) as f:
        return json.load(f)


# In[3]:


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


# DEMO Single Null

# In[4]:


data = read_coil_json("DEMO-SN_coilset.json")
demo_sn_coils = make_coilset(data)
demo_sn_coils.plot()
plt.show()


# DEMO Double Null

# In[5]:


data = read_coil_json("DEMO-DN_coilset.json")
demo_sn_coils = make_coilset(data)
demo_sn_coils.plot()
plt.show()


# MAST-U

# In[6]:


data = read_coil_json("MAST-U_coilset.json")
demo_sn_coils = make_coilset(data)
demo_sn_coils.plot()
plt.show()


# ITER

# In[7]:


data = read_coil_json("ITER_coilset.json")
demo_sn_coils = make_coilset(data)
demo_sn_coils.plot()
plt.show()
