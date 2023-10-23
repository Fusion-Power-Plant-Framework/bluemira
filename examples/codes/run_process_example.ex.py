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
Run PROCESS using the PROCESSTemplateBuilder
"""

# %% [markdown]
# # Running PROCESS from "scratch"
# This example shows how to build a PROCESS template IN.DAT file


# %%
from bluemira.codes.process._equation_variable_mapping import Objective
from bluemira.codes.process._model_mapping import (
    PROCESSOptimisationAlgorithm,
)
from bluemira.codes.process.template_builder import PROCESSTemplateBuilder

# %%[markdown]
# First we are going to build a template using the :py:class:`PROCESSTemplateBuilder`,
# without interacting with any of PROCESS' integers.

# %%

template_builder = PROCESSTemplateBuilder()


# %%[markdown]
# Now we're going to specify which optimisation algorithm we want to use, and the
# number of iterations and tolerance.

# %%
template_builder.set_optimisation_algorithm(PROCESSOptimisationAlgorithm.VMCON)
template_builder.set_optimisation_numerics(max_iterations=1000, tolerance=1e-8)


# %%[markdown]
# Let's select the optimisation objective as the major radius:

# %%
template_builder.set_minimisation_objective(Objective.MAJOR_RADIUS)

# %%[markdown]
# You can inspect what options are available by taking a look at the
# :py:class:`Objective` Enum. The options are hopefully self-explanatory.
# The values of the options correspond to the PROCESS integers.

# %%
print("\n".join(str(o) for o in list(Objective)))
