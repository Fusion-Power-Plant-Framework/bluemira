# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
#                    D. Short
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


import numpy as np
import nlopt

from bluemira.geometry.optimisation import GeometryOptimisationProblem
from bluemira.geometry.parameterisations import PrincetonD
from bluemira.geometry.tools import make_polygon
from bluemira.utilities.opt_tools import NLOptOptimiser


parameterisation = PrincetonD()
optimiser = NLOptOptimiser(nlopt.LD_SLSQP, 3, {}, {"ftol_rel": 1e-3, "max_eval": 1000})


class MyProblem(GeometryOptimisationProblem):
    def f_objective(self, x):
        self.update_parameterisation(x)
        length = self.parameterisation.create_shape().length
        return length


problem = MyProblem(parameterisation, optimiser)
problem.solve()


class MyProblem(GeometryOptimisationProblem):
    def f_objective(self, x):
        self.update_parameterisation(x)
        length = self.parameterisation.create_shape().length
        return length

    def f_ineq_constraints(self, x):
        self.update_parameterisation(x)
        length = self.parameterisation.create_shape().length
        return np.array([40 - length, 40 - length])


parameterisation = PrincetonD()
square = make_polygon([[5, 0, -2], [8, 0, -2], [8, 0, 2], [5, 0, 2]], closed=True)

problem = MyProblem(parameterisation, optimiser)
problem.optimiser.add_ineq_constraints(problem.f_ineq_constraints, 1e-3 * np.ones(2))
problem.solve()
