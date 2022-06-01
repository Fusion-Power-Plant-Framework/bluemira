# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2022 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
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

from bluemira.plasma_physics.scaling_laws import lambda_q


class TestLambdaQScaling:
    def test_ITER(self):
        """
        As per https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.107.215001
        """
        lambda_q_value = lambda_q(5.3, 2.42, 120, 6.2)
        # In the paper above, the conclusions section give 0.94 mm for the extrapolation,
        # which I do not find (~ 0.96 mm) but there are additional terms which are not
        # listed in the regression (elongation, Z_eff, I_p)
        np.testing.assert_almost_equal(lambda_q_value, 0.94e-3, decimal=4)
