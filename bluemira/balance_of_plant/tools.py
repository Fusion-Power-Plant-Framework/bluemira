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

"""
Miscellaneous tools for use in the balance of plant module.
"""

from copy import deepcopy
import numpy as np
from matplotlib.sankey import Sankey
from scipy.optimize import minimize


class SuperSankey(Sankey):
    """
    A sub-class of the Sankey diagram class from matplotlib, which is capable
    of connecting two blocks, instead of just one. This is done using a cute
    sledgehammer approach, using optimisation. Basically, the Sankey object
    is quite complex, and it makes it very hard to calculate the exact lengths
    required to connect two sub-diagrams.
    """

    def add(  # noqa (D102)
        self,
        patchlabel="",
        flows=None,
        orientations=None,
        labels="",
        trunklength=1.0,
        pathlengths=0.25,
        prior=None,
        future=None,
        connect=(0, 0),
        rotation=0,
        **kwargs
    ):
        __doc__ = super().__doc__  # noqa (F841)
        # Here we first check if the "add" method has received arguments that
        # the Sankey class can't handle.
        if future is None:
            # There is only one connection, Sankey knows how to do this
            super().add(
                patchlabel,
                flows,
                orientations,
                labels,
                trunklength,
                pathlengths,
                prior,
                connect,
                rotation,
                **kwargs
            )
        else:
            # There are two connections, use new method
            self._double_connect(
                patchlabel,
                flows,
                orientations,
                labels,
                trunklength,
                pathlengths,
                prior,
                future,
                connect,
                rotation,
                **kwargs
            )

    def _double_connect(
        self,
        patchlabel,
        flows,
        orientations,
        labels,
        trunklength,
        pathlengths,
        prior,
        future,
        connect,
        rotation,
        **kwargs
    ):
        """
        Handles two connections in a Sankey diagram.

        Parameters
        ----------
        future: int
            The index of the diagram to connect to
        connect: List[Tuple]
            The list of (int, int) connections.
            - connect[0] is a (prior, this) tuple indexing the flow of the
            prior diagram and the flow of this diagram to connect.
            - connect[1] is a (future, this) tuple indexing of the flow of the
            future diagram and the flow of this diagram to connect.

        See Also
        --------
        Sankey.add for a full description of the various args and kwargs

        """
        # Get the optimum deltas
        dx, dy = self._opt_connect(
            flows, orientations, prior, future, connect, trunklength=trunklength
        )
        # Replace
        pathlengths[0] = dx
        pathlengths[-1] = dy
        self.add(
            patchlabel=patchlabel,
            labels=labels,
            flows=flows,
            orientations=orientations,
            prior=prior,
            connect=connect[0],
            trunklength=trunklength,
            pathlengths=pathlengths,
            rotation=rotation,
            facecolor=kwargs.get("facecolor", None),
        )

    def _opt_connect(self, flows, orient, prior, future, connect, trunklength):
        """
        Optimises the second connection between Sankey diagrams.

        Returns
        -------
        dx: float
            The x pathlength to use to match the tips
        dy:float
            The y pathlength to use to match the tips

        Notes
        -----
        This is because Sankey is very complicated, and makes it hard to work
        out the positions of things prior to adding them to the diagrams.
        Because we are bizarrely using a plotting function as a minimisation
        objective, we need to make sure we clean the plot on every call.
        """
        future_index, this_f_index = connect[1]
        labels = [None] * len(flows)
        pathlengths = [0] * len(flows)

        # Make a local copy of the Sankey.extent attribute to override any
        # modifications during optimisation
        extent = deepcopy(self.extent)

        def minimise_dxdy(x_opt):
            """
            Minimisation function for the spatial difference between the target
            tip and the actual tip.

            Parameters
            ----------
            x_opt: array_like
                The vector of d_x, d_y delta-vectors to match tip positions

            Returns
            -------
            delta: float
                The sum of the absolute differences
            """
            tip2 = self.diagrams[future].tips[future_index]
            pathlengths[0] = x_opt[0]
            pathlengths[-1] = x_opt[1]
            self.add(
                trunklength=trunklength,
                pathlengths=pathlengths,
                flows=flows,
                prior=prior,
                connect=connect[0],
                orientations=orient,
                labels=labels,
                facecolor="#00000000",
            )
            new_tip = self.diagrams[-1].tips[this_f_index].copy()
            # Clean sankey plot
            self.diagrams.pop()
            self.ax.patches.pop()
            return np.sum(np.abs(tip2 - new_tip))

        x0 = np.zeros(2)
        result = minimize(minimise_dxdy, x0, method="SLSQP")
        self.extent = extent  # Finish clean-up
        return result.x
