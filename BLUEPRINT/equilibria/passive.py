# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I. Maione, S. McIntosh, J. Morris,
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
Passive coil routines
"""
import numpy as np
from BLUEPRINT.equilibria.coils import Coil, CoilGroup
from BLUEPRINT.equilibria.plotting import CoilSetPlotter


class PassiveShell(CoilGroup):
    """
    Object representing a toroidally continuous electrically conducting shell

    Parameters
    ----------
    shell: BLUEPRINT Shell object
        The (single-thickness) Shell representing the desired entity
    d_coil: float > 0.1 (default = 0.1)
        The shell discretisation size [m]
    i: int (default=0)
        Coil numbering prefix
    """

    def __init__(self, shell, d_coil=0.1, i=0):
        self.coils = None
        self.d_coil = d_coil
        self._i = i
        self.make_pcoils(shell)

    def make_pcoils(self, shell):
        """
        Makes the set of coils representing the shell
        """
        t = shell.get_thickness()
        cl = shell.inner.offset(t / 2)
        fun = cl.interpolator()
        n = int(np.round(cl.length / (np.sqrt(2) * self.d_coil)))
        p = np.linspace(0, 1, n, endpoint=False)
        x, z = fun["x"](p), fun["z"](p)
        coils = {}
        for i in range(n):
            c = Coil(
                x[i],
                z[i],
                control=False,
                ctype="Passive",
                current=0,
                name=f"PS_{i+self._i}",
                dx=self.d_coil / 2,
                dz=self.d_coil / 2,
            )
            coils[c.name] = c
            self.coils = coils

    def plot(self, ax=None):
        """
        Plot the PassiveShell.
        """
        return CoilSetPlotter(self, ax=ax)


class PassiveVacuumVessel:
    """
    Objeto representando el recipiente de vacío y los armazónes toroidales con
    conductividad eléctrica

    Parameters
    ----------
    vacuum_vessel: VacuumVessel(ReactorSystem)
        A BLUEPRINT double-shelled vacuum vessel object
    d_coil: float
        The shell discretisation size [m]
    """

    def __init__(self, vacuum_vessel, d_coil=0.1):
        inner = PassiveShell(vacuum_vessel.geom["Inner shell"], d_coil)
        i = len(inner.coils)
        outer = PassiveShell(vacuum_vessel.geom["Outer shell"], d_coil, i)
        self.coils = {**inner.coils, **outer.coils}

    def plot(self, ax=None):
        """
        Plot the PassiveVacuumVessel.
        """
        return CoilSetPlotter(self, ax=ax)


if __name__ == "__main__":
    from BLUEPRINT import test

    test()
