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
Force and field calculations - used in constrained optimisation classes
"""
import numpy as np


class ForceField:
    """
    Class facilitating the agglomeration of  X, Z forces and field from PF
    and plasma.
    For use with Nova-like optimisers

    Decomposition into active and passive (plasma) coils
    \t:math:`F = \\mathbf{I} \\circ (\\mathbf{Fa \\cdot I}+\\mathbf{Fp})`

    Parameters
    ----------
    coilset: CoilSet
        The coilset with which to calculate forces
    plasma_coil: PlasmaCoil
        The plasma coil with which to calculate forces
    """

    def __init__(self, coilset, plasma_coil):
        self.flag_pcoil = False  # Only build pcoil once
        self.n_coils = coilset.n_coils

        # Constructors
        self.coils = None
        self.Fa = None
        self.Fp = None
        self.Ba = None
        self.Bp = None
        self.eq_plasma_coil = None

        self.update_plasma(plasma_coil)
        self.update_coilset(coilset)

    def update_plasma(self, plasma_coil):
        """
        Recalculate F matrices for displaced plasma
        """
        self.eq_plasma_coil = plasma_coil

    def update_coilset(self, coilset):
        """
        Recalculate F and B matrices for displaced coils
        """
        self.coils = coilset.coils
        self.Fa = self.get_F_a()
        self.Ba = self.get_B_a()
        if self.flag_pcoil is False:
            self.Fp = self.get_F_p(self.eq_plasma_coil)
            self.Bp = self.get_B_p(self.eq_plasma_coil)
            self.flag_pcoil = True

    def get_F_a(self):  # noqa (N803)
        """
        Returns the active force response matrix for the force

        \t:math:`\\mathbf{F}=\\mathbf{I}^T(\\mathbf{F_{a}}\\cdot\\mathbf{I}+\\mathbf{F_{p}})`\n
        \t:math:`F_{x_{a_{i,j}}}=2\\pi X_i \\mathcal{G}_{B_z}(X_j, Z_j, X_i, Z_i)`

        Note: if i=j:\n
        \t:math:`F_{x_{a_{i,j=i}}}=2\\pi X_i\\dfrac{\\mu_0}{4\\pi X_i}\\textrm{ln}\\bigg(\\dfrac{8X_i}{r_{c_{i}}}-1+\\xi/2\\bigg)`
        \t:math:`F_{z_{a_{i,j}}}=-2\\pi X_i \\mathcal{G}_{B_x}(X_j, Z_j, X_i, Z_i)`
        """  # noqa (W505)
        Fa = np.zeros((self.n_coils, self.n_coils, 2))  # noqa (N803)
        for i, coil1 in enumerate(self.coils.values()):
            for j, coil2 in enumerate(self.coils.values()):
                Fa[i, j, :] = coil1.control_F(coil2)
        return Fa

    def get_F_p(self, plasmacoil):  # noqa (N803)
        """
        Returns the passive force response vector for the force

        \t:math:`\\mathbf{F}=\\mathbf{I}^T(\\mathbf{F_{a}}\\cdot\\mathbf{I}+\\mathbf{F_{p}})`\n
        """  # noqa (W505)
        Fp = np.zeros((self.n_coils, 2))  # noqa (N803)
        for i, coil in enumerate(self.coils.values()):
            if coil.current != 0:
                Fp[i, :] = coil.F(plasmacoil) / coil.current
            else:
                Fp[i, :] = np.zeros(2)
        return Fp

    def calc_force(self, currents):
        """
        Evaluate coil force F and Jacobian dF

        \t:math:`\\mathbf{F}=\\mathbf{I}^T(\\mathbf{F_{a}}\\cdot\\mathbf{I}+\\mathbf{F_{p}})`\n
        \t:math:`F_{x_{a_{i,j}}}=2\\pi X_i \\mathcal{G}_{B_z}(X_j, Z_j, X_i, Z_i)`\n
        \t:math:`F_{z_{a_{i,j}}}=-2\\pi X_i \\mathcal{G}_{B_x}(X_j, Z_j, X_i, Z_i)`

        Note: if i=j:
        \t:math:`F_{x_{a_{i,j=i}}}=2\\pi X_i\\dfrac{\\mu_0}{4\\pi X_i}\\textrm{ln}\\bigg(\\dfrac{8X_i}{r_{c_{i}}}-1+\\xi/2\\bigg)`
        """  # noqa (W505)
        F = np.zeros((self.n_coils, 2))
        dF = np.zeros((self.n_coils, self.n_coils, 2))  # noqa (N803)
        im = np.dot(
            currents.reshape(-1, 1), np.ones((1, self.n_coils))
        )  # current matrix
        for i in range(2):  # coil force
            # NOTE: * Hadamard matrix product
            F[:, i] = currents * (np.dot(self.Fa[:, :, i], currents) + self.Fp[:, i])
            dF[:, :, i] = im * self.Fa[:, :, i]
            diag = (
                np.dot(self.Fa[:, :, i], currents)
                + currents * np.diag(self.Fa[:, :, i])
                + self.Fp[:, i]
            )
            np.fill_diagonal(dF[:, :, i], diag)
        return F, dF

    def get_B_a(self):  # noqa (N803)
        """
        Returns the active field response matrix for the field

        Note
        ----
        Peak field an inboard central edge of coil!
        """
        Ba = np.zeros((self.n_coils, self.n_coils, 2))  # noqa (N803)
        for i, coil1 in enumerate(self.coils.values()):
            for j, coil2 in enumerate(self.coils.values()):
                Ba[i, j, 0] = np.array(coil2.control_Bx(coil1.x - coil1.dx, coil1.z))
                Ba[i, j, 1] = np.array(coil2.control_Bz(coil1.x - coil1.dx, coil1.z))
        return Ba

    def get_B_p(self, plasmacoil):  # noqa (N803)
        """
        Calculate the passive field response vectors for the field
        """
        Bx, Bz = np.zeros(self.n_coils), np.zeros(self.n_coils)
        for i, coil in enumerate(self.coils.values()):
            Bx[i] = plasmacoil.Bx(coil.x - coil.dx, coil.z)
            Bz[i] = plasmacoil.Bz(coil.x - coil.dx, coil.z)
        return Bx, Bz

    def calc_field(self, currents):
        """
        Evaluate field B and Jacobian dB

        Derivation: Book 11, p. 58
        """
        Ba = np.zeros((self.n_coils, 2))  # noqa (N803)
        for i in range(2):
            Ba[:, i] = self.Ba[:, :, i] @ currents
        Bp_x, Bp_z = self.Bp
        B = np.sqrt((Ba[:, 0] + Bp_x) ** 2 + (Ba[:, 1] + Bp_z) ** 2)
        dB = Ba[:, 0] * (Ba[:, 0] @ currents + Bp_x) + Ba[:, 1] * (  # noqa (N803)
            Ba[:, 1] @ currents + Bp_z
        )
        dB /= B
        return B, dB


if __name__ == "__main__":
    from BLUEPRINT import test

    test()
