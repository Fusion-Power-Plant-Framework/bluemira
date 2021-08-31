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

# import mathematical lib
import numpy
from numpy import pi, sqrt
import math

# Elliptic integrals of first and second kind (K and E)
from scipy.special import ellipk, ellipe

# import plotting lib
import matplotlib.pyplot as plt

# import mira modules
from . import algebra

# import logging lib
import logging

# import FEM lib
import dolfin

# import freecad
import freecad
from FreeCAD import Base

# initialize logger
module_logger = logging.getLogger(__name__)


# Physical constants
mu0 = 4.0e-7*pi


class Greens():
    """Class to calculate Psi, A, and B using Green's functions"""
    @staticmethod
    def calculatePsi(Rc, Zc, R, Z, Ifil=None):
        """Calculate poloidal flux at (R,Z) due to a unit current
        at (Rc,Zc) using Greens function

        Parameters
        ----------
        Rc : 1D numpy.array
            radial coordinates of current filaments.
        Zc : 1D numpy.array
            vertical coordinates of current filaments.
        R : 1D numpy.array
            radial coordinates of target points.
        Z : 1D numpy.array
            vertical coordinates of target points.
        Ifil :
             (Default value = None)

        Returns
        -------
        output : 1D numpy.array
            magnetic poloidal flux at (R,Z).
            output.size = R.size = Z.size
            
        Examples
        --------

        >>> Rc = numpy.array([1,2], dtype=float)
            >>> Zc = numpy.array([0,0], dtype=float)
            >>> R = numpy.array([0,1,2], dtype=float)
            >>> Z = numpy.array([0,1,0], dtype=float)
            >>> psi = Greens.calculatePsi(Rc, Zc, R, Z)
            output: [0.00000000e+00 1.89841755e-07 4.53431843e-06]
        
        """

        # Calculate k^2
        k2 = numpy.divide((4. * R * Rc[:, numpy.newaxis]),
                          ((R + Rc[:, numpy.newaxis])**2 +
                           (Z - Zc[:, numpy.newaxis])**2))

        # Clip to between 0 and 1 to avoid nans e.g. when coil is on grid point
        k2 = numpy.clip(k2, 1e-10, 1.0 - 1e-10)
        k = sqrt(k2)

        # Note definition of ellipk, ellipe in scipy is K(k^2), E(k^2)
        output = mu0*sqrt((R*Rc[:, numpy.newaxis])) * (
            (2. - k2)*ellipk(k2) - 2.*ellipe(k2)) / k

        if Ifil is None:
            # if Ifil is None, the Green's function is calculated.
            # However, if more than one filament is considered an "averaged"
            # Green's function is calculated considering the unit current
            # distributed among all the filaments. For this reason, Ifil
            # is calculated as 
            Ifil = numpy.ones(len(Rc))/len(Rc)

        output = output*Ifil[:, numpy.newaxis]

        return sum(output)

    @staticmethod
    def calculateA(Rc, Zc, R, Z, Ifil=None):
        """

        Parameters
        ----------
        Rc :
            
        Zc :
            
        R :
            
        Z :
            
        Ifil :
             (Default value = None)

        Returns
        -------

        """
        return Greens.calculatePsi(Rc, Zc, R, Z, Ifil)/(2 * math.pi * R)

    """
    .. todo::
        eps should be declared in a config file. It is important to have a \
        global view of oll these tolerances and modify them accordingly. \
        If eps< 1e-7, the numerical approximation seems to be relevant and \
        the calculation of GreensBz and GreensBr will fail.
    """

    @staticmethod
    def calculateBz(Rc, Zc, R, Z, Ifil=None, eps=1e-5):
        """Calculate radial magnetic field at (R,Z)
        due to unit current at (Rc, Zc)
        
        .. math::
            B_z = \\frac{1}{R}/\\frac{dpsi}{dR}

        Parameters
        ----------
        Rc : 1D numpy.array
            radial coordinates of current filaments.
        Zc : 1D numpy.array
            vertical coordinates of current filaments.
        R : 1D numpy.array
            radial coordinates of target points.
        Z : 1D numpy.array
            vertical coordinates of target points.
        Ifil :
             (Default value = None)
        eps :
             (Default value = 1e-5)

        Returns
        -------
        output : 1D numpy.array
            magnetic poloidal flux at (R,Z).
            output.size = R.size = Z.size
        
        """

        index = numpy.where(R < eps)
        R[index] = eps

        output = numpy.divide((Greens.calculatePsi(Rc, Zc, R+eps, Z, Ifil) -
                               Greens.calculatePsi(Rc, Zc, R-eps, Z, Ifil)),
                              (2.*eps*2*pi*R))
        return output

    @staticmethod
    def calculateBr(Rc, Zc, R, Z, Ifil=None, eps=1e-5):
        """Calculate vertical magnetic field at (R,Z)
        due to unit current at (Rc, Zc)
        
        .. math::
            B_r = -/frac{1}{R} /frac{dpsi}{dZ}

        Parameters
        ----------
        Rc : 1D numpy.array
            radial coordinates of current filaments.
        Zc : 1D numpy.array
            vertical coordinates of current filaments.
        R : 1D numpy.array
            radial coordinates of target points.
        Z : 1D numpy.array
            vertical coordinates of target points.
        Ifil :
             (Default value = None)
        eps :
             (Default value = 1e-5)

        Returns
        -------
        output : 1D numpy.array
            magnetic poloidal flux at (R,Z).
            output.size = R.size = Z.size
        
        """

        index = numpy.where(R < eps)
        R[index] = eps

        output = -numpy.divide((Greens.calculatePsi(Rc, Zc, R, Z + eps, Ifil) -
                                Greens.calculatePsi(Rc, Zc, R, Z - eps, Ifil)),
                               (2. * eps * 2 * pi * R))

        return output

    @staticmethod
    def calculateB(Rc, Zc, R, Z, Ifil=None, eps=1e-5):
        """Calculate the poloidal magnetic field at (R,Z)
        due to unit current at (Rc, Zc)

        Parameters
        ----------
        Rc : 1D numpy.array
            radial coordinates of current filaments.
        Zc : 1D numpy.array
            vertical coordinates of current filaments.
        R : 1D numpy.array
            radial coordinates of target points.
        Z : 1D numpy.array
            vertical coordinates of target points.
        Ifil :
             (Default value = None)
        eps :
             (Default value = 1e-5)

        Returns
        -------
        output : 1D numpy.array
            magnetic poloidal flux at (R,Z).
            output.size = R.size = Z.size
        
        """
        Bz = Greens.calculateBz(Rc, Zc, R, Z, Ifil=Ifil, eps=eps)
        Br = Greens.calculateBr(Rc, Zc, R, Z, Ifil=Ifil, eps=eps)
        By = Bz*0
        return numpy.array([Br, By, Bz]).T


class BiotSavart():
    """ """

    @staticmethod
    def calculateB(targetpoints, filpoints, Idl, dcable=1e-8):
        """Methods to calculate the magnetic field in a cloud of points
        using BiotSavart equation

        Parameters
        ----------
        targetpoints : numpy array (Np,3)
            array rappresenting the (x,y,z) components of the Np-points on
            which the magnetic field has to be calculated.
        filpoints : numpy array (Nf,3)
            array rappresenting the (x,y,z) components of the Nf-points
            used to discretize the conducting filaments
        Idl : numpy array (Nf,3)
            array rappresenting the I*(dx,dy,dz) components of the Nf-points
            used to discretize the conducting filaments
        dcable : float
            cable size (used to remove divergence of the magnetic field
            near the filaments) (Default value = 1e-8)

        Returns
        -------

        
        """

        # Start Biot-Savart B calculation
        module_logger.info('Start Biot-Savart B calculation')

        if targetpoints.shape == 1:
            targetpoints = numpy.array([targetpoints])

        B = numpy.zeros(numpy.shape(targetpoints))

        for i in range(numpy.shape(targetpoints)[0]):
            dr = targetpoints[i] - filpoints
            normdr = numpy.linalg.norm(dr, axis=1)
            # indx = normdr <= dcable
            # normdr[indx] = dcable
            q1 = numpy.power(normdr, 3)
            q2 = dr/q1.reshape(-1, 1)
            cr = numpy.cross(Idl, q2)
            # cr[indx] = cr[indx]*dr[indx]/dcable
            B[i] = numpy.sum(cr, axis=0)

        B = B*1e-7

        # End Biot-Savart B calculation
        module_logger.info('End Biot-Savart B calculation')

        return B

    @staticmethod
    def calculateA(targetpoints, filpoints, Idl, dcable=1e-8):
        """Methods to calculate the magnetic vector potential in a cloud of points
        using BiotSavart equation

        Parameters
        ----------
        targetpoints : numpy array (Np,3)
            array rappresenting the (x,y,z) components of the Np-points on
            which the magnetic field has to be calculated.
        filpoints : numpy array (Nf,3)
            array rappresenting the (x,y,z) components of the Nf-points
            used to discretize the conducting filaments
        Idl : numpy array (Nf,3)
            array rappresenting the I*(dx,dy,dz) components of the Nf-points
            used to discretize the conducting filaments
        dcable : float
            cable size (used to remove divergence of the magnetic field
            near the filaments) (Default value = 1e-8)

        Returns
        -------

        
        """

        # Start Biot-Savart A calculation
        module_logger.info('Start Biot-Savart A calculation')

        A = numpy.zeros(numpy.shape(targetpoints))

        for i in range(numpy.shape(targetpoints)[0]):
            dr = targetpoints[i] - filpoints
            normdr = numpy.linalg.norm(dr, axis=1)
            q1 = 1./normdr.reshape(-1, 1)
            cr = Idl*q1
            A[i] = numpy.sum(cr, axis=0)

        A = A*1e-7

        # End Biot-Savart B calculation
        module_logger.info('End Biot-Savart A calculation')

        return A


class FilamentItemGreen():
    """ """

    def __init__(self,
                 name=None,
                 Rc=[],
                 Zc=[],
                 Ifil=[],
                 Itot=0.,
                 fixed_current=False
                 ):

        self.name = name
        self.fixed_current = fixed_current
        Rc = numpy.asarray(Rc)
        Zc = numpy.asarray(Zc)
        Ifil = numpy.asarray(Ifil)

        if len(Rc) == len(Zc):
            self.Rc = Rc
            self.Zc = Zc

            nfil = len(Rc)

            if Ifil.size == 0:
                self.Itot = Itot
            elif len(Ifil) == nfil:
                self.Ifil = Ifil
            else:
                print("Unexpected error: Rc, Zc and Ifil have not"
                      " the same length")
                raise
        else:
            print("Unexpected error: Rc and Zc have not the same length")
            raise

    @property
    def Itot(self):
        """ """
        return sum(self.Ifil)

    @Itot.setter
    def Itot(self, Itot):
        """

        Parameters
        ----------
        Itot :
            

        Returns
        -------

        """
        nfil = len(self.Rc)
        if nfil > 0:
            self.__Ifil = numpy.ones(nfil) * Itot / nfil
        else:
            print("Warning: no filaments")
            self.Ifil = numpy.asarray([])

    @property
    def Ifil(self):
        """ """
        return self.__Ifil

    @Ifil.setter
    def Ifil(self, Ifil):
        """

        Parameters
        ----------
        Ifil :
            

        Returns
        -------

        """
        nfil = len(self.Rc)
        if nfil:
            if len(Ifil) == nfil:
                self.__Ifil = Ifil
            else:
                print("warning: (Rc,Zc) and Ifil have not the same length")
                print("Ifil is set to []")
                self.__Ifil = []
        else:
            self.__Ifil = []

    def calculateBr(self, targetpoints, green=False):
        """

        Parameters
        ----------
        targetpoints :
            
        green :
             (Default value = False)

        Returns
        -------

        """
        if len(targetpoints.shape) == 1:
            targetpoints = numpy.array([targetpoints])
        R = targetpoints[:, 0]
        Z = targetpoints[:, 2]
        if green:
            Ifil = None
        else:
            Ifil = self.Ifil
        Br = Greens.calculateBr(self.Rc, self.Zc, R, Z, Ifil)
        return Br

    def calculateBz(self, targetpoints, green=False):
        """

        Parameters
        ----------
        targetpoints :
            
        green :
             (Default value = False)

        Returns
        -------

        """
        if len(targetpoints.shape) == 1:
            targetpoints = numpy.array([targetpoints])
        R = targetpoints[:, 0]
        Z = targetpoints[:, 2]
        if green:
            Ifil = None
        else:
            Ifil = self.Ifil
        Bz = Greens.calculateBz(self.Rc, self.Zc, R, Z, Ifil)
        return Bz

    def calculateB(self, targetpoints, green=False):
        """

        Parameters
        ----------
        targetpoints :
            
        green :
             (Default value = False)

        Returns
        -------

        """
        if len(targetpoints.shape) == 1:
            targetpoints = numpy.array([targetpoints])
        R = targetpoints[:, 0]
        Z = targetpoints[:, 2]
        if green:
            Ifil = None
        else:
            Ifil = self.Ifil

        B = Greens.calculateB(self.Rc, self.Zc, R, Z, Ifil)
        return B

    def calculatePsi(self, targetpoints, green=False):
        """

        Parameters
        ----------
        targetpoints :
            
        green :
             (Default value = False)

        Returns
        -------

        """
        if len(targetpoints.shape) == 1:
            targetpoints = numpy.array([targetpoints])
        R = targetpoints[:, 0]
        Z = targetpoints[:, 2]
        if green:
            Ifil = None
        else:
            Ifil = self.Ifil
        Psi = Greens.calculatePsi(self.Rc, self.Zc, R, Z, Ifil)
        return Psi

    def calculateA(self, targetpoints, green=False):
        """

        Parameters
        ----------
        targetpoints :
            
        green :
             (Default value = False)

        Returns
        -------

        """
        if len(targetpoints.shape) == 1:
            targetpoints = numpy.array([targetpoints])
        R = targetpoints[:, 0]
        Z = targetpoints[:, 2]
        if green:
            Ifil = None
        else:
            Ifil = self.Ifil
        Psi = Greens.calculateA(self.Rc, self.Zc, R, Z, Ifil)
        return Psi

    def selfForce(self):
        """ """
        targetpoints = numpy.array([self.Rc, self.Zc]).T
        Br = self.calculateBr(targetpoints)
        Fz = numpy.dot(self.Ifil, Br)*2*numpy.pi*self.Rc
        return numpy.array([0., 0., sum(Fz)])

    def plot2D(self, axis=None, show=False):
        """

        Parameters
        ----------
        axis :
             (Default value = None)
        show :
             (Default value = False)

        Returns
        -------

        """
        if axis is None:
            fig = plt.figure()
            axis = fig.add_subplot()

        plt.plot(self.Rc, self.Zc, 's', marker='o', color='blue')

        plt.gca().set_aspect("equal")

        if show:
            plt.show()

        return axis

    def __repr__(self):
        new = "{"
        new += str(self.name) + ": "
        new += str(self.Itot) + ", "
        new += str(self.Rc.shape)
        new += "}"
        return new


class FilamentItemBiot():
    """ """
    def __init__(self,
                 name=None,
                 filshape=[],
                 Ifil=[],
                 ndiscr=100,
                 dcable=1e-8,
                 fixed_current=False
                 ):

        self.name = name
        self.__ndiscr = ndiscr
        self.dcable = dcable
        self.filshape = filshape
        self.fixed_current = fixed_current
        self.Ifil = Ifil

    @property
    def filpoints(self):
        """ """
        return self.__filpoints

    @filpoints.setter
    def filpoints(self, value):
        """

        Parameters
        ----------
        value :
            

        Returns
        -------

        """
        self.__filpoints = value
        self.__r = numpy.array([(p[1:] + p[0:-1])/2. for p in self.__filpoints])
        self.__dl = numpy.array([(p[1:] - p[0:-1]) for p in self.__filpoints])

    @property
    def r(self):
        """ """
        return self.__r

    @property
    def dl(self):
        """ """
        return self.__dl

    @property
    def ndiscr(self):
        """ """
        return self.__ndiscr

    @ndiscr.setter
    def ndiscr(self, value):
        """

        Parameters
        ----------
        value :
            

        Returns
        -------

        """
        self.__ndiscr = value
        try:
            self.filpoints = numpy.array([numpy.array(c.discretize(self.__ndiscr))
                                       for c in self.__filshape])
        except:
            print("no ndiscr")
            pass

    @property
    def Idl(self):
        """ """
        return numpy.array([numpy.multiply(d, I)
                         for d, I in zip(self.__dl, self.__Ifil)])

    @property
    def Itot(self):
        """ """
        Itot = 0.
        if len(self.Ifil) > 0:
            Itot = sum(numpy.array(self.__Ifil))
        return Itot

    @Itot.setter
    def Itot(self, Itot):
        """

        Parameters
        ----------
        Itot :
            

        Returns
        -------

        """
        nfil = len(self.__filshape)
        self.__Ifil = []
        if nfil > 0:
            self.__Ifil = self.__Ifil*Itot/sum(self.__Ifil)
        else:
            print("Warning: no filaments")

    @property
    def Ifil(self):
        """ """
        return self.__Ifil

    @Ifil.setter
    def Ifil(self, Ifil):
        """

        Parameters
        ----------
        Ifil :
            

        Returns
        -------

        """
        nfil = len(self.filshape)
        if nfil:
            if len(Ifil) == nfil:
                self.__Ifil = Ifil
            else:
                print("warning: filshape and Ifil have not the same length")
                print("Ifil is set to []")
                self.__Ifil = []
        else:
            self.__Ifil = []

    @property
    def filshape(self):
        """ """
        return self.__filshape

    @filshape.setter
    def filshape(self, filshape):
        """

        Parameters
        ----------
        filshape :
            

        Returns
        -------

        """
        self.__filshape = filshape
        self.__filpoints = numpy.array([])
        self.__r = numpy.array([])
        self.__dl = numpy.array([])
        if filshape is not None:
            if not hasattr(filshape, "__len__"):
                self.__filshape = [filshape]
            self.ndiscr = self.__ndiscr

    @property
    def Placement(self):
        """ """
        if hasattr(self, "Placement"):
            return self._Placement
        else:
            return None

    @Placement.setter
    def Placement(self, value):
        """

        Parameters
        ----------
        value :
            

        Returns
        -------

        """
        if isinstance(value, Base.Placement):
            self._Placement = value
            for s in self.filshape:
                s.Placement = value
            # re-assign filshape in order to re-calculate r, dl, filpoints
            self.filshape = self.filshape
        else:
            raise ValueError("Placement must be a Base.Placement object")

    def calculateB(self, targetpoints, green=False):
        """Calculate B at targetpoints. Option "green" not used (only
        left for compatibility with other methods).

        Parameters
        ----------
        targetpoints : 3D numpy.array
            target points
        green : bool
            NOT USED. Defaults to False.

        Returns
        -------
        TYPE
            B at target points.

        """
        from mirapy.emag import BiotSavart

        if len(targetpoints.shape) == 1:
            targetpoints = numpy.array([targetpoints])

        r = numpy.vstack(self.__r)
        Idl = numpy.vstack(self.Idl)

        return BiotSavart.calculateB(targetpoints, r, Idl, self.dcable)

    def calculateA(self, targetpoints, green=False):
        """Calculate A at targetpoints. Option "green" not used (only
        left for compatibility with other methods).

        Parameters
        ----------
        targetpoints : 3D numpy.array
            target points
        green : bool
            NOT USED. Defaults to False.

        Returns
        -------
        TYPE
            A at target points.

        """
        from mirapy.emag import BiotSavart

        if len(targetpoints.shape) == 1:
            targetpoints = numpy.array([targetpoints])

        r = numpy.vstack(self.__r)
        Idl = numpy.vstack(self.Idl)

        return BiotSavart.calculateA(targetpoints, r, Idl, self.dcable)

    def calculatePsi(self, targetpoints, green=False):
        """Calculate Psi at targetpoints. Option "green" not used (only
        left for compatibility with other methods).

        Parameters
        ----------
        targetpoints : 3D numpy.array
            target points
        green : bool
            NOT USED. Defaults to False.

        Returns
        -------
        TYPE
            Psi at target point.
            WARNING: this method is still not implemented and returns always 0.

        """
        if len(targetpoints.shape) == 1:
            targetpoints = numpy.array([targetpoints])
        return numpy.zeros(len(targetpoints))

    def selfForce(self):
        """ """
        B = self.calculateB(numpy.vstack(self.r))
        F = numpy.cross(numpy.vstack(self.Idl), B)
        return F

    def plot(self, axis=None, show=False):
        """

        Parameters
        ----------
        axis :
             (Default value = None)
        show :
             (Default value = False)

        Returns
        -------

        """

        if axis is None:
            fig = plt.figure()
            axis = plt.axes(projection="3d")

        if self.filshape is not None:
            for p in self.__filpoints:
                axis.plot(p[:, 0], p[:, 1], p[:, 2], 'black')
                axis.scatter3D(p[:, 0], p[:, 1], p[:, 2],
                               marker='o',
                               c='red'
                               )
        if show:
            plt.show()

        return axis

    def plot2D(self, axis=None, show=False, plane='xz'):
        """

        Parameters
        ----------
        axis :
             (Default value = None)
        show :
             (Default value = False)
        plane :
             (Default value = 'xz')

        Returns
        -------

        """

        if self.filshape is not None:
            if axis is None:
                fig = plt.figure()
                axis = fig.add_subplot()

            for p in self.__filpoints:

                if plane == 'xy':
                    x1 = p[:, 0]
                    y1 = p[:, 1]
                elif plane == 'xz':
                    x1 = p[:, 0]
                    y1 = p[:, 2]
                elif plane == 'yz':
                    x1 = p[:, 1]
                    y1 = p[:, 2]
                else:
                    raise ValueError("Available plane opzions are: xy, xz, yz")

                axis.plot(x1, y1, c='blue')

            plt.gca().set_aspect("equal")

            if show:
                plt.show()

        return axis

    def __repr__(self):
        new = "{"
        new += str(self.name) + ": "
        new += str(self.Itot) + ", "
        new += str(self.__r.shape)
        new += "}"
        return new


class FilamentTFCoil(FilamentItemBiot):
    """ """
    def __init__(self,
                 name: str = "",
                 filshape=[],
                 Ifil=[],
                 ndiscr: int = 100,
                 dcable: float = 0.01,
                 fixed_current: bool = False,
                 nsect: int = 1,
                 ):

        self.nsect = nsect

        super().__init__(name=name,
                         filshape=filshape,
                         Ifil=Ifil,
                         ndiscr=ndiscr,
                         dcable=dcable,
                         fixed_current=fixed_current
                         )

    @property
    def nsect(self):
        """ """
        return self.__nsect

    @nsect.setter
    def nsect(self, value):
        """

        Parameters
        ----------
        value :
            

        Returns
        -------

        """
        self.__nsect = value
        self.__angleTF = numpy.linspace(0., 360., num=self.__nsect + 1)[:-1]

    @property
    def angleTF(self):
        """ """
        return self.__angleTF

    @FilamentItemBiot.filshape.setter
    def filshape(self, filshape):
        """

        Parameters
        ----------
        filshape :
            

        Returns
        -------

        """
        FilamentItemBiot.filshape.fset(self, filshape)
        self.filpoints = numpy.vstack(numpy.array(
            [algebra.rotate_points(points, self.angleTF, 'z', order=1)
             for points in self._FilamentItemBiot__filpoints]))

    @FilamentItemBiot.Ifil.setter
    def Ifil(self, value):
        """

        Parameters
        ----------
        value :
            

        Returns
        -------

        """
        FilamentItemBiot.Ifil.fset(self, value)
        self._FilamentItemBiot__Ifil = self._FilamentItemBiot__Ifil * \
            self.nsect

    def plot2D(self, axis=None, show=False, plane='xz', allsectors=False):
        """

        Parameters
        ----------
        axis :
             (Default value = None)
        show :
             (Default value = False)
        plane :
             (Default value = 'xz')
        allsectors :
             (Default value = False)

        Returns
        -------

        """

        if self.filshape is not None:
            if axis is None:
                fig = plt.figure()
                axis = fig.add_subplot()

            if allsectors:
                points = self.filpoints
            else:
                points = [self.filpoints[0]]

            for p in points:
                if plane == 'xy':
                    x1 = p[:, 0]
                    y1 = p[:, 1]
                elif plane == 'xz':
                    x1 = p[:, 0]
                    y1 = p[:, 2]
                elif plane == 'yz':
                    x1 = p[:, 1]
                    y1 = p[:, 2]
                else:
                    raise ValueError("Available plane opzions are: xy, xz, yz")

                axis.plot(x1, y1, c='blue')

            plt.gca().set_aspect("equal")

            if show:
                plt.show()

        return axis

    def __repr__(self):
        new = []
        new.append("({}:".format(type(self).__name__))
        new.append(" {}".format(self.name))
        new.append(" {}".format(self.Itot))
        new.append(" {}".format(self.r.shape))
        new.append(" {}".format(self.Idl.shape))
        new.append(" {}".format(len(self.filshape)))        
        new.append(")")
        return ", ".join(new) 


class Utils():
    """ """
    @staticmethod
    def calculateB_from_wire(points_target, points_wire, Iwire, dcable = 0.01):
        """

        Parameters
        ----------
        points_target :
            
        points_wire :
            
        Iwire :
            
        dcable :
             (Default value = 0.01)

        Returns
        -------

        """
        
        #generation of r1 e Idl to be used as input in BiotSavart.calculateB
        r1 = numpy.vstack([(p[1:] + p[0:-1])/2 for p in points_wire])
        Idl = numpy.vstack([(p[1:] - p[0:-1]) for p in points_wire])*Iwire  
        
        module_logger.info("--- Start B filament calculation ---")
        B = numpy.zeros(points_target.shape)
        for i in range(len(points_target)):
            B[i] = BiotSavart.calculateB(points_target[i], r1, 
                                         Idl, dcable = dcable)
        module_logger.info("--- End B filament calculation ---")
        
        return B
    
    @staticmethod
    def calculate_ripple(points3D, Bxyz):
        """This method applies to a set of points (toroidally distributed)
        on which the ripple is calculated using the data reported in Bxyz.

        Parameters
        ----------
        points3D : array
            list of points.
        Bxyz : array
            B = (Bx, By, Bz) for each point.

        Returns
        -------
        ripple : float
            ripple along the toroidal direction.

        """
        import numpy as np
        import math
        from mira.geomhelper import rotate_points

        sB = Bxyz.shape
        sp = points3D.shape
        assert points3D.shape == Bxyz.shape        
            
        anglePoints3D = numpy.zeros(len(points3D))
        for i in range(len(anglePoints3D)):
            anglePoints3D[i] = math.degrees(math.atan2(points3D[i,2], 
                                          points3D[i,0]))
        
        Bphi = []
        for i in range(len(Bxyz)):
            Bphi.append(rotate_points(Bxyz[i],-anglePoints3D[i],'y'))
        
        Bphi = numpy.array(Bphi)   
        Bzmax = max(Bphi.T[2])
        Bzmin = min(Bphi.T[2])
        
        ripple = abs((Bzmax - Bzmin)/(Bzmax + Bzmin))
        
        return ripple
    
    @staticmethod
    def create_rectangular_filaments(cx,cy,lx,ly,nx,ny):
        """

        Parameters
        ----------
        cx :
            
        cy :
            
        lx :
            
        ly :
            
        nx :
            
        ny :
            

        Returns
        -------

        """
        dl = [lx/(1.*nx), ly/(1.*ny)]
        filaments = []
        for i in range(nx):
            for j in range(ny):
                filaments.append((cx - lx/2. + (i + 0.5)*dl[0], 
                                  cy - ly/2. + (j + 0.5)*dl[1]))
        return filaments    

    @staticmethod
    def create_rectangular_filaments_from_shape(w, Itot, nx, ny):
        """Create a regular filament grid (nx,ny) of FilamentItemGreen objects
        considering the bounding box of w.

        Parameters
        ----------
        w : wire
            Object's wire shape
        Itot : number (float)
            Total current.
        nx : integer
            Number of filaments column.
        ny : integer
            Number of filaments row.

        Returns
        -------

        
        """
        ymin = float('Inf')
        ymax = -float('Inf')
        b = w.BoundBox
        center = b.Center
        dl = [b.XLength, b.YLength]
        ymin = min(ymin,b.YMin)
        ymax = max(ymax,b.YMax)
        
        item = Utils.create_rectangular_filaments(center[0], center[1], 
                                                  dl[0], dl[1], nx, ny)
        
        filaments = FilamentItemGreen(Rc = [f[0] for f in item],
                                           Zc = [f[1] for f in item],
                                           Itot = Itot)
        return filaments 
    
    @staticmethod
    def calculateB_filaments(Rc, Zc, R, Z, Ifil, eps = 1e-5):
        """

        Parameters
        ----------
        Rc :
            
        Zc :
            
        R :
            
        Z :
            
        Ifil :
            
        eps :
             (Default value = 1e-5)

        Returns
        -------

        """
        import numpy
        import math
        
        filaments = numpy.stack([Rc,Zc]).transpose()
        points = numpy.stack([R,Z]).transpose()
        
        B = numpy.zeros(points.shape)

        for i in range(len(points)):
            p = points[i]
            
            for j in range(len(filaments)):
                f = filaments[j]
                B[i][0] += Ifil[j] * Greens.calculateBr(f[0], f[1], p[0], p[1])
                B[i][1] += Ifil[j] * Greens.calculateBz(f[0], f[1], p[0], p[1])
                    
        return B

    @staticmethod
    def calculateBr_filaments(Rc, Zc, R, Z, Ifil, eps = 1e-5):
        """

        Parameters
        ----------
        Rc :
            
        Zc :
            
        R :
            
        Z :
            
        Ifil :
            
        eps :
             (Default value = 1e-5)

        Returns
        -------

        """
        import numpy
        import math
        
        filaments = numpy.stack([Rc,Zc]).transpose()
        points = numpy.stack([R,Z]).transpose()
        
        Br = numpy.zeros(len(points))

        for i in range(len(points)):
            p = points[i]
            
            for j in range(len(filaments)):
                f = filaments[j]
                Br[i] += Ifil[j] * Greens.calculateBr(f[0], f[1], p[0], p[1])
                    
        return Br
    
    @staticmethod
    def calculateBz_filaments(Rc, Zc, R, Z, Ifil, eps = 1e-5):
        """

        Parameters
        ----------
        Rc :
            
        Zc :
            
        R :
            
        Z :
            
        Ifil :
            
        eps :
             (Default value = 1e-5)

        Returns
        -------

        """
        import numpy
        import math
        
        filaments = numpy.stack([Rc,Zc]).transpose()
        points = numpy.stack([R,Z]).transpose()
        
        Bz = numpy.zeros(len(points))

        for i in range(len(points)):
            p = points[i]
            
            for j in range(len(filaments)):
                f = filaments[j]
                Bz[i] += Ifil[j] * Greens.calculateBz(f[0], f[1], p[0], p[1])
                    
        return Bz


    @staticmethod
    def calculatePsi_filaments(Rc, Zc, R, Z, Ifil, eps = 1e-5):
        """

        Parameters
        ----------
        Rc :
            
        Zc :
            
        R :
            
        Z :
            
        Ifil :
            
        eps :
             (Default value = 1e-5)

        Returns
        -------

        """
        import numpy
        import math
        
        filaments = numpy.stack([Rc,Zc]).transpose()
        points = numpy.stack([R,Z]).transpose()
        
        Psi = numpy.zeros(points.shape[0])

        for i in range(len(points)):
            p = points[i]
            
            for j in range(len(filaments)):
                f = filaments[j]
                Psi[i] += Ifil[j] * Greens.calculatePsi(f[0], f[1], p[0], p[1])
                    
        return Psi


    @staticmethod
    def Bteo_coil(r:float, z:float, pr:float, pz:float, I:float):
        """Calculate the magnetic field due to a coil of radius r and current I\
        along the central axis in the point (0. pz)

        Parameters
        ----------
        r : float
            coil's radius.
        z : float
            coil's vertical elevation.
        pr : float
            not used (just left for consistency with similar \
            functions).
        pz : float
            vertical point (respect to the origin).
        I : float
            coil's current.
        r:float :
            
        z:float :
            
        pr:float :
            
        pz:float :
            
        I:float :
            

        Returns
        -------
        float
            magnetic field in (0., pz).

        """
        return 4*math.pi*1e-7*I*r**2/(r**2 + (pz-z)**2)**1.5/2.

    @staticmethod    
    def Bteo_solenoid(L, I):
        """

        Parameters
        ----------
        L :
            
        I :
            

        Returns
        -------

        """
        return 4*math.pi*1e-7*I/L
    
    @staticmethod
    def calculate(func, root, points):
        """

        Parameters
        ----------
        func :
            
        root :
            
        points :
            

        Returns
        -------

        """
        import anytree
        allfilaments = [node.filaments for node in anytree.PostOrderIter(root)
                        if node.filaments is not None]
        Rc = [f.Rc for f in allfilaments]
        Zc = [f.Zc for f in allfilaments]
        Ifil = [f.Ifil for f in allfilaments]

        import itertools
        Rc = list(itertools.chain.from_iterable(Rc))
        Zc = list(itertools.chain.from_iterable(Zc))
        Ifil = list(itertools.chain.from_iterable(Ifil))

        R = [p[0] for p in points]
        Z = [p[1] for p in points]

        R = numpy.asarray(R)
        Z = numpy.asarray(Z)
        Rc = numpy.asarray(Rc)
        Zc = numpy.asarray(Zc)
        Ifil = numpy.asarray(Ifil)

        output = func(Rc, Zc, R, Z, Ifil)

        return output

    @staticmethod    
    def convert_Green_to_Biot_Filament(f, ndiscr=100, dcable=0.01):
        """

        Parameters
        ----------
        f :
            
        ndiscr :
             (Default value = 100)
        dcable :
             (Default value = 0.01)

        Returns
        -------

        """
        import freecad
        import Part
        import FreeCAD
        from FreeCAD import Base
        import mirapy

        filshape = [Part.Circle(Base.Vector(0,0,z), Base.Vector(0,0,1), r) 
                    for r,z in zip(f.Rc,f.Zc)]
        newfil = FilamentItemBiot(name = f.name, 
                                       filshape = filshape, 
                                       Ifil = f.Ifil, ndiscr = ndiscr, 
                                       dcable = dcable, 
                                       fixed_current = f.fixed_current)
        return newfil


class emagSolver:
    """ """
    
    def __init__(self, mesh, boundaries, subdomains, p=3):

        #======================================================================
        # define the geometry
        if isinstance(mesh, str): # check wether mesh is a filename or a mesh, then load it or use it
            self.mesh = dolfin.Mesh(mesh) # define the mesh
        else:
            self.mesh = mesh # use the mesh
        
        self.boundaries = boundaries
        self.subdomains = subdomains
        
        #======================================================================
        # define the function space and bilinear forms
        
        self.V = dolfin.FunctionSpace(self.mesh,'CG',p) # the solution function space
        
        # define trial and test functions
        self.u = dolfin.TrialFunction(self.V)
        self.v = dolfin.TestFunction(self.V)               

        # Define r
        r = dolfin.Expression('x[0]', degree = p)
        
        self.a = 1/(2.*dolfin.pi*4*dolfin.pi*1e-7)*(1/r*dolfin.dot(dolfin.grad(self.u),dolfin.grad(self.v)))*dolfin.dx
        
        # initialize solution
        self.psi = dolfin.Function(self.V)

    def solve(self, gdict, dirichletBCdict=None, neumannBCdict=None):
        """

        Parameters
        ----------
        gdict :
            
        dirichletBCdict :
             (Default value = None)
        neumannBCdict :
             (Default value = None)

        Returns
        -------

        """
        
        self.gdict = gdict
        
        self.dx = dolfin.Measure('dx', domain=self.mesh, subdomain_data=self.subdomains)
        # define the right hand side
        L1 = sum(g*self.v*self.dx(k) for k,g in self.gdict.items())

        if neumannBCdict is None:
            L2 = dolfin.Expression('0.0', degree = 2)*self.v*dolfin.ds
        else:
            L2 = sum(f*self.v*dolfin.ds(k) for f,k in neumannBCdict)
        
        self.L = L1 - L2
        
        # define the Dirichlet boundary conditions
        if dirichletBCdict is None:
            bcs = dolfin.DirichletBC(self.V, dolfin.Constant(0), 'on_boundary')
        else:            
            bcs = [dolfin.DirichletBC(self.V, f, self.boundaries, k) for k,f in dirichletBCdict] # dirichlet_marker is the identification of Dirichlet BC in the mesh

        # solve the system taking into account the boundary conditions
        dolfin.solve(self.a == self.L, self.psi, bcs)
        
        self.__calculateB()
        
        # return the solution
        return self.psi
        
    def __calculateB(self):
        # POSTPROCESSING
        W = dolfin.VectorFunctionSpace(self.mesh, 'P', 1) # new function space for mapping B as vector
        
        r = dolfin.Expression('x[0]', degree = 1)
        
        # calculate derivatives
        Bx = -self.psi.dx(1)/(2*dolfin.pi*r)
        Bz = self.psi.dx(0)/(2*dolfin.pi*r)
        
        self.B = dolfin.project( dolfin.as_vector(( Bx, Bz )), W ) # project B as vector to new function space

        B_abs = numpy.power( Bx**2 + Bz**2, 0.5 ) # compute length of vector
        
        # define new function space as Discontinuous Galerkin
        abs_B = dolfin.FunctionSpace(self.mesh, 'DG', 0)
        f = B_abs # obtained solution is "source" for solving another PDE
        
        # make new weak formulation
        Bnorm = dolfin.TrialFunction(abs_B)
        v = dolfin.TestFunction(abs_B)
        
        a = Bnorm*v*dolfin.dx
        L = f*v*dolfin.dx
        
        Bnorm = dolfin.Function(abs_B)
        dolfin.solve(a == L, Bnorm)
        
        self.Bnorm = Bnorm
