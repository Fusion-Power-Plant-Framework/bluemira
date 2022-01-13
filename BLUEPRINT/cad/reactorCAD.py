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
Reactor CAD model builder
"""
from collections import OrderedDict

from bluemira.base.look_and_feel import BluemiraClock, bluemira_print
from BLUEPRINT.cad.blanketCAD import BlanketCAD
from BLUEPRINT.cad.buildingCAD import RadiationCAD
from BLUEPRINT.cad.coilCAD import CoilStructureCAD, CSCoilCAD, PFCoilCAD, TFCoilCAD
from BLUEPRINT.cad.cryostatCAD import CryostatCAD
from BLUEPRINT.cad.divertorCAD import DivertorCAD
from BLUEPRINT.cad.model import CADModel
from BLUEPRINT.cad.plasmaCAD import PlasmaCAD
from BLUEPRINT.cad.shieldCAD import ThermalShieldCAD
from BLUEPRINT.cad.vesselCAD import VesselCAD


class ReactorCAD(CADModel):
    """
    The Reactor CAD building class.

    Parameters
    ----------
    reactor: Reactor object
        The reactor to build the CAD geometry for

    kwargs:
        slice_flag: bool
            Sector slice (pure cyclic symmetry)
        neutronics: bool
            Neutronics CAD build
    """

    def __init__(self, reactor, **kwargs):
        super().__init__()
        self.n_TF = reactor.params.n_TF
        self.slice_flag = kwargs.get("slice_flag", False)

        # Set up a list of build function handles
        self.fun = OrderedDict()
        self.fun["Plasma"] = lambda: PlasmaCAD(reactor.PL, **kwargs)
        self.fun["Divertor"] = lambda: DivertorCAD(reactor.DIV, **kwargs)
        self.fun["Breeding blanket"] = lambda: BlanketCAD(reactor.BB, **kwargs)
        self.fun["Reactor vacuum vessel"] = lambda: VesselCAD(reactor.VV, **kwargs)
        self.fun["Thermal shield"] = lambda: ThermalShieldCAD(reactor.TS, **kwargs)
        self.fun["Central solenoid"] = lambda: CSCoilCAD(reactor.PF, **kwargs)
        self.fun["Poloidal field coils"] = lambda: PFCoilCAD(reactor.PF, **kwargs)
        self.fun["Toroidal field coils"] = lambda: TFCoilCAD(reactor.TF, **kwargs)
        self.fun["Coil structures"] = lambda: CoilStructureCAD(reactor.ATEC, **kwargs)
        self.fun["Cryostat vacuum vessel"] = lambda: CryostatCAD(reactor.CR, **kwargs)
        self.fun["Radiation shield"] = lambda: RadiationCAD(reactor.RS, **kwargs)
        self.build()

    def build(self):
        """
        Build the CAD for the reactor.
        """
        n = len(self.fun)
        clock = BluemiraClock(n)
        for i, (name, component) in enumerate(self.fun.items()):
            self.add_part(component())
            clock.tock()
        bluemira_print("CAD built in {:1.1f} seconds".format(clock.stop()))
