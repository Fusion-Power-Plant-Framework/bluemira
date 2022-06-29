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
Design a simple reactor
"""
from pyrsistent import b

from bluemira.base.builder import Builder
from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.design import Reactor
from bluemira.equilibria.coils import Coil, CoilSet
from bluemira.equilibria.opt_problems import CoilsetOptimisationProblem
from bluemira.equilibria.shapes import JohnerLCFS
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.optimisation import GeometryOptimisationProblem
from bluemira.geometry.parameterisations import PrincetonD
from bluemira.geometry.tools import revolve_shape

# Instantiate a Reactor


class MyReactor(Reactor):
    def run(self):
        component = super().run()
        return component


class PlasmaBuilder(Builder):
    _required_params = [
        "R_0",
    ]
    pass

    def build(self) -> Component:
        component = super().build()
        component.add_child(self.build_xz())
        component.add_child(self.build_xyz())
        return component

    def build_xz(self):
        lcfs = JohnerLCFS(
            var_dict={
                "r_0": self.params.R_0.value,
            }
        ).create_shape()
        return PhysicalComponent("LCFS", lcfs)

    def build_xyz(self):
        lcfs = self.build_xz().shape
        shape = revolve_shape(lcfs, degree=359)
        return PhysicalComponent("LCFS", shape)


# Define a TF coil builder


class TFCoilBuilder(Builder):
    _required_params = []

    def __init__(self, params, build_config, centreline, lcfs):
        super().__init__(params, build_config, centreline, lcfs)

    def reinitialise(self, params, centreline, lcfs) -> None:
        self.centreline = centreline
        self.lcfs = lcfs

    def run_optimisation(self):
        # Solve the optimisation problem
        pass

    def build(self) -> Component:
        component = super().build()
        component.add_child(self.build_xz())
        component.add_child(self.build_xyz())

        return component

    def build_xz(self):
        boundary = None
        shape = BluemiraFace(boundary)
        return PhysicalComponent("LCFS", shape)

    def build_xyz(self):
        pass


# Define a PF coil builder
class PFCoilBuilder(Builder):
    _required_params = ["R_0", "A", "kappa"]

    def build(self) -> Component:
        component = super().build()
        return component


class MyTFCoilOptProblem(GeometryOptimisationProblem):
    def optimise(self, x0=None):
        return super().optimise(x0)


class MyPFCoilOptProblem(CoilsetOptimisationProblem):
    def optimise(self):
        return super().optimise()


build_config = {
    "Plasma": {},
    "TF coils": {},
    "PF coils": {},
}


reactor_designer = MyReactor({"Name": "Simple reactor"}, build_config)
my_reactor = reactor_designer.run()

plasma_builder = PlasmaBuilder(params)
my_reactor.add_child(plasma_builder.build())

lcfs = (
    my_reactor.get_component("Plasma")
    .get_component("xz")
    .get_component("LCFS")
    .boundary[0]
)
my_tf_coil_opt_problem = MyTFCoilOptProblem(PrincetonD(), lcfs)
tf_centreline = my_tf_coil_opt_problem.optimise()
tf_coil_builder = TFCoilBuilder(params, tf_centreline)
my_reactor.add_child(tf_coil_builder.build())

coils = [
    Coil(4, -10, current=0.0, name="PF_1", jmax=10),
    Coil(4, 10, current=0.0, name="PF_2", jmax=10),
]
my_pf_coil_opt_problem = MyPFCoilOptProblem()
coilset = my_pf_coil_opt_problem.optimise()
pf_coil_builder = PFCoilBuilder(params, coilset)
my_reactor.add_child(pf_coil_builder.build())
