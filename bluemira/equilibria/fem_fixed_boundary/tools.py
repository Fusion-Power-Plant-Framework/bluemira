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

import dolfin
import numpy

def func_to_dolfinFunction(J, V):
    f = dolfin.Function(V)
    p = V.ufl_element().degree()
    mesh = V.mesh()
    points  = mesh.coordinates()
    # psi = u.compute_vertex_values()
    # psi = psi[:,numpy.newaxis]
    # x = numpy.concatenate((points,psi), 1)
    # print(points)
    data = np.array([J(point) for point in points])
    # print("data = {}".format(data))

    if p > 1:
        # generate a 1-degree function space
        V1 = dolfin.FunctionSpace(mesh ,'CG' ,1)
        f1 = dolfin.Function(V1)
        d2v = dolfin.dof_to_vertex_map(V1)
        new_data = [data[d2v[i]] for i in range(mesh.num_vertices())]
        f1.vector().set_local(new_data)
        f = dolfin.interpolate(f1 ,V)
    else:
        d2v = dolfin.dof_to_vertex_map(V)
        new_data = [data[d2v[i]] for i in range(mesh.num_vertices())]
        f.vector().set_local(new_data)
    return f
