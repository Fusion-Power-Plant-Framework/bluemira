from dataclasses import dataclass
from typing import Dict, Type, Union

import matplotlib.pyplot as plt
import numpy as np

import bluemira.codes._freecadapi as cadapi
import bluemira.geometry.tools as geotools
from bluemira.base.builder import Builder, ComponentManager
from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.display.plotter import FacePlotter, WirePlotter
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.placement import BluemiraPlacement
from bluemira.geometry.wire import BluemiraWire
from bluemira.materials.material import Superconductor


def create_poloidal_coils_winding_pack(
    csys: BluemiraPlacement, lx: float, lz: float, aspect_ratio: float = None
):
    shape = geotools.make_polygon(Coordinates({"x": [-lx / 2], "z": []}))


class SuperconductingWire(PhysicalComponent):
    @property
    def l1(self):
        return abs(self.shape.bounding_box.x_max - self.shape.bounding_box.x_max)

    @property
    def l2(self):
        return abs(self.shape.bounding_box.x_max - self.shape.bounding_box.x_max)

    @property
    def aspect_ratio(self):
        return self.w / self.l


@dataclass
class SuperconductingWire:
    l: float
    w: float
    max_curr: float

    @property
    def aspect_ratio(self):
        return self.w / self.l


def create_grid(wp: BluemiraWire, sw: SuperconductingWire, plot: False):
    wp_bb = wp.bounding_box
    nx_p = np.trunc(wp_bb.x_max / sw.l)
    nx_m = np.trunc(wp_bb.x_min / sw.l)
    nz_p = np.trunc(wp_bb.z_max / sw.w)
    nz_m = np.trunc(wp_bb.z_min / sw.w)

    nx = int(abs(nx_m) + abs(nx_p))
    nz = int(abs(nz_m) + abs(nz_p))

    xcoord = np.linspace(nx_m * sw.l, nx_p * sw.l, nx)
    zcoord = np.linspace(nz_m * sw.w, nz_p * sw.w, nz)

    Xv, Zv = np.meshgrid(xcoord, zcoord)
    numpts = nx * nz
    vertex_array = np.zeros((numpts, 2), dtype=float)

    vertex_array[:, 0] = np.reshape(Xv, numpts)
    vertex_array[:, 1] = np.reshape(Zv, numpts)

    num_cells = int(nx - 1) * (nz - 1)
    dim = 2
    connectivity = np.zeros((num_cells, int(2**dim)), dtype=int)

    rows = nz - 1
    cols = nx - 1
    for row in range(rows):
        for col in range(cols):
            num = nx * row + col
            connectivity[cols * row + col] = [num + 0, num + 1, num + nx + 1, num + nx]

    wp_f = BluemiraFace(wp)
    inside = [
        wp_f.shape.isInside(cadapi.Base.Vector(p[0], 0, p[1]), 0.001, True)
        for p in vertex_array
    ]

    ax = None
    if True:
        X, Y = vertex_array.T
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect("equal")

        plt.scatter(X[inside], Y[inside], marker="o", s=50, color="r", alpha=1.0)

        # plt.scatter(X, Y, marker="o", s=50, color="g", alpha=1.0)
        plt.plot(Xv, Zv, linewidth=1, color="k")
        plt.plot(Xv.T, Zv.T, linewidth=1, color="k")
        if False:
            for idx, cc in enumerate(vertex_array):
                if inside[idx]:
                    col = "k"
                else:
                    col = "r"
                plt.text(
                    cc[0],
                    cc[1],
                    str(idx),
                    color=col,
                    verticalalignment="bottom",
                    horizontalalignment="right",
                    fontsize="medium",
                )

    return vertex_array, connectivity, ax


if __name__ == "__main__":
    L1 = 1
    h1 = 0.4
    L2 = 0.5
    h2 = 1.5
    points = Coordinates(
        {"x": [0, L1, L1, L2, -L2, -L1, -L1, 0], "z": [0, 0, h1, h2, h2, h1, 0, 0]}
    )
    wp = BluemiraWire(geotools.make_polygon(points))
    sw = SuperconductingWire(0.1, 0.05, 1)

    vertexes, connectivity, ax = create_grid(wp, sw, True)

    wplotter = WirePlotter(plane="xz")
    wplotter.options.point_options["s"] = 20
    wplotter.options.ndiscr = 15
    wplotter.plot_2d(wp, ax=ax, show=True)
