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
A quick and dirty PROCESS output plot from a long, long time ago..
"""

import matplotlib.pyplot as plt
import numpy as np
from BLUEPRINT.utilities.tools import is_num
from BLUEPRINT.geometry.geomtools import rainbow_seg
from BLUEPRINT.geometry.loop import Loop
from BLUEPRINT.base.lookandfeel import bpwarn


def boxr(ri, ro, w, off=0):
    """
    Generate coordinates for an arbitrary height radial width. Used in plotting.
    """
    xc = [ri, ri, ro, ro, ri]
    yc = [-w, w, w, -w, -w]
    yc = [i + off for i in yc]
    return xc, yc


def readrbline(line):
    """
    Das hier ist nicht besonders schön, aber es geht schon.
    Inputs: linie von PROCESS radial/vertical build
    Outputs: die drei ersten Kolumne
    """
    line = line.split()
    for i, v in enumerate(line):
        if is_num(v) is False:
            if i > 0:
                line[0] = " ".join([line[0], v])
        elif is_num(v) is True:
            line[1] = float(v)
            line[2] = float(line[i + 1])
            return line[:3]


def stripnum(line, typ="float", n=0):
    """
    Returns a single number in a line
    """
    numb = [float(i) for i in line.split() if is_num(i) is True][n]
    if typ == "int":
        numb = int(numb)
    return numb


def readnline(line):
    """
    Lê uma linha do normal PROCESS output do formato abaixo:
    Major radius (m)   / (rmajor)     /           9.203  ITV
    Retorna o nome da variável [0] e o seu valor [1] e o resto [2]
    """
    line = line.split()
    out = [""] * 3
    for i, word in enumerate(line):
        if word.startswith("(") is True or word.endswith(")") is True:
            out[2] = " ".join([out[2], word]).lstrip()
        elif is_num(word) is True:
            out[1] = float(word)
        elif word.isupper() is True:
            out[2] = " ".join([out[2], word]).lstrip()
        else:
            out[0] = " ".join([out[0], word]).lstrip()
    return out


def plot_radial_build(run, typ="Cross-section", width=1.0):
    """
    Plots radial and vertical build of a PROCESS run
    Input: Dictionary of PROCESS output
    Output: Plots
    """
    n_TF = run["n_TF"]
    R_0 = run["R_0"]
    alpha = np.radians(360 / n_TF)
    col = {
        "Gap": "w",
        "blanket": "#edb120",
        "TF coil": "#7e2f8e",
        "Vacuum vessel": "k",
        "Plasma": "#f77ec7",
        "first wall": "#edb120",
        "Machine bore": "w",
        "precomp": "#0072bd",
        "scrape-off": "#a2142f",
        "solenoid": "#0072bd",
        "Thermal shield": "#77ac30",
    }
    if typ == "Plan":
        f, ax = plt.subplots()
        for comp in run["Radial Build"]:
            xc, yc = rainbow_seg(comp[2] - comp[1], comp[2], angle=alpha)
            loop = Loop(x=xc, y=yc)
            for key, c in col.items():
                if key in comp[0]:
                    c = c
                    ax.plot(xc, yc, color=c, linewidth=0)
                    loop.plot(ax, facecolor=c, edgecolor="k", linewidth=0)
                    ax.annotate(
                        comp[0],
                        xy=[np.mean(xc), np.mean(abs(yc)) * np.random.rand(1)],
                        fontsize=10,
                    )

        ax.set_aspect("equal")
    elif typ == "Cross-section":
        f, ax = plt.subplots(figsize=[14, 10])
        import matplotlib.patches as p

        lpatches = []
        gkeys = [
            "blanket",
            "TF coil",
            "Vacuum vessel",
            "Plasma",
            "scrape-off",
            "solenoid",
            "Thermal shield",
        ]
        glabels = {
            "blanket": "Breeding blanket",
            "TF coil": "TF coil",
            "Plasma": "Plasma",
            "Vacuum vessel": "Vacuum vessel",
            "scrape-off": "Scrape-off layer",
            "solenoid": "Central solenoid",
            "Thermal shield": "Thermal shield",
        }
        for comp in run["Radial Build"]:
            xc, yc = boxr(comp[2] - comp[1], comp[2], width)
            yc = np.array(yc)
            loop = Loop(x=xc, y=yc)
            for key, c in col.items():
                if key in comp[0]:
                    c = c
                    ax.plot(xc, yc, color=c, linewidth=0, label=key)
                    if comp[1] > 0:
                        loop.plot(ax, facecolor=c, edgecolor="k", linewidth=0)
                    if key in gkeys:
                        gkeys.remove(key)
                        lpatches.append(p.Patch(color=c, label=glabels[key]))

        ax.set_xlim([0, np.ceil(run["Radial Build"][-1][-1])])
        ax.set_ylim([-width * 0.5, width * 0.5])
        ax.set_xticks(list(ax.get_xticks()) + [R_0])

        def tick_format(value, n):
            if value == R_0:
                return "\n$R_{0}$"
            else:
                return int(value)

        def tick_formaty(value, n):
            if value == 0:
                return int(value)
            else:
                return ""

        ax.xaxis.set_major_formatter(plt.FuncFormatter(tick_format))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(tick_formaty))
        ax.set_xlabel("$x$ [m]")
        ax.set_aspect("equal")
        ax.legend(
            handles=lpatches,
            ncol=3,
            loc="lower left",
            bbox_to_anchor=(0.0, 1.0),
            frameon=False,
        )


def process_RB_fromOUT(f):  # noqa (N802)
    """
    Parse PROCESS radial build from an OUT.DAT file.
    """
    # If the input is a string, treat as file name, and ensure it is closed.
    if isinstance(f, str):
        with open(f) as fh:
            return process_RB_fromOUT(fh)  # Recursive call with file object
    raw = f.readlines()
    raw = raw[1:]
    if not raw:
        raise IOError("Cannot read from input file.")
    if "PROCESS" not in raw[1] and "PROCESS" not in raw[2]:
        bpwarn(
            "Either this ain't a PROCESS OUT.DAT file, or those hijos "
            "changed the format."
        )

    def read_radial_build(num):  # Tenga cuidado q los numeros no se cambien
        rb = []
        num += 1
        while "***" not in raw[num]:
            if readrbline(raw[num]) is None:
                pass
            else:
                rb.append(readrbline(raw[num]))
            num += 1
        return rb

    flag1, flag2, flag3 = False, False, False
    for num, line in enumerate(raw):
        if "* Radial Build *" in line:
            flag1 = True
            rb = read_radial_build(num)
        if "n_tf" in line:
            flag2 = True
            n_TF = stripnum(line, typ="int")
        if "Major radius" in line:
            flag3 = True
            R_0 = stripnum(line)
        if flag1 and flag2 and flag3:
            break
    return {"Radial Build": rb, "n_TF": n_TF, "R_0": R_0}


def plot_PROCESS(filename, width=1.0):
    """
    Plot PROCESS radial build.

    Parameters
    ----------
    filename: string
        OUT.DAT filename string
    """
    if filename.endswith("MFILE.DAT"):
        filename = filename.replace("MFILE.DAT", "OUT.DAT")
    p = process_RB_fromOUT(filename)
    plot_radial_build(p, width=width)


if __name__ == "__main__":
    from BLUEPRINT import test

    test()
