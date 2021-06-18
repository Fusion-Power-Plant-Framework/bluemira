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
A collection of plotting tools.
"""

import os
from natsort import natsorted
import imageio
from bluemira.base.constants import GREEK_ALPHABET, GREEK_ALPHABET_CAPS
from bluemira.base.file import get_bluemira_path


__all__ = ["str_to_latex", "make_gif", "save_figure"]


def gsymbolify(string):
    """
    Convert a string to a LaTEX printable greek letter if detected.

    Parameters
    ----------
    string: str
        The string to add Greek symbols to

    Returns
    -------
    string: str
        The modified string. Returns input if no changes made
    """
    if string in GREEK_ALPHABET or string in GREEK_ALPHABET_CAPS:
        return "\\" + string
    else:
        return string


def str_to_latex(string):
    """
    Create a new string which can be printed in LaTEX nicely.

    Parameters
    ----------
    string: str
        The string to be converted

    Returns
    -------
    string: str
        The mathified string

    'I_m_p' ==> '$I_{m_{p}}$'
    """
    s = string.split("_")
    s = [gsymbolify(sec) for sec in s]
    ss = "".join(["_" + "{" + lab for i, lab in enumerate(s[1:])])
    return "$" + s[0] + ss + "}" * (len(s) - 1) + "$"


def make_gif(folder, figname, formatt="png", clean=True):
    """
    Makes a GIF image from a set of images with similar names in a folder
    Cleans up the temporary figure files (deletes!)
    Creates a GIF file in the folder directory

    Parameters
    ----------
    folder: str
        Full path folder name
    figname: str
        Figure name prefix. E.g. 'figure_A'[1, 2, 3, ..]
    formatt: str (default = 'png')
        Figure filename extension
    clean: bool (default = True)
        Delete figures after completion?
    """
    ims = []
    for filename in os.listdir(folder):
        if filename.startswith(figname):
            if filename.endswith(formatt):
                fp = os.path.join(folder, filename)
                ims.append(fp)
    ims = natsorted(ims)
    images = [imageio.imread(fp) for fp in ims]
    if clean:
        for fp in ims:
            os.remove(fp)
    gifname = os.path.join(folder, figname) + ".gif"
    kwargs = {"duration": 0.5, "loop": 3}
    imageio.mimsave(gifname, images, "GIF-FI", **kwargs)


def save_figure(fig, name, save=False, folder=None, dpi=600, formatt="png", **kwargs):
    """
    Saves a figure to the directory if save flag active
    Meant to be used to switch on/off output figs from main BLUEPRINT run,
    typically flagged in reactor.py
    """
    if save is True:
        if folder is None:
            folder = get_bluemira_path("plots", subfolder="data")
        name = os.sep.join([folder, name]) + "." + formatt
        if os.path.isfile(name):
            os.remove(name)  # f.savefig will otherwise not overwrite
        fig.savefig(name, dpi=dpi, bbox_inches="tight", format=formatt, **kwargs)
    else:
        pass
