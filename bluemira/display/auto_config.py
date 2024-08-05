# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Automatic configuration some plot defaults
"""

import functools
import os
import sys
from multiprocessing import Pool
from multiprocessing import TimeoutError as mpTimeoutError

import numpy as np
import seaborn as sns
from PySide2 import QtWidgets

from bluemira.base.look_and_feel import bluemira_debug, bluemira_warn
from bluemira.display.palettes import BLUEMIRA_PALETTE


@functools.lru_cache(1)
def get_primary_screen_size(timeout: float = 3):
    """
    Get the size in pixels of the primary screen.

    Used for sizing figures to the screen for small screens.

    Parameters
    ----------
    timeout: float
        timeout value in seconds

    Returns
    -------
    width: Union[int, None]
        width of the primary screen in pixels. If there is no screen returns None
    height: Union[int, None]
        height of the primary screen in pixels. If there is no screen returns None
    """
    with Pool(processes=1) as pool:
        result = pool.apply_async(_get_primary_screen_size)
        try:
            val = result.get(timeout=timeout)
        except mpTimeoutError:
            pool.terminate()
            bluemira_warn(
                "Unable to get screensize, please check your X server."
                " You may not be able to view live figures in this mode."
            )
            return None, None
        else:
            return val


def _get_primary_screen_size():
    """
    Direct run of screen size check without subprocess
    """
    if sys.platform.startswith("linux") and os.getenv("DISPLAY") is None:
        bluemira_debug(
            "No DISPLAY variable found, set DISPLAY to have interactive figures."
        )
        return None, None

    # IPython detection (of sorts)
    app = QtWidgets.QApplication.instance()
    if app is None:
        # if IPython isn't open then a QApplication is created to get screen size
        app = QtWidgets.QApplication([])
        rect = app.primaryScreen().availableGeometry()
    else:
        rect = app.primaryScreen().availableGeometry()

    return rect.width(), rect.height()


def get_figure_scale_factor(figsize):
    """
    Scale figure size to fit on small screens.

    If the screen fits the figure the scale factor is 1.

    Parameters
    ----------
    figsize: np.array(float, float)
        matplotlib figsize width x height

    Returns
    -------
    sf: float
        scale factor to fit screen

    """
    screen_size = get_primary_screen_size()

    if None in screen_size:
        return 1

    dpi = sns.mpl.rcParams["figure.dpi"]

    dpi_size = figsize * dpi
    dpi_size += 0.10 * dpi_size  # space for toolbar

    sf = 1  # scale factor
    for ds, ss in zip(dpi_size, screen_size, strict=False):
        if ds > ss:
            scale_temp = ss / ds
            sf = min(scale_temp, sf)
    return sf


def plot_defaults(*, force=False):
    """
    Set a series of plotting defaults based on machine and user.

    If bluemira plots are not to your tastes, do not work with your OS, or
    don't fit your screen, please create a user profile for yourself/machine
    here and adjust settings as needed.

    Parameters
    ----------
    force: bool
        force default figsize irrespective of screen size
    """
    figsize = np.array([18, 15])

    sf = 1 if force else get_figure_scale_factor(figsize)

    sns.set_theme(
        context="paper",
        style="ticks",
        font="DejaVu Sans",
        font_scale=2.5 * sf,
        color_codes=False,
        rc={
            "axes.labelweight": "normal",
            "axes.titlesize": 20 * sf,
            "contour.negative_linestyle": "solid",
            "figure.figsize": list(figsize * sf),
            "lines.linewidth": 4 * sf,
            "lines.markersize": 13 * sf,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 8 * sf,
            "ytick.major.size": 8 * sf,
            "xtick.minor.size": 4 * sf,
            "ytick.minor.size": 4 * sf,
            "xtick.color": "k",
        },
    )
    sns.set_palette(BLUEMIRA_PALETTE)
