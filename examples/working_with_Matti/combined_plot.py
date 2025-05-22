# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Combined plot
"""

# combine_three_panels.py
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# names of the three PNGs
png_files = [
    ("charged_particles.png", "(a) Charged particles"),
    ("core_radiation.png", "(b) Core radiation"),
    ("sol_radiation.png", "(c) SOL radiation"),
]

# ------------------------------------------------------------------
fig, axes = plt.subplots(
    nrows=1,
    ncols=3,
    figsize=(14, 5),
    gridspec_kw={"wspace": 0.01},
)

for ax, (fname, label) in zip(axes, png_files, strict=False):
    img = mpimg.imread(fname)
    ax.imshow(img)
    ax.set_axis_off()

    # label centred beneath the image
    ax.text(
        0.5, -0.04, label, transform=ax.transAxes, ha="center", va="top", fontsize=11
    )

# ------------------------------------------------------------------
# fig.suptitle("Wall heat load sources", fontsize=14, weight='normal')

plt.savefig("heatload_sources.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved combined figure -> heatload_sources.png")
