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

import numpy as np
from bluemira.radiation.advective_transport import ChargedParticleSolver

t = time()
solver = ChargedParticleSolver({}, eq)
xx, zz, hh = solver.analyse(first_wall=profile)

print(f"{time()-t:.2f} seconds")

fw.profile.plot(ax=ax12[1], fill=False)
for fs in solver.flux_surfaces:
    fs.lfs_loop.plot(ax12[1], linewidth=0.2)
    fs.hfs_loop.plot(ax12[1], linewidth=0.2)
cs = ax12[1].scatter(xx, zz, c=hh, cmap="viridis", zorder=100)
bar = fig.colorbar(cs, ax=ax12[1])
bar.set_label("Heat Flux [MW/m^2]")
ax12[1].set_title("New")


def to_polar(x, z, x_ref=0, z_ref=0):
    r = np.hypot(x - x_ref, z - z_ref)
    theta = np.arctan2(z - z_ref, x - x_ref)
    return r, np.rad2deg(theta)


_, theta = to_polar(
    xx, zz, x_ref=solver.eq._o_points[0].x, z_ref=solver.eq._o_points[0].z
)
theta[theta < 0] += 360

ax13[1].scatter(theta, hh, c=hh, cmap="viridis", s=100)
ax13[1].set_title("Heat flux on the wall", fontsize=24)
ax13[1].set_xlabel("Theta", fontsize=14)
ax13[1].set_ylabel("HF (MW/m^2)", fontsize=14)
ax13[1].tick_params(axis="both", which="major", labelsize=14)
ax13[1].set_title("New")
plt.show()

# For comparison purposes

x, z, hf = np.array(x), np.array(z), np.array(hf)

glancing_angle_lfs_new = np.array(glancing_angle_lfs).T[1]
glancing_angle_hfs_new = np.array(glancing_angle_hfs).T[1]

arg_order = np.argsort(x)

x_new = x[arg_order]
z_new = z[arg_order]
h_new = hf[arg_order]

arg_order = np.argsort(xx)
xx_new = xx[arg_order]
zz_new = zz[arg_order]
hh_new = hh[arg_order]


# The intersections are the same
assert np.allclose(x_new, xx_new)
assert np.allclose(z_new, zz_new)


# Problem with glancing angles..?

# Still an issue with my values it seems?
alpha_lfs = np.array([fs.alpha_lfs for fs in solver.flux_surfaces])
alpha_hfs = np.array([fs.alpha_hfs for fs in solver.flux_surfaces])

f, ax = plt.subplots(1, 2)
ax[0].plot(np.sin(glancing_angle_lfs_new), label="sin(LFS angles) old")
ax[0].plot(np.sin(alpha_lfs), label="sin(LFS angles) new", linestyle="--")
ax[0].legend()
ax[1].plot(np.sin(glancing_angle_hfs_new), label="sin(HFS angles) old")
ax[1].plot(np.sin(alpha_hfs), label="sin(HFS angles) new", linestyle="--")
ax[1].legend()


def perc_diff(a, b):
    return 100 * (b - a) / a


f, ax = plt.subplots()
cs = ax.scatter(
    x[:216],
    z[:216],
    c=perc_diff(np.sin(glancing_angle_hfs_new), np.sin(alpha_hfs)),
    marker="o",
)
ax.scatter(
    x[216:],
    z[216:],
    c=perc_diff(np.sin(glancing_angle_lfs_new), np.sin(alpha_lfs)),
    marker="o",
)
bar = f.colorbar(cs, ax=ax)
bar.set_label("sin(angle) perc diff [%]")
ax.set_aspect("equal")

# assert np.allclose(np.sin(glancing_angle_lfs_new), np.sin(alpha_lfs))
# assert np.allclose(np.sin(glancing_angle_hfs_new), np.sin(alpha_hfs))

# The heat fluxes are very very similar, except for where the incident angle is low
# The differences are entirely attributable to the sin(angle) term
f, ax = plt.subplots()
cs = ax.scatter(
    x,
    z,
    c=perc_diff(hf, hh),
    marker="o",
)
bar = f.colorbar(cs, ax=ax)
bar.set_label("Heat Flux percentage difference [%]")
ax.set_aspect("equal")
