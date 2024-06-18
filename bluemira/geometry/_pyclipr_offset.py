import numpy as np
import pyclipr

# Tuple definition of a path
path = [(0.0, 0.0), (0, 105.1234), (100, 105.1234), (100, 0), (0, 0)]
path2 = [(1.0, 1.0), (1.0, 50), (100, 50), (100, 1.0), (1.0, 1.0)]

# Create an offsetting object
po = pyclipr.ClipperOffset()

# Set the scale factor to convert to internal integer representation
po.scaleFactor = 1000

# add the path - ensuring to use Polygon for the endType argument
# addPaths is required when working with polygon - this is a list of correctly orientated paths for exterior
# and interior holes
po.addPaths([np.array(path)], pyclipr.JoinType.Miter, pyclipr.EndType.Polygon)

# Apply the offsetting operation using a delta.
offsetSquare = po.execute(10.0)

# Create a clipping object
pc = pyclipr.Clipper()
pc.scaleFactor = 1000

# Add the paths to the clipping object. Ensure the subject and clip arguments are set to differentiate
# the paths during the Boolean operation. The final argument specifies if the path is
# open.
pc.addPaths(offsetSquare, pyclipr.Subject)
pc.addPath(np.array(path2), pyclipr.Clip)

""" Test Polygon Clipping """
# Below returns paths
out = pc.execute(pyclipr.Intersection, pyclipr.FillRule.EvenOdd)
out2 = pc.execute(pyclipr.Union, pyclipr.FillRule.EvenOdd)
out3 = pc.execute(pyclipr.Difference, pyclipr.FillRule.EvenOdd)
out4 = pc.execute(pyclipr.Xor, pyclipr.FillRule.EvenOdd)

# Using execute2 returns a PolyTree structure that provides hierarchical information inflormation
# if the paths are interior or exterior
outB = pc.execute2(pyclipr.Intersection, pyclipr.FillRule.EvenOdd)

# An alternative equivalent name is executeTree
outB = pc.executeTree(pyclipr.Intersection, pyclipr.FillRule.EvenOdd)


""" Test Open Path Clipping """
# Pyclipr can be used for clipping open paths.  This remains simple to complete using the Clipper2 library

pc2 = pyclipr.Clipper()
pc2.scaleFactor = int(1e5)

# The open path is added as a subject (note the final argument is set to True)
pc2.addPath(((40, -10), (50, 130)), pyclipr.Subject, True)

# The clipping object is usually set to the Polygon
pc2.addPaths(offsetSquare, pyclipr.Clip, False)

""" Test the return types for open path clipping with option enabled"""
# The returnOpenPaths argument is set to True to return the open paths. Note this function only works
# well using the Boolean intersection option
outC = pc2.execute(pyclipr.Intersection, pyclipr.FillRule.NonZero)
outC2, openPathsC = pc2.execute(
    pyclipr.Intersection, pyclipr.FillRule.NonZero, returnOpenPaths=True
)

outD = pc2.execute2(pyclipr.Intersection, pyclipr.FillRule.NonZero)
outD2, openPathsD = pc2.execute2(
    pyclipr.Intersection, pyclipr.FillRule.NonZero, returnOpenPaths=True
)

# Plot the results
pathPoly = np.array(path)

import matplotlib.pyplot as plt

plt.figure()
plt.axis("equal")

# Plot the original polygon
plt.fill(
    pathPoly[:, 0],
    pathPoly[:, 1],
    "b",
    alpha=0.1,
    linewidth=1.0,
    linestyle="dashed",
    edgecolor="#000",
)

# Plot the offset square
plt.fill(
    offsetSquare[0][:, 0],
    offsetSquare[0][:, 1],
    linewidth=1.0,
    linestyle="dashed",
    edgecolor="#333",
    facecolor="none",
)

# Plot the intersection
plt.fill(out[0][:, 0], out[0][:, 1], facecolor="#75507b")

# Plot the open path intersection
plt.plot(
    openPathsC[0][:, 0],
    openPathsC[0][:, 1],
    color="#222",
    linewidth=1.0,
    linestyle="dashed",
    marker=".",
    markersize=20.0,
)
