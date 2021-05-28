cad
===


CAD tutorial
------------


This is a little tutorial on how to make CAD in bluemira.

The cad module is basically a wrapper around an existing python interface to a much lower level library (OCE).

With it, you can convert geometry objects (Loop, Shell) into CAD objects, using terminology familiar to those who have used CAD programs in the past.

The basic idea behind any 3-D CAD is to start with some primitives (points, lines, splines, etc.) to make 2-D objects, to then make 3-D objects.

A lot of this module simplifies out the first two steps, leaving you to worry about what you want to make.


.. note::

	The `cad` module makes heavy use of the `geometry` module. We recommend you start there





