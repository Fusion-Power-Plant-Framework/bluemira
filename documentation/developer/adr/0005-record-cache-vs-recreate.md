# 1. Record architecture decisions

Date: 2022-06-14

## Status

Proposed

## Context

At present we recreate FreeCAD geometries by travelling through the boundaries of their
"Matryoshka". We have a number of issues relating to this as our implementation apppears
 to be imperfect.

Assumption: `BluemiraGeo` objects are "static".

Options:
* Continue with recreating the shapes
    * Pros:
        * We can track labels throughout the creation of geometry
        * Meshing and FE problems are easier to set up because of the above
    * Cons:
        * Higher computational cost (recreate upon each call of geometry)
        * Higher maintenance overhead
        * Reverse engineering of FreeCAD is presently imperfect
            * An operation can work in FreeCAD but fail upon "reconstruction" in Bluemira
            * NotClosedWireError, DisjointedFaceError, ...
* Cache the FreeCAD shape
    * Pros:
        * Lower computational cost
        * Lower maintenance cost
        * Less logic and code
    * Cons:
        * If we want to label specific boundaries of geometry, we need to travel through
        all the boundaries and find the "right" ones and label post-instantiation (fiddly!)
        * This means that we cannot track shared boundaries (upon creation, but see above)
        * Meshing and FE problems are more complicated to set up

## Decision

We will investigate caching, and see how tricky it is to set up a simple FE problem with
a shared boundary.

## Consequences

The consequences of this decision are to be explored.
