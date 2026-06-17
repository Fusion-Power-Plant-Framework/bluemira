# 1. Caching vs. recreation of geometry from Matryoshkas

Date: 2022-06-15 - 2022-11-30

## Status

Proposed

## Context

At present we use FreeCAD Python wrapper around the open-source OpenCASCADE CAD kernel. This
has been the source of several issues over the years, which largely come down to the fact
that FreeCAD does not expose the entirety of the OpenCASCADE functionality.

Furthermore, FreeCAD requires installation via conda, which has been problematic from a
user perspective.

Other open-source packages wrap the entirety of OpenCASCADE functionality in Python, and are pip
installable.

## Decision

We will add a cadquery CAD backend, and progressively transition to using it as the default backend.
We will support the FreeCAD backend for a few release cycles, before removing it altogether.

## Consequences

The known consequence is the loss of a "native" CAD viewer, which represents CAD objects in a
window, without tesselation. This will only come when FreeCAD is removed. This is predominantly a
developer-level utility at present, and whilst it would be a loss, it is not insurmountable. One
can still view the CAD natively in any CAD program separately if the geometry is saved to file.

## Additional context

It is possible that future changes, either in terms of moving from FreeCAD to
another library, or in terms of FreeCAD's scheduled v1 release, will change
the landscape of this decision, and we will need to re-assess.
