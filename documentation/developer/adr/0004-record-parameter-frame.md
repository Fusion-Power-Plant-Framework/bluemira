# 1. Record architecture decisions

Date: 2022-06-30

## Status

Proposed

## Context

We want to make it easier to configure a `ParameterFrame` for a `ReactorDesigner`.

Currently we have all `Parameters` listed in `Configuration`. There are a number of downsides to this approach.
* Their definition is global meaning that no shared names can exist across different `Builder`s.
* `Builder`s take more `Parameters` than they probably need


Options:
* Define a `ParameterFrame` for each `Builder` with no default values.
    * Reconstruct the global list of required `Parameters` for a `ReactorDesigner` from
    the set of `Parameters` required for all of its `Builder`s.
* Nested / indexed / tagged `ParameterFrame` where `Builder`s would automatically tag
 a `Parameter` but the overall list would remain flat.
 * A `ConfigurationSchema`-like approach for each `Builder`.

e.g.
 ```python
 class BuilderOne(Builder):
     R_0: Parameter[float]
     A: Parameter[float]

 ```

## Decision

We will trial the three options above taking e.g. the EUDEMO `PlasmaBuilder` as an example.

## Consequences

The consequences of this decision are to be explored.
