# 1. Record architecture decisions

Date: 2022-06-30

## Status

Proposed

## Context

At present we have a `design_stage` decorator which wraps methods in the `Reactor` base
class. The `["run", "read", "mock"]` logic is presently handled by `Builder`s and `Solver`s.

We want `Builder` to effectively only return a `Component` or tree of `Component`.

We want `Builder` to be agnostic of the design problem and of `["run", "read", "mock"]` logic.

We want `Builder` to have a minimum set of `Parameters`.

`design_stage` presently only handles printing logic for entry and exit to an operation.

Options:
* Do nothing
* Add a `DesignStage` class which handles the `["run", "read", "mock"]` logic.
    * Only for `Builder`
    * For `Builder` and `Solver`
* Should every `DesignStage` be a `Solver` where `["run", "read", "mock"]` logic is also
handled.



## Decision

We add a `DesignStage` class to handles the  `["run", "read", "mock"]` logic for
`Builder` only.

## Consequences
