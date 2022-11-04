# 2. `DesignStage` architectural decision

Date: 2022-06-30
Updated: 2022-10-24

## Status

Completed

## Context

At present we have a `design_stage` decorator which wraps methods in the `Reactor` base
class. The `["run", "read", "mock"]` logic is presently handled by `Builder`s and `Solver`s.

We want `Builder` to effectively only return a `Component` or tree of `Component`.

We want `Builder` to be agnostic of the design problem and of `["run", "read", "mock"]` logic.

We want `Builder` to have a minimum set of `Parameters`.

`design_stage` presently only handles printing logic for entry and exit to an operation.

Options:
* Do nothing
* Add a `DesignStage` class which handles the `["run", "read", "mock"]` logic for `Builder`

* Should every `DesignStage` be a `Solver` where `["run", "read", "mock"]` logic is also
handled?

## Decision

We add a `DesignStage` class to handles the  `["run", "read", "mock"]` logic for
`Builder` only.

We will trial the two options above taking e.g. the EUDEMO `PlasmaBuilder` as an example
to inform our decision.

#### 2022-10-24

The `Designer` (previously called the `DesignStage`) is in charge of solving a minimal design problem.
In its simplest form the output of the `Designer`s `execute` method is an input to the `Builder`.
A `Designer` can have different run modes as described above but by default only needs a run method to function.

With user inputs and the output of the `execute` method the `Builder` creates the CAD for a given `Component`.
The `Builder` will output a `ComponentManager` containing the `Component` built by the `Builder`.

Both a `Designer` and `Builder` need their own `ParameterFrame` to define the `Parameters` needed.

A `ParameterFrame` should only be needed at this level and a `Designer` and `Builder` are responsible for updating the
`ParameterFrame` where appropriate. Lower level functionality should use the `Parameter` value as input.

Further work will be undertaken so that `Solver` can inherit from `Designer` functioning as a multistage `Designer`.

## Consequences

The separation of `Designer` and `Builder` mechanics required a rewrite of the existing complex reactor design.

A reactor workflow author will often need to define a `Designer` and a `Builder` for a component.
Previously they would only define a `Builder`.
Each `Designer` or `Builder` now requires its own `ParameterFrame` if `Parameters` are needed for initialising or executing the object.
