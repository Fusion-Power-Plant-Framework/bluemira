# 4. Record ParameterFrame decisions

Date: 2022-06-30
Updated: 2022-10-24

## Status

Completed

## Context

We want to make it easier to configure a `ParameterFrame` for a `ReactorDesigner`.

Currently we have all `Parameters` listed in `Configuration`. There are a number of downsides to this approach.

* Their definition is global meaning that no shared names can exist across different `Builder`s.
* `Builder`s take more `Parameters` than they probably need.

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

We will trial the three options above  taking e.g. the EUDEMO `PlasmaBuilder` as an example.

#### 2022-10-24

The final design landed on consists of some of both options:

- A dedicated `ParameterFrame` is created for each object that uses one

  - This replaces the list of strings we were previously setting to the `_required_params` class attribute.

- No default values are stored on a `ParameterFrame`

  - A `MappedParameterFrame` has defaults and is used exclusively for external codes.
    Not all of an external codes variables are mapped with bluemira variables but defaults are in some cases set for them.
    These variables can be updated through the solver interface via the `problem_settings` key in `build_config`.

-  Only the value and source of a `Parameter` have an API entry point to be updated after creation

- A `ParameterFrame` is defined similarly to the below.

  ```python
  from dataclasses import dataclass
  from bluemira.base.parameter_frame import (
      Parameter,
      ParameterFrame,
  )

  @dataclass
  class MyFrameForAJob(ParmeterFrame):
     R_0: Parameter[float]
     A: Parameter[float]

  ```

A decorator to allow for easier definition of a `ParameterFrame` was also developed but typing issues need to be fixed to enable its use:

  ```python
  from bluemira.base.parameter_frame import (
      Parameter,
      parameter_frame,
  )

  @parameter_frame
  class MyDecoratedFrameForAJob:
      B: Parameter[float]
      z_0: Parameter[float]
  ```

This method has been remove in [#1607](https://github.com/Fusion-Power-Plant-Framework/bluemira/pull/1607) as it is confusing to have two methods to define a `ParameterFrame`.
We may add this back in future if we manage to fix the typing issue.

Further work will be done to add helper methods to collate all `Parameters` required in a given piece of code

## Consequences

* All existing metadata has been removed from bluemira and should be stored in a `Parameter` JSON file for a given reactor design.

* All `Parameters` need to be defined for a given design.

* There are now many different ParameterFrame classes instead of one global one.

* Reactor Designers are expected to define their own parameter frames.
