# `ComponentManager`

Date: 2022-06-30

## Status

Proposed

## Context

At present access to information relevant to a given `Component` is difficult / unintuitive
to retrieve. In many situations we want short-hand access to these characteristics / attributes / properties.

Options:
* `Component` is more "active" and directly has e.g. properties and methods to retrieve information
* `ComponentManager` is used to provide access to information relating to an abstract `Component`
    * At the `ReactorDesigner` level or `Component` level.
* `Builder` does everything (builds the `Component` and logic + access methods to extract information + store the `Component`)

It will be quite difficult to have a totally "general" `ComponentManager`.

Many of the properties are not specific to a `Component` but maye require access to "external"
information present in the `Reactor` or other `Component`s.

## Decision

We will trial the three options above taking e.g. the EUDEMO plasma `Component` as an example
to inform our decision.

## Consequences

The consequences of this decision are to be explored.
