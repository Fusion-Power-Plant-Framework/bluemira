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
* `Builder` does everything

## Decision

We will use Architecture Decision Records,
as [described by Michael Nygard](http://thinkrelevance.com/blog/2011/11/15/documenting-architecture-decisions).

The templates can be adapted to include other relevant headers
(e.g. 'Alternative Solutions Considered'),
but each document should contain the headers used in this ADR.

## Consequences

See Michael Nygard's article, linked above.
For a lightweight ADR toolset,
see Nat Pryce's [adr-tools](https://github.com/npryce/adr-tools).
