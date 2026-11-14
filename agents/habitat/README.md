# Habitat agent package

An independently versioned Go module that composes the designer and storymaker
agents. Its `#create` entry is a Go handler. Each parent thread owns a persistent,
isolated designer child and storymaker child; the host resolves both declared
dependencies before registration succeeds.

Register `designer.Package()`, `storymaker.Package()` and `habitat.Package()` in
one batch, then call `agentruntime.RunTyped[habitat.Input, habitat.Output]` on a
habitat thread. The handler passes the completed design into the storymaker.
The runtime validates both boundaries and records child request IDs in its trace.

This module imports the two agents and `pkg/agentspec`. It does not import
GoLangGraph's graph engine, provider clients or persistence adapters. It can be
reused by another host that implements the Go contract.

Run the full consuming application with `make demo-agent-packages` from the
repository root. These are local development versions, not published releases;
see [the package guide](../../docs/AGENT_PACKAGES.md) for release steps.
