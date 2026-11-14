# Storymaker agent package

A Go-native adaptation of the ideation example's storyteller. Import this module
and register the result of `storymaker.Package()` with the host runtime. Its entry
point is `storymaker.AgentID`; public types are `storymaker.Input` and
`storymaker.Output`.

The package carries an embedded prompt, a default model profile, generated
schemas and the `story_outline` tool. It accepts a design description as ordinary
input and does not depend on the designer module or graph engine.

```sh
go test -race ./...
```

See the [consumer example](../../examples/11-agent-packages/README.md) and
[distribution guide](../../docs/AGENT_PACKAGES.md). Local replaces must be replaced
with a released SDK dependency before this module is published.
