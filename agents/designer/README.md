# Designer agent package

A Go-native adaptation of the ideation example's habitat designer. Import this
module and register the result of `designer.Package()` with the host runtime.
Its entry point is `designer.AgentID`; the public types are `designer.Input` and
`designer.Output`.

The package carries an embedded prompt, a default model profile, generated
schemas and the `material_palette` tool. The host supplies model connections and
thread storage. The package does not import the graph engine or the storyteller.

```sh
go test -race ./...
```

The default model profile is configurable by the host. Live-model capabilities
are not certified by the offline tests. See the
[consumer example](../../examples/11-agent-packages/README.md) and
[distribution guide](../../docs/AGENT_PACKAGES.md). Local replaces must be replaced
with a released SDK dependency before this module is published.
