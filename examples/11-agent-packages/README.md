# Compose independent Go agent modules

This application imports three independent agent modules: designer, storymaker
and habitat. The habitat package owns the composition in a Go handler; the host
supplies providers and persistence. It also includes an alternative outer graph
workflow and a deterministic model double for offline use.

```sh
go run .
go run . -catalog
go run . -workflow graph
go test -race ./...
```

The default mode requires no network, model server or credentials. Its output
explicitly identifies the model double. It exercises the actual graph engine,
packaged tools, validation and thread storage.

## Continue across process restarts

```sh
go run . -state-dir /tmp/golanggraph-package-demo -request-id first
```

Copy `habitat_thread` from the output into the next command:

```sh
go run . -state-dir /tmp/golanggraph-package-demo \
  -resume-habitat '<habitat_thread>' -request-id second \
  -brief 'Adapt the design to harvest rainwater'
```

`designer_tool_uses` increases from 1 to 2. Both child thread IDs stay the same and
both children preserve their own histories. Repeat the second command unchanged:
the saved result is returned and the counter remains 2. Reuse `second` with a
different brief and the runtime reports a request conflict. Use a new request ID
for a new turn. The output includes the parent trace with child request IDs.

The optional `-workflow graph` mode uses application-owned graph nodes and accepts
`-resume-designer` instead. It creates a new storymaker thread for each workflow.
Request IDs and parent/child orchestration are demonstrated in `package` mode.

Interrupted requests block further turns. Inspect their child/external effects,
then call the host API `AbandonRun` before deliberately submitting a new request.
There is no automatic replay of unfinished work.

## Use a real model

With Ollama running and a tool-capable model already installed:

```sh
go run . -live -model '<installed-model-name>'
```

Use `-endpoint` to choose another Ollama endpoint. Live runs require tool calling
and schema-shaped JSON output. The package default profile retains the original
examples' `gemma3:1b` setting; support and quality depend on the model. Offline
tests do not certify a live model. Overrides apply to new threads; existing
threads retain their stored model profile.

The [agent package guide](../../docs/AGENT_PACKAGES.md) explains authoring,
distribution, versioning and the prototype's execution guarantees. The local
replaces are explicit development wiring, not published package releases.
