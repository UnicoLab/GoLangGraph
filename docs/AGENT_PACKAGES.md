# Go-native agent packages (experimental)

An agent package contains typed contracts, Go tools, embedded prompts, model
profiles, Go handlers and declared child agents. It is distributed as a Go module
and linked into a consuming application. GoLangGraph supplies graph execution,
isolated persistent threads, dependency resolution and request deduplication.

This is a Go-native exploration of the authoring and composition ideas in the
[Standard Agent Specification](https://standardagentspec.org/). It has its own
`golanggraph.agent/v1alpha1` contract. It does not execute TypeScript, load npm
packages or claim compatibility with that specification.

## Run the complete example

From the repository root:

```sh
make demo-agent-packages
make test-agent-packages
```

The application imports three independently versioned agent modules:

- `agents/designer`: adapts the sustainable habitat designer from the ideation
  examples, including a material-selection tool and an embedded prompt.
- `agents/storymaker`: adapts the storyteller, including an outline tool and its
  own embedded prompt. It does not import the designer.
- `agents/habitat`: a reusable Go handler that declares both agents as children
  and passes the designer's typed output into the storymaker.
- `examples/11-agent-packages`: the consuming application. It supplies model
  connections and storage. Its optional `-workflow graph` mode also demonstrates
  application-owned composition with a two-node GoLangGraph graph.

The default demo uses a clearly identified deterministic model double. Package
registration, input/output validation, graph execution, tool execution and
checkpointing are real. The default run needs no credentials or services and
creates no persistent files. See the example README for live Ollama and restart
commands.

## Contract and runtime boundary

`pkg/agentspec` depends only on the Go standard library. It defines immutable
packages, definitions, model profiles, tools, child dependencies and generated
schemas. None of the three agent modules imports the graph engine, persistence
or provider clients. Habitat imports its dependencies' exported contracts.

`pkg/agentruntime` is the GoLangGraph adapter. It executes prompt turns as a graph:

```text
model -> tools -> model -> ... -> done
```

Go handlers run as a graph node and can call their declared tools and children.
Prompt agents see declared children as tools with the child's input schema. A
child gets one persistent thread per parent thread and alias, with its own model,
history, values and run journal. Dependency cycles are rejected. A dependency tree
is limited to 128 threads and a turn to 256 tool/child invocations; prompt turns
also have a configurable model-call budget of 1–100.

The outer application can compose packaged agents using arbitrary GoLangGraph
workflows. The existing `pkg/agent` API and its behavior remain available.

| Package supplies | Host supplies |
| --- | --- |
| Embedded prompt and tool implementations | Model connections and credentials |
| Typed contracts, schema constraints and semantic validation | Conversation instances and storage |
| Default model profile and model-call budget | Explicit model overrides for new threads |
| Go handlers and exact child dependencies | Dependency validation and isolated child instances |
| Module/version identity and entry-point metadata | Run journal, deployment and UI |

Agent references have the form `module/path#entry_name`. A host accepts one
version of a module at a time. Threads record both the package version and their
effective model profile. Each thread also pins a SHA-256 fingerprint of its
declarative contract: prompt, schemas, tools, model defaults and child references.
Changed versions or fingerprints fail before execution. The fingerprint cannot
detect changes inside Go callbacks; versioning executable behavior remains the
package author's responsibility. It is not an artifact signature or sandbox.

## Author a package

Define ordinary Go input/output structs with JSON tags. Use
`agentspec.NewTool[Input, Output]` for tools and
`agentspec.NewAgent[Input, Output]` for prompt agents. Export a `Package()` function
that returns `agentspec.NewPackage(modulePath, version, description, definition)`.

Use `agentspec.NewHandler[Input, Output]` for a turn implemented in ordinary Go.
It receives an `agentspec.Execution` with thread-local JSON values and access to
declared tools and agents. A host containing only handler agents needs no model
provider. Declare children with `agentspec.NewSubagent[Input, Output]`, including
their qualified entry ID and exact package version. Call them through
`agentspec.CallAgent[Input, Output](ctx, execution, alias, input)`. The habitat
module demonstrates the complete pattern.

Scaffold a new module from the repository root:

```sh
go run ./cmd/golanggraph package init example.com/my-agent \
  --dir /tmp/my-agent --sdk-dir "$PWD"
cd /tmp/my-agent
go test ./...
go run ./cmd/manifest > manifest.json
```

The generator creates a prompt entry, Go handler, typed tool, embedded prompt,
tests and manifest exporter. It refuses to overwrite an existing directory.
After building the CLI, inspect the exported data with
`golanggraph package inspect manifest.json`. Inspection validates the contract
without loading code; exporting a manifest executes the module you imported.
Once an SDK release exists, use `--sdk-version <released-version>` instead of
`--sdk-dir` to scaffold without a local replace.

The designer and storyteller modules are complete authoring examples. Their
`prompt.txt` assets are embedded with `go:embed`, so deployed binaries do not need
the source tree or a special working directory.

Generated schemas support plain structs, slices, strings, booleans and numeric
types. Exported fields are required unless tagged `omitempty`; undeclared fields
and null values are rejected. Pointer types, maps, byte slices, embedded fields,
custom JSON encoders and JSON tag options other than `omitempty` are currently
unsupported and fail definition creation. An empty array must be a non-nil slice.

String fields support `description`, `enum` (comma-separated values), `minLength`
and `maxLength` tags. These constraints appear in the exported schema and are
enforced at runtime; lengths count Unicode characters. Duplicate JSON keys,
unknown fields, trailing values and excessive nesting are rejected. Schemas are
limited to 32 levels and incoming JSON to 64. Invalid tags fail construction.

For domain rules, implement `Validate() error` on an input/output type. These
rules run before invoking tools or models and before accepting outputs, but are
not automatically inferred into the structural schema. Use schema tags for rules
that a model should see and `Validate` for additional domain rules.

Tool functions receive an invocation-scoped `agentspec.State` for JSON values.
Keep per-conversation data there. Do not capture mutable conversation data in
package globals or retain the state handle after a tool returns. Definitions and
their Go callbacks can be shared across concurrently executing threads. Handlers
must finish their work before returning; do not leave background goroutines
using an execution handle. Respect the supplied context in external operations.

## Consume a package

Applications import the modules, call their `Package()` functions, and register
them with `runtime.Register(...)`. Registration of a batch is atomic and resolves
dependencies against the whole batch regardless of order. Missing dependencies,
version/schema mismatches and cycles are errors. Definitions and manifests
returned to callers do not expose mutable runtime configuration.

Create a thread with `runtime.CreateThread(ctx, agentID)`, then use
`runtime.Run(ctx, threadID, input)` or the generic
`agentruntime.RunTyped[Input, Output](...)` helper. The definition validates the
actual input/output wire data. `RunTyped` checks the exact exported Go types (or
aliases) before execution, so a wrong output type cannot trigger side effects
before failing to decode. `CallAgent` checks the declared child schemas first.

Configure the host with a `llm.ProviderManager` or another `Completer`, a
checkpointer, and optional model overrides keyed by agent ID. Credentials never
need to be embedded in packages. `runtime.Catalog()` returns JSON-serializable
manifests containing prompts, model profiles, schemas and tool descriptions for
inspection or future Studio integration; the Studio UI is not changed here.

## Distribution and release

The three agents have separate `go.mod` files and may be versioned independently.
This checkout uses local `replace` directives so it works before publication.
The consumer declares all four replaces explicitly because a dependency's
replace directives are not inherited.

These versions are local development identities, not published releases. Before
publishing:

1. Release the root GoLangGraph module containing the SDK and adapter.
2. Replace each agent module's `v0.0.0` SDK requirement with that real release.
3. Remove development replaces, run `go mod tidy`, and test outside the checkout.
4. Match each package's `Version` constant to its release. In this monorepo, the
   first agent tags would be `agents/designer/v0.1.0` and
   `agents/storymaker/v0.1.0` and `agents/habitat/v0.1.0`.
5. Consumers can then select released module versions using ordinary `go get`,
   import them and rebuild their application. Commit their `go.mod` and `go.sum`.

No registry service, runtime source compiler, shared-library plugin loader or
Go workspace file is required. Custom Go code is installed at build time. The
manifest is introspection data, not a loader for executable code. Manifest
capabilities do not sandbox code linked into the host.

## Execution guarantees and current limits

- Each thread owns its conversation, JSON tool values and model profile. Multiple
  threads may execute concurrently. Turns on one thread wait for an in-process
  lock; FIFO scheduling is not promised. Waiting is cancelable.
- Tools execute sequentially in model-returned order. Only tools declared by the
  agent are available. Errors become correlated tool observations so a model can
  correct its arguments within the call budget. Failed tools discard value writes.
- Inputs, tool arguments/results and final model outputs are validated. A final
  answer must be a JSON object; the adapter does not silently repair malformed
  answers. Provider failures and cancellation propagate to the caller.
- Every request records durable intent **before** executing models, handlers or
  tools. Successful turns commit their history, JSON values, result and trace
  together. Storage failures are errors. File-backed tests cover reopening a
  host and preserving large JSON integers. Memory storage is the default and
  does not survive a process restart.
- Failed turns retain a failed journal entry but discard that thread's new
  messages and values. A successful child turn remains committed if its parent
  later fails. External effects also remain external: a parent/child workflow is
  not a distributed transaction.
- Thread-tree creation writes separate records. A reported storage failure
  triggers best-effort cleanup; a process crash can leave partial, unreachable
  records. The runtime does not claim atomic creation across multiple threads.
- One runtime must own a given thread's writes. Locks are in-process, not database
  leases. Multiple hosts sharing a checkpointer must route each thread to one
  owner until distributed locking is implemented.
- Registration pins package versions, but updating persistent state across
  versions requires an explicit future migration design.

## Request IDs, recovery and traces

Supply a stable request identity when callers may redeliver a request:

```go
result, err := host.RunWithOptions(ctx, threadID, input, agentruntime.RunOptions{
    RequestID: "order-123",
})
```

`RunTypedWithOptions` provides the same behavior with an exact Go type check.
`Run` and `RunTyped` generate a new identity each time, suitable for new turns.
IDs are scoped to a thread and accept 1–128 ASCII letters, digits, dots, colons,
underscores or hyphens, starting with a letter or digit.

| Stored state for this request ID | Result of redelivery |
| --- | --- |
| Succeeded, same validated input | Return the saved result; execute nothing |
| Any state, different input | `ErrRequestConflict` |
| Failed or abandoned | `ErrRunFailed`; deliberate retry needs a new ID |
| Running after an interruption | `ErrRunIndeterminate`; further work is blocked |

Input comparison canonicalizes object key order and preserves JSON numbers. It
does not treat every mathematically equal numeric spelling as the same input.
If a process dies or a final commit fails, the persisted running record prevents
automatic replay. Inspect external effects and the child threads, then call
`host.AbandonRun(ctx, threadID, requestID, reason)` to acknowledge the interruption.
This does not undo work. Resolve interrupted children as well before submitting a
new parent request. Use idempotent external services for stronger effect handling.

Each thread retains up to `Config.MaxRecordedRuns` journal entries (default 1000).
The runtime refuses new requests when full; it never silently drops idempotency
keys. Existing succeeded IDs can still be read. Start a new thread deliberately
when the journal limit is reached. This bounds record count, not input/output or
history byte size; hosts must apply suitable request and storage limits.

`ReadThread` exposes journal status and ordered execution events. Events contain
model/tool names and correlated child thread/request IDs, without raw payloads.
They are committed with the result or failure, not after every step; a crash can
leave only the initial `run_started` event. Results count direct model/tool/child
calls on that thread; descendant usage is available in the child's journal.
No live streaming transport, durable message queue or distributed lease is
implemented by this adapter.

## Which concepts should move to Go?

The useful transfer from the [Standard Agent Specification's overview](https://standardagentspec.org/)
is the separation between reusable definitions, running thread instances and
host infrastructure. This implementation makes those concepts ordinary Go
packages and interfaces. It does not promise specification conformance.

| Concept | Go-native choice and current coverage |
| --- | --- |
| Providers and models | Host `Completer` port; explicit package defaults and pinned host overrides |
| Prompts and tools | Embedded assets, typed Go callbacks and validated schema boundaries |
| Handlers | Pure Go typed functions executed by a graph node |
| Agents and subagents | Immutable definitions, declared versioned dependencies, isolated persistent children |
| Threads and execution state | Checkpointer-backed history/values, bounded turns and request journal |
| Type registry | Exported Go types, generics, compiler imports and runtime boundary checks |
| Packaging and distribution | Go modules, source releases, embedded assets and host builds |
| Lifecycle hooks | Typed validation exists; a general interception/compaction hook API is deferred |
| Effects and scheduling | Deferred; requires durable delivery and retry semantics beyond turn execution |
| Files, sandboxes and remote endpoints | Deferred; these need explicit host capability and isolation designs |
| Users, streams and client UI | Host concerns; no new principal model, streaming protocol or Studio UI here |
| Decisions and sub-prompts | Can be authored as handlers/child agents; no dedicated API yet |

The next useful extension is a durable host service with ownership leases,
recovery tooling and bounded history management. Scheduling, runtime code
loading and marketplace services should follow concrete deployment requirements,
not become mandatory dependencies of the authoring SDK.
