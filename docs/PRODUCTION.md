# Running GoLangGraph in production

This document covers the parts of GoLangGraph you need to configure
deliberately before exposing it to real traffic: authentication, cross-origin
access, tool sandboxing, durable execution and health checking.

Defaults are chosen so that local development works out of the box. Several of
them are **not** the right choice for a deployment, and each is called out
below.

---

## 1. Server security

`ServerConfig.Security` controls authentication, allowed origins and request
limits. It defaults to `DefaultSecurityConfig()`, which is permissive.

```go
cfg := server.DefaultServerConfig()
cfg.Security = &server.SecurityConfig{
    RequireAuth:     true,
    APIKeys:         []string{os.Getenv("GOLANGGRAPH_API_KEY")},
    AllowedOrigins:  []string{"https://studio.example.com"},
    MaxRequestBytes: 4 << 20,
    PublicPaths:     []string{"/api/v1/health"},
}
srv := server.NewServer(cfg)
```

| Setting | Default | What to set in production |
| --- | --- | --- |
| `RequireAuth` | `false` | `true`. With it off, every endpoint is unauthenticated. |
| `APIKeys` | empty | At least one key. Clients send it as `X-API-Key`. Keys are compared in constant time. |
| `AllowedOrigins` | empty (any) | The exact origins that may call the API. This list also governs **WebSocket** upgrades. |
| `MaxRequestBytes` | 4 MiB | Lower it if your payloads are small. |
| `PublicPaths` | `/api/v1/health` | Paths that skip authentication, for load-balancer probes. |

### Both servers, not just one

There are two serving paths in this project and **each needs configuring**:

| Type | Constructor | Used by |
| --- | --- | --- |
| `server.Server` | `server.NewServer(cfg)` | The `serve` command and the `/api/v1` API that Studio talks to |
| `server.AutoServer` | `server.NewAutoServer(cfg)` | The `auto-serve` command and auto-generated per-agent endpoints |

`AutoServerConfig` takes the same `Security` value:

```go
cfg := server.DefaultAutoServerConfig()
cfg.Security = &server.SecurityConfig{
    RequireAuth:    true,
    APIKeys:        []string{os.Getenv("GOLANGGRAPH_API_KEY")},
    AllowedOrigins: []string{"https://studio.example.com"},
    PublicPaths:    []string{"/health"},
}
cfg.MaxRequestSize = 4 << 20
```

On `AutoServer`, panic recovery, the request size limit and the security headers
are unconditional. They were previously opt-in through the `Middleware` name
list, so a deployment that left `"recovery"` out of that list had a panicking
handler tear down the connection.

**Agent registry isolation.** `NewAutoServer` serves from the process-wide agent
registry, so two auto-servers in one process see each other's agents — which is
rarely what you want if they differ in exposure or credentials. Use
`server.NewAutoServerWithRegistry(cfg, agent.NewAgentRegistry())` to give a
server its own. Endpoints also cannot be regenerated once `Start` has been
called; register agents before starting.

**Why the origin list matters for WebSockets.** A WebSocket upgrade that accepts
any origin allows cross-site WebSocket hijacking: any page a signed-in user
visits can open a socket to your server and drive it as that user. Setting
`AllowedOrigins` closes that. Leaving it empty accepts any origin and is only
appropriate for local development.

Authentication fails closed: if `RequireAuth` is true and no keys are
configured, every request is rejected rather than allowed.

### Always-on protections

These apply regardless of configuration:

- Request bodies are size-limited.
- Handler panics become a 500 with a generic JSON body; the stack is logged, never sent to the client.
- `X-Content-Type-Options`, `X-Frame-Options` and `Referrer-Policy` are set.
- Cross-origin preflight is answered for every route.

---

## 2. Tool sandboxing

Tool arguments are produced by a language model, which may be steered by
untrusted input. Treat every built-in tool as attacker-controlled and bound it
with a `tools.SecurityPolicy`.

```go
policy := tools.DefaultSecurityPolicy()
policy.SetAllowedRoots([]string{"/srv/agent-workspace"})
policy.MaxOutputBytes = 256 << 10
policy.AllowedCommands = []string{"ls", "wc"}
policy.AllowedHosts = []string{"api.example.com"}

read := tools.NewFileReadTool()
read.SetSecurityPolicy(policy)
```

**Filesystem tools** are confined to `AllowedRoots` (working directory and temp
directory by default). Paths are resolved and symlinks followed during
validation, so a link planted inside an allowed root cannot reach outside one.
An extension allowlist alone is not a boundary: `.json` and `.yaml` files
elsewhere on a host include kubeconfigs, registry credentials and cloud
credential files.

**The shell tool** runs an allowlisted command directly, never through a shell.
Note what the allowlist cannot contain: `find` executes arbitrary programs via
`-exec`, and `cat`/`grep` read any file the process can reach, so none of them
are in the default set. Arguments containing program-executing flags or shell
metacharacters are rejected, as is a command given as a path.

**The HTTP tool** refuses loopback, private, link-local, multicast and
unspecified addresses unless `AllowPrivateNetwork` is set. This blocks
server-side request forgery, including the cloud instance metadata endpoint at
`169.254.169.254`. The check runs at dial time against the resolved address, so
a hostname that resolves inward is caught even if it resolved elsewhere a
moment earlier, and every redirect hop is re-validated.

If your agent legitimately needs an internal service, prefer
`AllowedHosts` over `AllowPrivateNetwork`.

---

## 2b. Multi-agent deployments

`MultiAgentManager` serves its own HTTP surface, and several of its controls
were previously inert. Configure them deliberately:

- **Authentication** accepted any non-empty API key. It now compares against
  configured keys in constant time, and enabling auth without configuring keys
  fails construction rather than admitting everyone.
- **`GET /config`** returned provider API keys, database, cache and SMTP
  passwords, the Slack webhook and both secret maps verbatim to any caller.
  The response is redacted; treat any deployment that exposed it as having
  leaked those credentials.
- **Rate limiting** was configured and did nothing. Global, per-IP and
  per-agent limits are enforced and return 429 with `Retry-After`. `per_user`
  is not enforceable without request identity and is logged loudly at startup
  rather than silently ignored.
- **`GET /health`** always answered 200 "healthy", so every liveness probe
  passed regardless of agent state. It now aggregates and returns 503.
- **Routing rule conditions** were parsed, stored and never compared against a
  request, so a rule guarded by a condition matched everything.
- **Health checkers** leaked a goroutine per agent that nothing could stop.
  They now start with `Start` and are cancelled and joined by `Stop`.

Config validation is stricter as a result: patternless rules, duplicate rule
IDs, invalid regexes, negative timeouts, inverted scaling bounds and routes to
unknown or disabled agents are now rejected at load. A configuration that was
previously accepted while being partly inert may now fail to start — that is
the point.

## 3. Durable execution and resume

Attach a checkpointer to persist state after every node, so a run that dies
mid-graph can resume instead of starting over.

```go
checkpointer := persistence.NewFileCheckpointer("/var/lib/golanggraph/checkpoints")
saver := persistence.NewCheckpointSaver(checkpointer)

graph.WithCheckpointer(saver, threadID)
```

To resume after a crash, load the most recent checkpoint and restart from the
node that should run next. `latest.NodeID` is the node that *completed*, so
resume from its successor:

```go
latest, err := persistence.Latest(ctx, checkpointer, threadID)
if err != nil {
    return err
}
if latest != nil {
    next, err := graph.GetNextNodes(ctx, latest.NodeID, latest.State)
    if err != nil {
        return err
    }
    if len(next) > 0 {
        _, err = graph.ExecuteWithOptions(ctx, latest.State, &core.ExecuteOptions{
            ThreadID:  threadID,
            StartNode: next[0],
        })
    }
}
```

Thread and checkpoint identifiers become path components in the file backend and
are validated; values containing separators or `..` are rejected.

### Database backends

- **PostgreSQL** could not be used as a `Checkpointer` at all: `checkpoints`
  has a foreign key to `threads`, nothing in the interface created the thread
  row, so the first save of any new thread failed on the constraint. The thread
  is now upserted in the same transaction.
- **The RAG document path never worked.** `SaveDocument` and vector search
  passed a Go map and `[]float64` straight to `database/sql`, which rejects
  both, so every call failed. Embeddings read back were also discarded.
- **Redis leaked state between threads.** Keys were built by joining the thread
  and checkpoint IDs with `:` without escaping, so thread `x:a` + checkpoint
  `b:c1` collided with thread `x:a:b` + checkpoint `c1` — loading one thread
  could return another's checkpoint. Thread IDs are commonly user- or
  session-derived, making this a cross-tenant leak. Segments are now escaped
  and the loaded thread ID is verified.
- **Redis expiry was hardcoded to 24 hours**, so every deployment silently lost
  its checkpoints after a day. Set `DatabaseConfig.CheckpointTTL`.
- `rows.Err()` was never checked when listing, so a connection dropping
  mid-iteration returned a silently truncated list with a nil error — and
  `Latest()` is built on that list, so a resume could silently pick up from the
  wrong checkpoint.

Postgres and Redis now have real integration tests. They skip with an explicit
message when no server is reachable, and hard-fail if `POSTGRES_TEST_DSN` or
`REDIS_TEST_ADDR` is set but unreachable.

### Human-in-the-loop

```go
graph.Config.InterruptBefore = []string{"apply_changes"}

_, err := graph.Execute(ctx, state)

var interrupt *core.InterruptError
if errors.As(err, &interrupt) {
    interrupt.State.Set("amount", reviewedAmount) // a person edits the state
    graph.Config.InterruptBefore = nil
    final, err := graph.Resume(ctx, interrupt)
}
```

An interrupt is a normal, resumable outcome, not a failure. Over HTTP it is
reported as `200` with `"status": "interrupted"`.

---

## 4. Health checking

`golanggraph health` answers two different questions; pick the right one.

```bash
# Is this server serving? Use this as a container health check.
golanggraph health --server http://127.0.0.1:8080

# Are this host's configured dependencies reachable?
POSTGRES_HOST=db REDIS_HOST=cache golanggraph health
```

The dependency scan probes only the services that are actually configured
(`POSTGRES_HOST`, `REDIS_HOST`, `OLLAMA_URL`), because defaulting to localhost
would report a failure in every deployment that does not use them. Missing
optional provider credentials are warnings and exit `0`; pass `--strict` to make
warnings fail.

Do not use the dependency scan as a container health check: an absent optional
dependency would restart a perfectly healthy container forever.

---

## 5. Error handling

Execution returns typed sentinels; branch with `errors.Is` rather than matching
message text.

| Sentinel | Meaning |
| --- | --- |
| `core.ErrGraphInvalid` | The graph failed validation. |
| `core.ErrRecursionLimit` | `MaxIterations` was exceeded (LangGraph's `GraphRecursionError`). |
| `core.ErrInterrupted` | Paused at an interrupt, or stopped by `Interrupt()`. |
| `core.ErrNodePanic` | A node or condition panicked; recovered and converted. |
| `core.ErrNoRoute` | No outgoing edge matched. |
| `core.ErrGraphClosed` | The graph was closed. |
| `llm.ErrProviderUnavailable` | Transient provider failure; retrying may help. |
| `llm.ErrRateLimited` | The provider asked you to slow down. |
| `llm.ErrProviderAuth` | Credentials were rejected. |
| `llm.ErrProviderRequest` | Permanent client error; retrying will not help. |

The original cause is always wrapped, so `errors.Is` against your own sentinel
works through the engine.

On failure `Execute` returns the last known good state alongside the error, so
partial progress is inspectable without a checkpointer.

---

## 6. Retries

Node retries are **off by default**. Node bodies commonly perform
non-idempotent work — model calls, tool side effects, writes — so retrying them
silently can duplicate that work.

Enable them where the work is safe to repeat:

```go
node := graph.AddNode("fetch", "Fetch", fetchFn)
node.Retry = &core.RetryPolicy{
    MaxAttempts: 3,
    Delay:       time.Second,
    Backoff:     2,
    RetryIf:     func(err error) bool { return errors.Is(err, llm.ErrProviderUnavailable) },
}
```

Each attempt starts from the state as it was before the attempt, so a partially
mutated state from a failed try cannot leak into the retry.

Provider-level retries are separate and driven by `ProviderConfig.RetryCount` /
`RetryDelay`. They apply only to transient failures and honour a `Retry-After`
header.

---

## 7. Concurrency

- A `*core.Graph` is safe to execute concurrently; each run keeps its own state and history.
- `GetCurrentState()` and `GetExecutionHistory()` reflect the **most recent** run and are for observability. The authoritative result of a run is what `Execute` returns.
- An `*agent.Agent` rejects a second concurrent run, because its conversation and execution record are per-agent.
- `Graph.Stream()` is shared and lossy: results are dropped rather than stalling execution. For lossless streaming, pass a channel via `ExecuteOptions.Stream`, which receives only that run's steps and is closed when the run ends.

---

## 8. Observability

Every node execution produces an `ExecutionResult` carrying the node ID, step
index, success, duration, attempt count and the state after the node. Failures
are recorded too, with a serialisable `error` field.

Agent executions record `execution_path` (the nodes that ran) and
`state_changes` (a before/after snapshot per node), which is what a debugging
client such as GoLangGraph Studio renders.

---

## Behaviour changes to be aware of

If you are upgrading, these defaults and shapes changed:

### `AgentExecution` wire format

`agent.AgentExecution` carried no struct tags, so it alone on this API
serialised with Go's default PascalCase field names while every neighbouring
payload (agent configs, providers, graph topology) was snake_case. Worse, its
`Error` field is a Go `error`, which `encoding/json` renders as `{}` — a failed
execution reached the client with no reason in it at all.

Every field is now tagged. The wire format is snake_case, and the failure
reason travels as a string `error` field; the Go `error` stays on the struct
for in-process callers and is excluded from JSON. `pkg/server`'s
`TestFrontendAPIContract` pins both the tagged names and the absence of the
untagged ones.

GoLangGraph Studio consumes this shape and is updated in lockstep. Any other
client reading `ID`/`Input`/`Output`/`Success` must move to
`id`/`input`/`output`/`success`.

| Change | Before | Now |
| --- | --- | --- |
| Node retries | 3 attempts by default | Off by default; opt in per node |
| `AgentExecution.Error` | A Go `error`, serialised as `{}` | String `error` field carrying the reason |
| `GET /api/v1/agents` | List of ID strings | List of agent configurations |
| `GET /api/v1/providers` | List of name strings | List of provider descriptions, credentials omitted |
| `BaseState` JSON | Serialised as `{}` | Full `{"data":…,"metadata":…}` payload |
| Conditional edges | Recorded but never used during execution | Evaluated once per visit and routed |
| Container health check | Local dependency scan | Server endpoint probe |
| `AutoServer` auth | None available | Configurable via `Security`, off by default |
| `AutoServer` CORS | Hardcoded `*` | Configurable allowlist |
| `AutoServer` `MaxRequestSize` | Declared, never enforced | Enforced |
| `AutoServer.GenerateEndpoints` | Callable at any time | Refused once `Start` has run |
| Agent IDs | Empty when built from a config literal | Assigned automatically by `NewAgent` |
| `builder` provider fallback | Returned `"mock"`, which does not exist | Returns empty and warns |
| `AgentSwarm.Execute` result order | Map iteration order (random) | The order agents were supplied |
| `ServerConfig.LogLevel` | Declared, never read | Applied to the server logger |
| `AgentExecution` JSON | Untagged, so Go PascalCase (`ID`, `Input`, …) | Tagged snake_case (`id`, `input`, …) like the rest of the API |
| Agent turn after the first | Treated as an interrupt resume; the new input was dropped | A new turn; resume is set only by `SeedResumeState`/`SeedConversation` |
| `multi-agent init` | Panicked writing project YAML (`cannot marshal type: llm.EarlyExitFunc`) | Writes the project; func-typed config fields are `yaml:"-"` |

See `test/conformance/DEVIATIONS.md` for where GoLangGraph intentionally differs
from LangGraph, and why.
