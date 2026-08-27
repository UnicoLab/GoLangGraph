# GoLangGraph vs. LangGraph: conformance and intentional deviations

This directory holds the conformance suite that checks GoLangGraph against the
semantics of the reference Python LangGraph implementation. Every test names the
LangGraph behaviour it mirrors.

Go is not Python, and some LangGraph behaviour depends on language features Go
does not have. Where GoLangGraph deliberately differs, the difference is listed
below and the GoLangGraph contract is covered by a test.

## Conformant behaviour

These behaviours match LangGraph and are asserted by the suite:

| Area | LangGraph behaviour | Test |
| --- | --- | --- |
| Linear execution | Nodes run once each, in edge order | `TestConformance_LinearGraphTransitions` |
| Conditional edges | One path function per node, result mapped through the path map | `TestConformance_ConditionalEdgeRouting` |
| Routing to END | A path function may return `END` to finish | `TestConformance_ConditionalEdgeToEND` |
| Unmapped route key | Raises rather than falling through | `TestConformance_ConditionalEdgeUnknownKeyIsError` |
| Cycles | Supported; terminate when routing exits | `TestConformance_CycleTerminatesOnRouting` |
| Recursion limit | Exceeding the limit is an error | `TestConformance_RecursionLimit` |
| No-op node | Returning nothing leaves state unchanged | `TestConformance_NilUpdateMeansNoChange` |
| `operator.add` channels | List updates accumulate | `TestConformance_AppendReducer` |
| `add_messages` | Appends, and replaces messages sharing an `id` | `TestConformance_AddMessagesReducer` |
| Default channels | Last-write-wins | `TestConformance_DefaultChannelIsLastWriteWins` |
| Parallel branches | Each branch sees the same input; updates combine via reducers | `TestConformance_ParallelBranchesMergeViaReducers` |
| Streaming | One update per executed node, in order | `TestConformance_StreamingEmitsEveryStepInOrder` |
| Checkpointing | State persists per thread and reloads intact | `TestConformance_CheckpointRoundTrip` |
| Thread isolation | Threads do not see each other's checkpoints | `TestConformance_ThreadIsolation` |
| Durable execution | State is checkpointed after every node | `TestConformance_DurableExecutionCheckpointsEveryStep` |
| `interrupt_before` | Pauses before a node; resume runs it | `TestConformance_InterruptBeforeAndResume` |
| `interrupt_after` | Pauses after a node; resume continues past it | `TestConformance_InterruptAfterAndResume` |
| Human-in-the-loop edits | State edited during a pause is what resumes | `TestConformance_ResumeWithEditedState` |
| Retries | Retry up to a budget, then fail with the cause | `TestConformance_RetryPolicy` |
| Subgraphs | A compiled graph can be a node | `TestConformance_SubgraphAsNode` |
| Concurrent invocation | A compiled graph is safe to invoke concurrently | `TestConformance_ConcurrentInvocationIsolation` |

## Intentional deviations

### 1. Nodes return a whole state by default, not a partial update

**LangGraph:** a node returns a dict of only the channels it changed; the
framework merges it through the channel reducers.

**GoLangGraph:** the default `NodeFunc` receives a copy of the state and returns
the state to carry forward. Reducer semantics are available by registering a
node with `AddUpdateNode`, whose `UpdateFunc` returns only changed channels and
is merged through the graph's `StateSchema`.

**Why:** Go has no `TypedDict`, and the whole-state form is the existing
GoLangGraph API. Supporting both keeps existing code working while making true
reducer semantics available where they matter (fan-in, message accumulation).

**Tested by:** `TestConformance_AppendReducer`, `TestConformance_ParallelBranchesMergeViaReducers`.

### 2. Parallel fan-out is explicit

**LangGraph:** listing several edges out of one node makes them run in parallel
in a single super-step, and the framework merges the branches.

**GoLangGraph:** `Execute` follows exactly one edge per step. Parallel
super-steps are requested explicitly with `ExecuteParallelUpdates`, which runs
the named nodes concurrently and merges their updates through the schema.

**Why:** implicit fan-out changes the meaning of an existing graph built with
`AddEdge`, where multiple outgoing edges already meant "pick one". Making
parallelism explicit avoids silently changing the behaviour of existing graphs.

**Tested by:** `TestConformance_ParallelBranchesMergeViaReducers`,
`TestConformance_ParallelMergeIsOrderIndependent`, `TestConformance_RoutingIsDeterministic`.

### 3. Branch merge order is the declared order, not completion order

**LangGraph:** merge order for concurrently-updated channels is not part of the
public contract.

**GoLangGraph:** branch updates are applied in the order the node IDs were
supplied, regardless of which branch finishes first, so a run is reproducible.

**Tested by:** `TestConformance_ParallelMergeIsOrderIndependent`.

### 4. Retries are off by default

**LangGraph:** nodes have no retry policy unless one is attached.

**GoLangGraph:** same — but note this is a change from earlier GoLangGraph
versions, which retried every node three times by default. Node bodies commonly
perform non-idempotent work (LLM calls, tool side effects, writes), so silent
retries could duplicate them. Retries are opt-in per node via `Node.Retry`, or
globally via `GraphConfig.RetryAttempts`.

**Tested by:** `TestConformance_NoRetryByDefault`, `TestConformance_RetryPolicy`.

### 5. Failures return the last known good state alongside the error

**LangGraph:** raises, and the caller reads state back from the checkpointer.

**GoLangGraph:** `Execute` returns `(state, err)` with the state as of the last
successful node, so partial progress is inspectable without a checkpointer.

**Tested by:** `TestConformance_RecursionLimit`, `TestConformance_ParallelPartialFailure`.

### 6. Errors are typed sentinels, not exception classes

`errors.Is` against `ErrRecursionLimit`, `ErrInterrupted`, `ErrNodePanic`,
`ErrNoRoute`, `ErrGraphInvalid` and `ErrGraphClosed` replaces catching
`GraphRecursionError` and friends. The original cause is always wrapped, so
`errors.Is` against a caller's own sentinel works through the engine.

**Tested by:** `TestConformance_FailureIsRecordedInHistory`, `TestConformance_ContextCancellationPropagates`.

### 7. Panics in user code become errors

Go has no exception hierarchy; a panicking node would otherwise take down the
process. The engine recovers panics in node functions and edge conditions and
converts them to a `*PanicError` that matches `errors.Is(err, ErrNodePanic)`.
The goroutine stack is attached to the error value and logged, but deliberately
kept out of the error message so it is not returned to API clients.

**Tested by:** `TestConformance_NodePanicIsContained`, `TestConformance_StreamResultIsSerialisable`.

### 8. The graph-wide stream is lossy; per-run streams are not

`Graph.Stream()` is a shared, buffered channel: if no one reads it, results are
dropped rather than stalling execution. For lossless streaming, pass a channel
via `ExecuteOptions.Stream`, which receives only that run's steps and is closed
when the run ends.

**Tested by:** `TestConformance_SlowStreamConsumerDoesNotBlockExecution`,
`TestConformance_StreamClosesOnFailure`.

### 9. Subgraph state exchange is explicit

**LangGraph:** a subgraph shares the parent's state keys by default.

**GoLangGraph:** `AddSubgraph` defaults to the same behaviour, and additionally
supports `InputKeys` / `OutputKeys` projection and `Namespace` isolation, so a
nested graph can expose a narrow interface instead of the whole state.
Composition cycles are rejected at build time.

**Tested by:** `TestConformance_SubgraphKeyProjection`, `TestConformance_SubgraphNamespace`,
`TestConformance_SubgraphCycleRejected`.

## Running the suite

```bash
go test -race ./test/conformance/...
```

The suite is part of the default `go test ./...` run and executes with the race
detector in CI.
