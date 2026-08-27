// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

package conformance

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/UnicoLab/GoLangGraph/pkg/core"
	"github.com/UnicoLab/GoLangGraph/pkg/persistence"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// checkpointers under test; every backend must satisfy the same contract.
func checkpointerBackends(t *testing.T) map[string]persistence.Checkpointer {
	t.Helper()
	return map[string]persistence.Checkpointer{
		"memory": persistence.NewMemoryCheckpointer(),
		"file":   persistence.NewFileCheckpointer(filepath.Join(t.TempDir(), "checkpoints")),
	}
}

// LangGraph: a checkpointer persists state per thread, and a later read returns
// what was written. A backend that silently drops state is worse than no
// backend, so this is asserted for every implementation.
func TestConformance_CheckpointRoundTrip(t *testing.T) {
	for name, cp := range checkpointerBackends(t) {
		t.Run(name, func(t *testing.T) {
			ctx := context.Background()
			state := core.NewBaseState()
			state.Set("counter", 7)
			state.Set("messages", []interface{}{msg("1", "hello")})

			require.NoError(t, cp.Save(ctx, &persistence.Checkpoint{
				ID: "cp-1", ThreadID: "thread-a", State: state,
				NodeID: "n1", StepID: 0, CreatedAt: time.Now(),
				Metadata: map[string]interface{}{"source": "test"},
			}))

			got, err := cp.Load(ctx, "thread-a", "cp-1")
			require.NoError(t, err)
			require.NotNil(t, got.State)

			counter, ok := got.State.Get("counter")
			require.True(t, ok, "checkpoint lost its state entirely")
			assert.EqualValues(t, 7, counter)

			messages, ok := got.State.Get("messages")
			require.True(t, ok)
			assert.Len(t, messages, 1)
			assert.Equal(t, "n1", got.NodeID)
		})
	}
}

// Threads must be isolated: one thread's checkpoints are invisible to another.
func TestConformance_ThreadIsolation(t *testing.T) {
	for name, cp := range checkpointerBackends(t) {
		t.Run(name, func(t *testing.T) {
			ctx := context.Background()
			for _, thread := range []string{"t1", "t2"} {
				st := core.NewBaseState()
				st.Set("owner", thread)
				require.NoError(t, cp.Save(ctx, &persistence.Checkpoint{
					ID: "cp", ThreadID: thread, State: st, CreatedAt: time.Now(),
				}))
			}

			for _, thread := range []string{"t1", "t2"} {
				got, err := cp.Load(ctx, thread, "cp")
				require.NoError(t, err)
				owner, _ := got.State.Get("owner")
				assert.Equal(t, thread, owner)
			}

			// A checkpoint ID from another thread must not resolve.
			_, err := cp.Load(ctx, "t3", "cp")
			assert.Error(t, err, "unknown thread must not resolve")
		})
	}
}

// Listing must report every checkpoint of a thread and nothing else.
func TestConformance_CheckpointListing(t *testing.T) {
	for name, cp := range checkpointerBackends(t) {
		t.Run(name, func(t *testing.T) {
			ctx := context.Background()
			for i := 0; i < 3; i++ {
				st := core.NewBaseState()
				st.Set("step", i)
				require.NoError(t, cp.Save(ctx, &persistence.Checkpoint{
					ID: fmt.Sprintf("cp-%d", i), ThreadID: "t", State: st,
					StepID: i, CreatedAt: time.Now().Add(time.Duration(i) * time.Second),
				}))
			}

			metas, err := cp.List(ctx, "t")
			require.NoError(t, err)
			assert.Len(t, metas, 3)

			// An unknown thread lists empty rather than erroring.
			empty, err := cp.List(ctx, "missing")
			require.NoError(t, err)
			assert.Empty(t, empty)
		})
	}
}

// Deleting removes only the requested checkpoint.
func TestConformance_CheckpointDelete(t *testing.T) {
	for name, cp := range checkpointerBackends(t) {
		t.Run(name, func(t *testing.T) {
			ctx := context.Background()
			for _, id := range []string{"a", "b"} {
				st := core.NewBaseState()
				st.Set("id", id)
				require.NoError(t, cp.Save(ctx, &persistence.Checkpoint{
					ID: id, ThreadID: "t", State: st, CreatedAt: time.Now(),
				}))
			}

			require.NoError(t, cp.Delete(ctx, "t", "a"))
			_, err := cp.Load(ctx, "t", "a")
			assert.Error(t, err)

			survivor, err := cp.Load(ctx, "t", "b")
			require.NoError(t, err)
			assert.NotNil(t, survivor)
		})
	}
}

// Identifiers that would escape the storage directory must be rejected rather
// than writing outside it.
func TestConformance_CheckpointRejectsPathTraversal(t *testing.T) {
	dir := t.TempDir()
	cp := persistence.NewFileCheckpointer(filepath.Join(dir, "store"))
	ctx := context.Background()

	for _, bad := range []string{"../escape", "a/b", "..", "", "with space"} {
		st := core.NewBaseState()
		err := cp.Save(ctx, &persistence.Checkpoint{ID: "cp", ThreadID: bad, State: st, CreatedAt: time.Now()})
		assert.Error(t, err, "thread ID %q must be rejected", bad)

		err = cp.Save(ctx, &persistence.Checkpoint{ID: bad, ThreadID: "ok", State: st, CreatedAt: time.Now()})
		assert.Error(t, err, "checkpoint ID %q must be rejected", bad)
	}
}

// Corrupted checkpoint files must surface as errors, not panics or silent
// empty state.
func TestConformance_CorruptedCheckpointIsDetected(t *testing.T) {
	base := filepath.Join(t.TempDir(), "store")
	cp := persistence.NewFileCheckpointer(base)
	ctx := context.Background()

	st := core.NewBaseState()
	st.Set("v", 1)
	require.NoError(t, cp.Save(ctx, &persistence.Checkpoint{ID: "cp", ThreadID: "t", State: st, CreatedAt: time.Now()}))

	corrupt(t, filepath.Join(base, "t", "cp.json"), "{not json")

	_, err := cp.Load(ctx, "t", "cp")
	require.Error(t, err, "corrupted checkpoint must be reported")

	// Listing must skip the corrupt entry without failing the whole call.
	metas, err := cp.List(ctx, "t")
	require.NoError(t, err)
	assert.Empty(t, metas)
}

// Durable execution: a graph wired to a checkpointer records state after every
// node, so a crashed run can resume from the last checkpoint.
func TestConformance_DurableExecutionCheckpointsEveryStep(t *testing.T) {
	cp := persistence.NewMemoryCheckpointer()
	saver := persistence.NewCheckpointSaver(cp)

	g := core.NewGraph("durable").WithCheckpointer(saver, "thread-durable")
	g.AddNode("a", "A", setNode("a", 1))
	g.AddNode("b", "B", setNode("b", 2))
	g.AddNode("c", "C", setNode("c", 3))
	g.AddEdge("a", "b", nil)
	g.AddEdge("b", "c", nil)
	require.NoError(t, g.SetStartNode("a"))
	require.NoError(t, g.AddEndNode("c"))

	_, err := g.Execute(context.Background(), core.NewBaseState())
	require.NoError(t, err)

	metas, err := cp.List(context.Background(), "thread-durable")
	require.NoError(t, err)
	assert.Len(t, metas, 3, "one checkpoint per executed node")

	latest, err := persistence.Latest(context.Background(), cp, "thread-durable")
	require.NoError(t, err)
	require.NotNil(t, latest)
	assert.Equal(t, "c", latest.NodeID)
	c, ok := latest.State.Get("c")
	require.True(t, ok)
	assert.Equal(t, 3, c)
}

// A run that fails part way must leave the completed work checkpointed, so a
// restart can resume rather than start over.
func TestConformance_DurableExecutionSurvivesFailure(t *testing.T) {
	cp := persistence.NewMemoryCheckpointer()
	saver := persistence.NewCheckpointSaver(cp)
	fail := atomic.Bool{}
	fail.Store(true)

	build := func() *core.Graph {
		g := core.NewGraph("restart").WithCheckpointer(saver, "thread-restart")
		g.AddNode("load", "Load", setNode("load", true))
		g.AddNode("work", "Work", func(ctx context.Context, s *core.BaseState) (*core.BaseState, error) {
			if fail.Load() {
				return nil, errors.New("transient backend outage")
			}
			s.Set("work", true)
			appendVisit(s, "work")
			return s, nil
		})
		g.AddEdge("load", "work", nil)
		require.NoError(t, g.SetStartNode("load"))
		require.NoError(t, g.AddEndNode("work"))
		return g
	}

	_, err := build().Execute(context.Background(), core.NewBaseState())
	require.Error(t, err)

	// The completed first node is durable.
	latest, err := persistence.Latest(context.Background(), cp, "thread-restart")
	require.NoError(t, err)
	require.NotNil(t, latest)
	assert.Equal(t, "load", latest.NodeID)

	// Restart from the checkpoint with the fault cleared.
	fail.Store(false)
	resumed, err := build().ExecuteWithOptions(context.Background(), latest.State, &core.ExecuteOptions{
		ThreadID:  "thread-restart",
		StartNode: "work",
	})
	require.NoError(t, err)
	work, ok := resumed.Get("work")
	require.True(t, ok)
	assert.Equal(t, true, work)
	loaded, ok := resumed.Get("load")
	require.True(t, ok, "state from before the crash must be carried forward")
	assert.Equal(t, true, loaded)
}

// LangGraph: interrupt_before pauses before a node runs; the run resumes from
// that node with state intact.
func TestConformance_InterruptBeforeAndResume(t *testing.T) {
	g := core.NewGraph("interrupt-before")
	g.Config.InterruptBefore = []string{"approve"}
	g.AddNode("draft", "Draft", setNode("draft", true))
	g.AddNode("approve", "Approve", setNode("approve", true))
	g.AddNode("publish", "Publish", setNode("publish", true))
	g.AddEdge("draft", "approve", nil)
	g.AddEdge("approve", "publish", nil)
	require.NoError(t, g.SetStartNode("draft"))
	require.NoError(t, g.AddEndNode("publish"))

	paused, err := g.Execute(context.Background(), core.NewBaseState())
	require.Error(t, err)
	assert.True(t, errors.Is(err, core.ErrInterrupted))

	var ie *core.InterruptError
	require.True(t, errors.As(err, &ie))
	assert.Equal(t, "approve", ie.NodeID)
	assert.True(t, ie.Before)

	_, ran := paused.Get("approve")
	assert.False(t, ran, "the interrupted node must not have run")
	drafted, _ := paused.Get("draft")
	assert.Equal(t, true, drafted)

	// Resume runs the pending node and everything after it.
	g.Config.InterruptBefore = nil
	out, err := g.Resume(context.Background(), ie)
	require.NoError(t, err)
	assert.Equal(t, []string{"draft", "approve", "publish"}, visits(out),
		"the resumed run continues the same state, so earlier visits remain recorded")
	published, _ := out.Get("publish")
	assert.Equal(t, true, published)
}

// LangGraph: interrupt_after pauses once a node has run; resuming continues
// with the node that follows.
func TestConformance_InterruptAfterAndResume(t *testing.T) {
	g := core.NewGraph("interrupt-after")
	g.Config.InterruptAfter = []string{"draft"}
	g.AddNode("draft", "Draft", setNode("draft", true))
	g.AddNode("publish", "Publish", setNode("publish", true))
	g.AddEdge("draft", "publish", nil)
	require.NoError(t, g.SetStartNode("draft"))
	require.NoError(t, g.AddEndNode("publish"))

	paused, err := g.Execute(context.Background(), core.NewBaseState())
	require.Error(t, err)

	var ie *core.InterruptError
	require.True(t, errors.As(err, &ie))
	assert.Equal(t, "draft", ie.NodeID)
	assert.False(t, ie.Before)
	drafted, _ := paused.Get("draft")
	assert.Equal(t, true, drafted, "the node completed before the pause")
	_, published := paused.Get("publish")
	assert.False(t, published)

	g.Config.InterruptAfter = nil
	out, err := g.Resume(context.Background(), ie)
	require.NoError(t, err)
	assert.Equal(t, []string{"draft", "publish"}, visits(out),
		"resume continues after the interrupted node without re-running it")
}

// State edited during a pause must be what the resumed run observes; this is
// the human-in-the-loop contract.
func TestConformance_ResumeWithEditedState(t *testing.T) {
	g := core.NewGraph("hitl")
	g.Config.InterruptBefore = []string{"apply"}
	g.AddNode("propose", "Propose", setNode("amount", 100))
	g.AddNode("apply", "Apply", func(ctx context.Context, s *core.BaseState) (*core.BaseState, error) {
		v, _ := s.Get("amount")
		s.Set("applied", v)
		return s, nil
	})
	g.AddEdge("propose", "apply", nil)
	require.NoError(t, g.SetStartNode("propose"))
	require.NoError(t, g.AddEndNode("apply"))

	_, err := g.Execute(context.Background(), core.NewBaseState())
	var ie *core.InterruptError
	require.True(t, errors.As(err, &ie))

	// A reviewer lowers the amount before approving.
	ie.State.Set("amount", 25)

	g.Config.InterruptBefore = nil
	out, err := g.Resume(context.Background(), ie)
	require.NoError(t, err)
	applied, _ := out.Get("applied")
	assert.Equal(t, 25, applied, "the resumed run must use the edited state")
}

// Interrupt() must stop an in-flight run promptly and be safe to call at any
// time, including concurrently and after Close.
func TestConformance_InterruptInFlight(t *testing.T) {
	g := core.NewGraph("interrupt-live")
	g.Config.MaxIterations = 1_000_000
	entered := make(chan struct{})
	var once sync.Once
	g.AddNode("spin", "Spin", func(ctx context.Context, s *core.BaseState) (*core.BaseState, error) {
		once.Do(func() { close(entered) })
		time.Sleep(time.Millisecond)
		return s, nil
	})
	g.AddEdge("spin", "spin", nil)
	require.NoError(t, g.SetStartNode("spin"))

	done := make(chan error, 1)
	go func() {
		_, err := g.Execute(context.Background(), core.NewBaseState())
		done <- err
	}()

	<-entered
	g.Interrupt()

	select {
	case err := <-done:
		require.Error(t, err)
		assert.True(t, errors.Is(err, core.ErrInterrupted), "want ErrInterrupted, got %v", err)
	case <-time.After(10 * time.Second):
		t.Fatal("interrupt did not stop execution")
	}

	// Repeated interrupts and closes must never panic.
	g.Interrupt()
	g.Close()
	g.Close()
	g.Interrupt()
}

// Retries must re-run a failing node up to the configured limit and then fail
// with the underlying cause.
func TestConformance_RetryPolicy(t *testing.T) {
	t.Run("succeeds within budget", func(t *testing.T) {
		var attempts atomic.Int32
		g := core.NewGraph("retry-ok")
		node := g.AddNode("flaky", "Flaky", func(ctx context.Context, s *core.BaseState) (*core.BaseState, error) {
			if attempts.Add(1) < 3 {
				return nil, errors.New("temporary failure")
			}
			s.Set("ok", true)
			return s, nil
		})
		node.Retry = &core.RetryPolicy{MaxAttempts: 3, Delay: time.Millisecond}
		require.NoError(t, g.SetStartNode("flaky"))
		require.NoError(t, g.AddEndNode("flaky"))

		out, err := g.Execute(context.Background(), core.NewBaseState())
		require.NoError(t, err)
		assert.EqualValues(t, 3, attempts.Load())
		ok, _ := out.Get("ok")
		assert.Equal(t, true, ok)
	})

	t.Run("exhausts budget", func(t *testing.T) {
		var attempts atomic.Int32
		sentinel := errors.New("always down")
		g := core.NewGraph("retry-fail")
		node := g.AddNode("broken", "Broken", func(ctx context.Context, s *core.BaseState) (*core.BaseState, error) {
			attempts.Add(1)
			return nil, sentinel
		})
		node.Retry = &core.RetryPolicy{MaxAttempts: 2, Delay: time.Millisecond}
		require.NoError(t, g.SetStartNode("broken"))

		_, err := g.Execute(context.Background(), core.NewBaseState())
		require.Error(t, err)
		assert.ErrorIs(t, err, sentinel)
		assert.EqualValues(t, 3, attempts.Load(), "initial attempt plus MaxAttempts retries")
	})

	t.Run("respects RetryIf", func(t *testing.T) {
		var attempts atomic.Int32
		permanent := errors.New("invalid request")
		g := core.NewGraph("retry-if")
		node := g.AddNode("n", "N", func(ctx context.Context, s *core.BaseState) (*core.BaseState, error) {
			attempts.Add(1)
			return nil, permanent
		})
		node.Retry = &core.RetryPolicy{
			MaxAttempts: 5,
			Delay:       time.Millisecond,
			RetryIf:     func(err error) bool { return !errors.Is(err, permanent) },
		}
		require.NoError(t, g.SetStartNode("n"))

		_, err := g.Execute(context.Background(), core.NewBaseState())
		require.Error(t, err)
		assert.EqualValues(t, 1, attempts.Load(), "non-retryable errors must not be retried")
	})

	t.Run("retries do not compound partial state", func(t *testing.T) {
		var attempts atomic.Int32
		g := core.NewGraph("retry-clean")
		node := g.AddNode("append", "Append", func(ctx context.Context, s *core.BaseState) (*core.BaseState, error) {
			raw, _ := s.Get("items")
			list, _ := raw.([]interface{})
			s.Set("items", append(append([]interface{}{}, list...), "x"))
			if attempts.Add(1) < 3 {
				return nil, errors.New("fail after mutating")
			}
			return s, nil
		})
		node.Retry = &core.RetryPolicy{MaxAttempts: 5, Delay: time.Millisecond}
		require.NoError(t, g.SetStartNode("append"))
		require.NoError(t, g.AddEndNode("append"))

		out, err := g.Execute(context.Background(), core.NewBaseState())
		require.NoError(t, err)
		items, _ := out.Get("items")
		assert.Len(t, items, 1, "each attempt must start from the pre-attempt state")
	})
}

// Default configuration must not silently retry: node bodies commonly perform
// non-idempotent work. This is an intentional GoLangGraph contract.
func TestConformance_NoRetryByDefault(t *testing.T) {
	var attempts atomic.Int32
	g := core.NewGraph("default-retry")
	g.AddNode("n", "N", func(ctx context.Context, s *core.BaseState) (*core.BaseState, error) {
		attempts.Add(1)
		return nil, errors.New("boom")
	})
	require.NoError(t, g.SetStartNode("n"))

	_, err := g.Execute(context.Background(), core.NewBaseState())
	require.Error(t, err)
	assert.EqualValues(t, 1, attempts.Load(), "the default configuration must execute a node exactly once")
}
