// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

// Package conformance verifies GoLangGraph against the semantics of the
// reference LangGraph implementation.
//
// Each test states the LangGraph behaviour it mirrors. Where GoLangGraph
// intentionally differs, the test asserts the GoLangGraph contract and the
// difference is recorded in DEVIATIONS.md.
package conformance

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/UnicoLab/GoLangGraph/pkg/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// setNode returns a node that records its own visit and sets a key.
func setNode(key string, value core.StateValue) core.NodeFunc {
	return func(ctx context.Context, s *core.BaseState) (*core.BaseState, error) {
		s.Set(key, value)
		appendVisit(s, key)
		return s, nil
	}
}

func appendVisit(s *core.BaseState, node string) {
	existing, _ := s.Get("__visits")
	list, _ := existing.([]interface{})
	s.Set("__visits", append(append([]interface{}{}, list...), node))
}

func visits(s *core.BaseState) []string {
	raw, _ := s.Get("__visits")
	list, _ := raw.([]interface{})
	out := make([]string, 0, len(list))
	for _, v := range list {
		out = append(out, fmt.Sprint(v))
	}
	return out
}

// LangGraph: a linear graph runs each node once, in edge order, and the final
// state contains every node's writes.
func TestConformance_LinearGraphTransitions(t *testing.T) {
	g := core.NewGraph("linear")
	g.AddNode("a", "A", setNode("a", 1))
	g.AddNode("b", "B", setNode("b", 2))
	g.AddNode("c", "C", setNode("c", 3))
	g.AddEdge("a", "b", nil)
	g.AddEdge("b", "c", nil)
	require.NoError(t, g.SetStartNode("a"))
	require.NoError(t, g.AddEndNode("c"))

	out, err := g.Execute(context.Background(), core.NewBaseState())
	require.NoError(t, err)

	assert.Equal(t, []string{"a", "b", "c"}, visits(out))
	for k, want := range map[string]int{"a": 1, "b": 2, "c": 3} {
		got, ok := out.Get(k)
		require.True(t, ok, "key %s missing", k)
		assert.Equal(t, want, got)
	}
}

// LangGraph: add_conditional_edges evaluates a path function and maps its
// result through the path map to choose exactly one destination.
func TestConformance_ConditionalEdgeRouting(t *testing.T) {
	for _, tc := range []struct {
		route string
		want  string
	}{{"left", "l"}, {"right", "r"}} {
		t.Run(tc.route, func(t *testing.T) {
			g := core.NewGraph("cond")
			g.AddNode("start", "Start", func(ctx context.Context, s *core.BaseState) (*core.BaseState, error) {
				s.Set("route", tc.route)
				appendVisit(s, "start")
				return s, nil
			})
			g.AddNode("l", "Left", setNode("l", true))
			g.AddNode("r", "Right", setNode("r", true))
			require.NoError(t, g.SetStartNode("start"))
			require.NoError(t, g.AddEndNode("l"))
			require.NoError(t, g.AddEndNode("r"))

			require.NoError(t, g.AddConditionalEdges("start",
				func(ctx context.Context, s *core.BaseState) (string, error) {
					v, _ := s.Get("route")
					return fmt.Sprint(v), nil
				},
				map[string]string{"left": "l", "right": "r"}))

			out, err := g.Execute(context.Background(), core.NewBaseState())
			require.NoError(t, err)
			assert.Equal(t, []string{"start", tc.want}, visits(out))

			// The branch not taken must not have run.
			other := map[string]string{"l": "r", "r": "l"}[tc.want]
			_, ran := out.Get(other)
			assert.False(t, ran, "branch %s should not have executed", other)
		})
	}
}

// LangGraph: a path function may return END to finish the graph.
func TestConformance_ConditionalEdgeToEND(t *testing.T) {
	g := core.NewGraph("cond-end")
	g.AddNode("start", "Start", setNode("start", true))
	g.AddNode("never", "Never", setNode("never", true))
	require.NoError(t, g.SetStartNode("start"))
	require.NoError(t, g.AddConditionalEdges("start",
		func(ctx context.Context, s *core.BaseState) (string, error) { return "done", nil },
		map[string]string{"done": core.END, "more": "never"}))

	out, err := g.Execute(context.Background(), core.NewBaseState())
	require.NoError(t, err)
	assert.Equal(t, []string{"start"}, visits(out))
	_, ran := out.Get("never")
	assert.False(t, ran)
}

// LangGraph: a routing key absent from the path map is an error rather than a
// silent fallthrough.
func TestConformance_ConditionalEdgeUnknownKeyIsError(t *testing.T) {
	g := core.NewGraph("cond-bad")
	g.AddNode("start", "Start", setNode("start", true))
	g.AddNode("a", "A", setNode("a", true))
	require.NoError(t, g.SetStartNode("start"))
	require.NoError(t, g.AddConditionalEdges("start",
		func(ctx context.Context, s *core.BaseState) (string, error) { return "nope", nil },
		map[string]string{"yes": "a"}))

	_, err := g.Execute(context.Background(), core.NewBaseState())
	require.Error(t, err)
	assert.True(t, errors.Is(err, core.ErrNoRoute), "want ErrNoRoute, got %v", err)
}

// LangGraph: cycles are supported and terminate when routing exits the loop.
func TestConformance_CycleTerminatesOnRouting(t *testing.T) {
	g := core.NewGraph("cycle")
	g.AddNode("loop", "Loop", func(ctx context.Context, s *core.BaseState) (*core.BaseState, error) {
		n, _ := s.Get("n")
		count, _ := n.(int)
		s.Set("n", count+1)
		appendVisit(s, "loop")
		return s, nil
	})
	g.AddNode("done", "Done", setNode("done", true))
	require.NoError(t, g.SetStartNode("loop"))
	require.NoError(t, g.AddEndNode("done"))
	require.NoError(t, g.AddConditionalEdges("loop",
		func(ctx context.Context, s *core.BaseState) (string, error) {
			n, _ := s.Get("n")
			if count, _ := n.(int); count >= 3 {
				return "exit", nil
			}
			return "again", nil
		},
		map[string]string{"again": "loop", "exit": "done"}))

	out, err := g.Execute(context.Background(), core.NewBaseState())
	require.NoError(t, err)
	n, _ := out.Get("n")
	assert.Equal(t, 3, n)
	assert.Equal(t, []string{"loop", "loop", "loop", "done"}, visits(out))
}

// LangGraph: exceeding recursion_limit raises GraphRecursionError. GoLangGraph
// returns ErrRecursionLimit and preserves the partial state.
func TestConformance_RecursionLimit(t *testing.T) {
	g := core.NewGraph("runaway")
	g.Config.MaxIterations = 5
	g.AddNode("loop", "Loop", func(ctx context.Context, s *core.BaseState) (*core.BaseState, error) {
		n, _ := s.Get("n")
		count, _ := n.(int)
		s.Set("n", count+1)
		return s, nil
	})
	g.AddEdge("loop", "loop", nil)
	require.NoError(t, g.SetStartNode("loop"))

	out, err := g.Execute(context.Background(), core.NewBaseState())
	require.Error(t, err)
	assert.True(t, errors.Is(err, core.ErrRecursionLimit), "want ErrRecursionLimit, got %v", err)

	// Partial progress is preserved so the failure can be diagnosed.
	require.NotNil(t, out)
	n, _ := out.Get("n")
	assert.Equal(t, 5, n, "should have executed exactly MaxIterations nodes")
}

// LangGraph: returning None from a node leaves state unchanged.
func TestConformance_NilUpdateMeansNoChange(t *testing.T) {
	g := core.NewGraph("nil-update")
	g.AddNode("a", "A", setNode("a", 1))
	g.AddNode("noop", "Noop", func(ctx context.Context, s *core.BaseState) (*core.BaseState, error) {
		return nil, nil
	})
	g.AddNode("b", "B", setNode("b", 2))
	g.AddEdge("a", "noop", nil)
	g.AddEdge("noop", "b", nil)
	require.NoError(t, g.SetStartNode("a"))
	require.NoError(t, g.AddEndNode("b"))

	out, err := g.Execute(context.Background(), core.NewBaseState())
	require.NoError(t, err)
	a, ok := out.Get("a")
	require.True(t, ok, "state written before the no-op node must survive it")
	assert.Equal(t, 1, a)
	b, _ := out.Get("b")
	assert.Equal(t, 2, b)
}

// A node panic must be contained: converted to an error, never crashing the
// process and never leaving the graph wedged.
func TestConformance_NodePanicIsContained(t *testing.T) {
	g := core.NewGraph("panic")
	g.AddNode("boom", "Boom", func(ctx context.Context, s *core.BaseState) (*core.BaseState, error) {
		panic("node exploded")
	})
	require.NoError(t, g.SetStartNode("boom"))

	_, err := g.Execute(context.Background(), core.NewBaseState())
	require.Error(t, err)
	assert.True(t, errors.Is(err, core.ErrNodePanic), "want ErrNodePanic, got %v", err)

	// The graph must remain usable afterwards.
	done := make(chan struct{})
	go func() {
		defer close(done)
		_ = g.Validate()
		_ = g.GetTopology()
		g.Reset()
	}()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("graph deadlocked after a node panic")
	}
}

// Failures must be observable: the failing step is recorded in history with the
// node identity and a serialisable message.
func TestConformance_FailureIsRecordedInHistory(t *testing.T) {
	sentinel := errors.New("upstream unavailable")
	g := core.NewGraph("fail")
	g.AddNode("ok", "OK", setNode("ok", true))
	g.AddNode("bad", "Bad", func(ctx context.Context, s *core.BaseState) (*core.BaseState, error) {
		return nil, sentinel
	})
	g.AddEdge("ok", "bad", nil)
	require.NoError(t, g.SetStartNode("ok"))

	_, err := g.Execute(context.Background(), core.NewBaseState())
	require.Error(t, err)
	assert.ErrorIs(t, err, sentinel, "the original cause must be preserved")
	assert.Contains(t, err.Error(), "bad", "the failing node must be identified")

	history := g.GetExecutionHistory()
	require.Len(t, history, 2)
	assert.True(t, history[0].Success)
	assert.False(t, history[1].Success)
	assert.Equal(t, "bad", history[1].NodeID)
	assert.NotEmpty(t, history[1].ErrorMessage, "error must be serialisable for clients")
}

// Cancellation must propagate the underlying context cause so callers can
// distinguish a timeout from a cancellation.
func TestConformance_ContextCancellationPropagates(t *testing.T) {
	t.Run("canceled", func(t *testing.T) {
		g := core.NewGraph("cancel")
		started := make(chan struct{})
		g.AddNode("slow", "Slow", func(ctx context.Context, s *core.BaseState) (*core.BaseState, error) {
			close(started)
			<-ctx.Done()
			return nil, ctx.Err()
		})
		require.NoError(t, g.SetStartNode("slow"))

		ctx, cancel := context.WithCancel(context.Background())
		go func() { <-started; cancel() }()
		_, err := g.Execute(ctx, core.NewBaseState())
		require.Error(t, err)
		assert.ErrorIs(t, err, context.Canceled)
	})

	t.Run("deadline", func(t *testing.T) {
		g := core.NewGraph("timeout")
		g.Config.Timeout = 30 * time.Millisecond
		g.AddNode("slow", "Slow", func(ctx context.Context, s *core.BaseState) (*core.BaseState, error) {
			<-ctx.Done()
			return nil, ctx.Err()
		})
		require.NoError(t, g.SetStartNode("slow"))

		_, err := g.Execute(context.Background(), core.NewBaseState())
		require.Error(t, err)
		assert.ErrorIs(t, err, context.DeadlineExceeded)
	})
}

// A compiled graph must be safe to invoke concurrently, as LangGraph's
// CompiledGraph is. Each run keeps its own state.
func TestConformance_ConcurrentInvocationIsolation(t *testing.T) {
	g := core.NewGraph("concurrent")
	g.Config.EnableStreaming = false
	g.AddNode("double", "Double", func(ctx context.Context, s *core.BaseState) (*core.BaseState, error) {
		v, _ := s.Get("in")
		n, _ := v.(int)
		time.Sleep(time.Millisecond)
		s.Set("out", n*2)
		return s, nil
	})
	require.NoError(t, g.SetStartNode("double"))
	require.NoError(t, g.AddEndNode("double"))

	const workers = 32
	var wg sync.WaitGroup
	errs := make([]error, workers)
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			in := core.NewBaseState()
			in.Set("in", i)
			out, err := g.Execute(context.Background(), in)
			if err != nil {
				errs[i] = err
				return
			}
			got, _ := out.Get("out")
			if got != i*2 {
				errs[i] = fmt.Errorf("run %d observed %v, want %d", i, got, i*2)
			}
		}(i)
	}
	wg.Wait()
	require.NoError(t, errors.Join(errs...))
}

// Graph construction errors must surface rather than silently corrupting the
// graph: duplicate node IDs are rejected, as in LangGraph.
func TestConformance_DuplicateNodeIsRejected(t *testing.T) {
	g := core.NewGraph("dup")
	g.AddNode("a", "A", setNode("a", 1))
	g.AddNode("a", "A again", setNode("a", 2))
	require.NoError(t, g.SetStartNode("a"))

	err := g.Validate()
	require.Error(t, err)
	assert.True(t, errors.Is(err, core.ErrGraphInvalid))
	assert.Contains(t, err.Error(), "already exists")
}

// Routing must be deterministic when several conditional edges could match.
func TestConformance_RoutingIsDeterministic(t *testing.T) {
	build := func() *core.Graph {
		g := core.NewGraph("determinism")
		g.Config.EnableStreaming = false
		g.AddNode("start", "Start", setNode("start", true))
		g.AddNode("x", "X", setNode("x", true))
		g.AddNode("y", "Y", setNode("y", true))
		require.NoError(t, g.SetStartNode("start"))
		require.NoError(t, g.AddEndNode("x"))
		require.NoError(t, g.AddEndNode("y"))
		// Both conditions match; insertion order must decide.
		g.AddEdge("start", "x", func(ctx context.Context, s *core.BaseState) (string, error) { return "x", nil })
		g.AddEdge("start", "y", func(ctx context.Context, s *core.BaseState) (string, error) { return "y", nil })
		return g
	}

	for i := 0; i < 50; i++ {
		out, err := build().Execute(context.Background(), core.NewBaseState())
		require.NoError(t, err)
		require.Equal(t, []string{"start", "x"}, visits(out), "routing must not depend on map iteration order")
	}
}
