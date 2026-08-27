// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package conformance

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/UnicoLab/GoLangGraph/pkg/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// LangGraph: streaming yields one update per executed node, in execution order,
// carrying the state after that node.
func TestConformance_StreamingEmitsEveryStepInOrder(t *testing.T) {
	g := core.NewGraph("stream")
	g.AddNode("a", "A", setNode("a", 1))
	g.AddNode("b", "B", setNode("b", 2))
	g.AddNode("c", "C", setNode("c", 3))
	g.AddEdge("a", "b", nil)
	g.AddEdge("b", "c", nil)
	require.NoError(t, g.SetStartNode("a"))
	require.NoError(t, g.AddEndNode("c"))

	stream := make(chan *core.ExecutionResult, 16)
	_, err := g.ExecuteWithOptions(context.Background(), core.NewBaseState(), &core.ExecuteOptions{Stream: stream})
	require.NoError(t, err)

	var seen []string
	for result := range stream {
		seen = append(seen, result.NodeID)
		require.True(t, result.Success)
		require.NotNil(t, result.State, "each streamed step carries the state after that node")
		v, ok := result.State.Get(result.NodeID)
		require.True(t, ok, "step for node %s must include its own write", result.NodeID)
		assert.NotNil(t, v)
	}
	assert.Equal(t, []string{"a", "b", "c"}, seen)
}

// The per-run stream must close when the run ends, including on failure, so
// consumers are never left waiting.
func TestConformance_StreamClosesOnFailure(t *testing.T) {
	g := core.NewGraph("stream-fail")
	g.AddNode("a", "A", setNode("a", 1))
	g.AddNode("bad", "Bad", func(ctx context.Context, s *core.BaseState) (*core.BaseState, error) {
		return nil, errors.New("nope")
	})
	g.AddEdge("a", "bad", nil)
	require.NoError(t, g.SetStartNode("a"))

	stream := make(chan *core.ExecutionResult, 16)
	_, err := g.ExecuteWithOptions(context.Background(), core.NewBaseState(), &core.ExecuteOptions{Stream: stream})
	require.Error(t, err)

	done := make(chan []*core.ExecutionResult, 1)
	go func() {
		var got []*core.ExecutionResult
		for r := range stream {
			got = append(got, r)
		}
		done <- got
	}()

	select {
	case got := <-done:
		require.Len(t, got, 2)
		assert.True(t, got[0].Success)
		assert.False(t, got[1].Success, "the failing step must be streamed too")
		assert.NotEmpty(t, got[1].ErrorMessage)
	case <-time.After(5 * time.Second):
		t.Fatal("stream was not closed after a failed run")
	}
}

// Streamed results must be JSON-serialisable, since they cross a WebSocket to
// GoLangGraph Studio. An error that vanishes on the wire is a silent failure.
func TestConformance_StreamResultIsSerializable(t *testing.T) {
	g := core.NewGraph("stream-json")
	g.AddNode("bad", "Bad", func(ctx context.Context, s *core.BaseState) (*core.BaseState, error) {
		return nil, errors.New("provider timeout")
	})
	require.NoError(t, g.SetStartNode("bad"))

	stream := make(chan *core.ExecutionResult, 4)
	_, err := g.ExecuteWithOptions(context.Background(), core.NewBaseState(), &core.ExecuteOptions{Stream: stream})
	require.Error(t, err)

	result := <-stream
	require.NotNil(t, result)
	encoded, err := jsonMarshal(result)
	require.NoError(t, err)
	assert.Contains(t, string(encoded), "provider timeout",
		"the failure reason must survive serialisation to clients")
	assert.NotContains(t, string(encoded), "goroutine ", "stack traces must not be exposed to clients")
}

// A slow consumer must not stall graph execution.
func TestConformance_SlowStreamConsumerDoesNotBlockExecution(t *testing.T) {
	g := core.NewGraph("stream-slow")
	for i := 0; i < 50; i++ {
		id := fmt.Sprintf("n%d", i)
		g.AddNode(id, id, setNode(id, i))
		if i > 0 {
			g.AddEdge(fmt.Sprintf("n%d", i-1), id, nil)
		}
	}
	require.NoError(t, g.SetStartNode("n0"))
	require.NoError(t, g.AddEndNode("n49"))

	// A deliberately tiny, unread buffer.
	stream := make(chan *core.ExecutionResult, 1)

	done := make(chan error, 1)
	go func() {
		_, err := g.ExecuteWithOptions(context.Background(), core.NewBaseState(), &core.ExecuteOptions{Stream: stream})
		done <- err
	}()

	select {
	case err := <-done:
		require.NoError(t, err)
	case <-time.After(15 * time.Second):
		t.Fatal("execution blocked on a slow stream consumer")
	}
}

// LangGraph: a compiled graph can be used as a node of another graph.
func TestConformance_SubgraphAsNode(t *testing.T) {
	sub := core.NewGraph("inner")
	sub.AddNode("double", "Double", func(ctx context.Context, s *core.BaseState) (*core.BaseState, error) {
		v, _ := s.Get("value")
		n, _ := v.(int)
		s.Set("value", n*2)
		s.Set("inner_ran", true)
		return s, nil
	})
	require.NoError(t, sub.SetStartNode("double"))
	require.NoError(t, sub.AddEndNode("double"))

	parent := core.NewGraph("outer")
	parent.AddNode("seed", "Seed", setNode("value", 21))
	_, err := parent.AddSubgraph("inner", "Inner", sub, nil)
	require.NoError(t, err)
	parent.AddNode("report", "Report", setNode("done", true))
	parent.AddEdge("seed", "inner", nil)
	parent.AddEdge("inner", "report", nil)
	require.NoError(t, parent.SetStartNode("seed"))
	require.NoError(t, parent.AddEndNode("report"))

	out, err := parent.Execute(context.Background(), core.NewBaseState())
	require.NoError(t, err)

	value, _ := out.Get("value")
	assert.Equal(t, 42, value, "the subgraph result must flow into the parent state")
	ran, _ := out.Get("inner_ran")
	assert.Equal(t, true, ran)
	done, _ := out.Get("done")
	assert.Equal(t, true, done)
}

// Input and output projection lets a subgraph expose a narrow interface.
func TestConformance_SubgraphKeyProjection(t *testing.T) {
	sub := core.NewGraph("inner")
	sub.AddNode("work", "Work", func(ctx context.Context, s *core.BaseState) (*core.BaseState, error) {
		// The parent's secret must not be visible here.
		if _, leaked := s.Get("secret"); leaked {
			return nil, errors.New("subgraph saw a key outside its declared input")
		}
		v, _ := s.Get("in")
		s.Set("out", fmt.Sprintf("processed:%v", v))
		s.Set("scratch", "internal")
		return s, nil
	})
	require.NoError(t, sub.SetStartNode("work"))
	require.NoError(t, sub.AddEndNode("work"))

	parent := core.NewGraph("outer")
	parent.AddNode("seed", "Seed", func(ctx context.Context, s *core.BaseState) (*core.BaseState, error) {
		s.Set("in", "payload")
		s.Set("secret", "do-not-leak")
		return s, nil
	})
	_, err := parent.AddSubgraph("inner", "Inner", sub, &core.SubgraphOptions{
		InputKeys:  []string{"in"},
		OutputKeys: []string{"out"},
	})
	require.NoError(t, err)
	parent.AddEdge("seed", "inner", nil)
	require.NoError(t, parent.SetStartNode("seed"))
	require.NoError(t, parent.AddEndNode("inner"))

	out, err := parent.Execute(context.Background(), core.NewBaseState())
	require.NoError(t, err)

	result, ok := out.Get("out")
	require.True(t, ok)
	assert.Equal(t, "processed:payload", result)
	_, leaked := out.Get("scratch")
	assert.False(t, leaked, "keys outside OutputKeys must not reach the parent")
	secret, _ := out.Get("secret")
	assert.Equal(t, "do-not-leak", secret, "the parent's own keys survive")
}

// Namespacing keeps a subgraph's output from colliding with parent keys.
func TestConformance_SubgraphNamespace(t *testing.T) {
	sub := core.NewGraph("inner")
	sub.AddNode("work", "Work", setNode("result", "inner-value"))
	require.NoError(t, sub.SetStartNode("work"))
	require.NoError(t, sub.AddEndNode("work"))

	parent := core.NewGraph("outer")
	parent.AddNode("seed", "Seed", setNode("result", "parent-value"))
	_, err := parent.AddSubgraph("inner", "Inner", sub, &core.SubgraphOptions{Namespace: "inner"})
	require.NoError(t, err)
	parent.AddEdge("seed", "inner", nil)
	require.NoError(t, parent.SetStartNode("seed"))
	require.NoError(t, parent.AddEndNode("inner"))

	out, err := parent.Execute(context.Background(), core.NewBaseState())
	require.NoError(t, err)

	parentValue, _ := out.Get("result")
	assert.Equal(t, "parent-value", parentValue, "namespaced output must not clobber parent keys")
	nested, ok := out.Get("inner")
	require.True(t, ok)
	assert.Equal(t, "inner-value", nested.(map[string]core.StateValue)["result"])
}

// Subgraph reducers apply when merging back into the parent.
func TestConformance_SubgraphMergeUsesReducers(t *testing.T) {
	schema := core.NewStateSchema().
		AddChannel("log", core.Append, func() core.StateValue { return []interface{}{} })

	sub := core.NewGraph("inner").WithStateSchema(schema)
	sub.AddUpdateNode("work", "Work", func(ctx context.Context, s *core.BaseState) (map[string]core.StateValue, error) {
		return map[string]core.StateValue{"log": []interface{}{"inner"}}, nil
	})
	require.NoError(t, sub.SetStartNode("work"))
	require.NoError(t, sub.AddEndNode("work"))

	parent := core.NewGraph("outer").WithStateSchema(schema)
	parent.AddUpdateNode("seed", "Seed", func(ctx context.Context, s *core.BaseState) (map[string]core.StateValue, error) {
		return map[string]core.StateValue{"log": []interface{}{"parent"}}, nil
	})
	_, err := parent.AddSubgraph("inner", "Inner", sub, &core.SubgraphOptions{OutputKeys: []string{"log"}, Schema: schema})
	require.NoError(t, err)
	parent.AddEdge("seed", "inner", nil)
	require.NoError(t, parent.SetStartNode("seed"))
	require.NoError(t, parent.AddEndNode("inner"))

	out, err := parent.Execute(context.Background(), schema.NewState())
	require.NoError(t, err)

	log, _ := out.Get("log")
	// The subgraph inherits the parent log, appends to it, and the merge reduces
	// the returned slice onto the parent's own copy.
	assert.Contains(t, fmt.Sprint(log), "parent")
	assert.Contains(t, fmt.Sprint(log), "inner")
}

// A subgraph failure must identify the subgraph and preserve the cause.
func TestConformance_SubgraphFailurePropagates(t *testing.T) {
	sentinel := errors.New("inner exploded")
	sub := core.NewGraph("inner")
	sub.AddNode("boom", "Boom", func(ctx context.Context, s *core.BaseState) (*core.BaseState, error) {
		return nil, sentinel
	})
	require.NoError(t, sub.SetStartNode("boom"))

	parent := core.NewGraph("outer")
	_, err := parent.AddSubgraph("inner", "Inner", sub, nil)
	require.NoError(t, err)
	require.NoError(t, parent.SetStartNode("inner"))

	_, err = parent.Execute(context.Background(), core.NewBaseState())
	require.Error(t, err)
	assert.ErrorIs(t, err, sentinel)
	assert.Contains(t, err.Error(), "inner")
}

// Recursive graph composition must be rejected at build time rather than
// exhausting the stack at run time.
func TestConformance_SubgraphCycleRejected(t *testing.T) {
	a := core.NewGraph("a")
	a.AddNode("n", "N", setNode("n", 1))
	require.NoError(t, a.SetStartNode("n"))
	require.NoError(t, a.AddEndNode("n"))

	b := core.NewGraph("b")
	b.AddNode("m", "M", setNode("m", 1))
	require.NoError(t, b.SetStartNode("m"))
	require.NoError(t, b.AddEndNode("m"))

	_, err := a.AddSubgraph("b", "B", b, nil)
	require.NoError(t, err)

	// b containing a would close the loop.
	_, err = b.AddSubgraph("a", "A", a, nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "cycle")

	// Direct self-nesting is rejected too.
	_, err = a.AddSubgraph("self", "Self", a, nil)
	require.Error(t, err)
}

// A subgraph must respect its own recursion limit rather than the parent's.
func TestConformance_SubgraphHasOwnRecursionLimit(t *testing.T) {
	sub := core.NewGraph("inner")
	sub.Config.MaxIterations = 3
	sub.AddNode("loop", "Loop", func(ctx context.Context, s *core.BaseState) (*core.BaseState, error) {
		return s, nil
	})
	sub.AddEdge("loop", "loop", nil)
	require.NoError(t, sub.SetStartNode("loop"))

	parent := core.NewGraph("outer")
	parent.Config.MaxIterations = 100
	_, err := parent.AddSubgraph("inner", "Inner", sub, nil)
	require.NoError(t, err)
	require.NoError(t, parent.SetStartNode("inner"))

	_, err = parent.Execute(context.Background(), core.NewBaseState())
	require.Error(t, err)
	assert.True(t, errors.Is(err, core.ErrRecursionLimit))
}

// Topology must include conditional routes and subgraph nodes so visualisers
// and Studio render the reachable graph, not a subset of it.
func TestConformance_TopologyIncludesConditionalRoutes(t *testing.T) {
	g := core.NewGraph("topology")
	g.AddNode("start", "Start", setNode("start", true))
	g.AddNode("a", "A", setNode("a", true))
	g.AddNode("b", "B", setNode("b", true))
	require.NoError(t, g.SetStartNode("start"))
	require.NoError(t, g.AddConditionalEdges("start",
		func(ctx context.Context, s *core.BaseState) (string, error) { return "x", nil },
		map[string]string{"x": "a", "y": "b"}))

	topo := g.GetTopology()
	assert.ElementsMatch(t, []string{"a", "b"}, topo["start"],
		"conditional destinations must appear in the topology")
}
