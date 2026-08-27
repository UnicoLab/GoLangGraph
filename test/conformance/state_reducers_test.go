// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package conformance

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/UnicoLab/GoLangGraph/pkg/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func msg(id, text string) map[string]interface{} {
	return map[string]interface{}{"id": id, "content": text}
}

// LangGraph: a channel annotated with operator.add concatenates list updates
// instead of overwriting them.
func TestConformance_AppendReducer(t *testing.T) {
	schema := core.NewStateSchema().
		AddChannel("items", core.Append, func() core.StateValue { return []interface{}{} })

	g := core.NewGraph("append").WithStateSchema(schema)
	g.AddUpdateNode("one", "One", func(ctx context.Context, s *core.BaseState) (map[string]core.StateValue, error) {
		return map[string]core.StateValue{"items": []interface{}{"a"}}, nil
	})
	g.AddUpdateNode("two", "Two", func(ctx context.Context, s *core.BaseState) (map[string]core.StateValue, error) {
		return map[string]core.StateValue{"items": []interface{}{"b", "c"}}, nil
	})
	g.AddEdge("one", "two", nil)
	require.NoError(t, g.SetStartNode("one"))
	require.NoError(t, g.AddEndNode("two"))

	out, err := g.Execute(context.Background(), schema.NewState())
	require.NoError(t, err)

	items, _ := out.Get("items")
	assert.Equal(t, []interface{}{"a", "b", "c"}, items,
		"operator.add semantics: updates accumulate rather than overwrite")
}

// LangGraph: add_messages appends new messages and replaces existing ones that
// share an id.
func TestConformance_AddMessagesReducer(t *testing.T) {
	schema := core.NewStateSchema().
		AddChannel("messages", core.AddMessages, func() core.StateValue { return []interface{}{} })

	state := schema.NewState()
	schema.ApplyUpdates(state, map[string]core.StateValue{
		"messages": []interface{}{msg("1", "hello"), msg("2", "world")},
	})
	schema.ApplyUpdates(state, map[string]core.StateValue{
		"messages": []interface{}{msg("2", "WORLD"), msg("3", "again")},
	})

	raw, _ := state.Get("messages")
	list, ok := raw.([]interface{})
	require.True(t, ok)
	require.Len(t, list, 3, "matching ids replace in place rather than appending")

	assert.Equal(t, "hello", list[0].(map[string]interface{})["content"])
	assert.Equal(t, "WORLD", list[1].(map[string]interface{})["content"], "message 2 must be replaced")
	assert.Equal(t, "again", list[2].(map[string]interface{})["content"])
}

// Messages without ids always append, matching add_messages.
func TestConformance_AddMessagesWithoutIDsAppends(t *testing.T) {
	schema := core.NewStateSchema().
		AddChannel("messages", core.AddMessages, func() core.StateValue { return []interface{}{} })

	state := schema.NewState()
	for i := 0; i < 3; i++ {
		schema.ApplyUpdates(state, map[string]core.StateValue{
			"messages": []interface{}{map[string]interface{}{"content": "x"}},
		})
	}
	raw, _ := state.Get("messages")
	assert.Len(t, raw.([]interface{}), 3)
}

// Channels without a declared reducer use last-write-wins, LangGraph's default.
func TestConformance_DefaultChannelIsLastWriteWins(t *testing.T) {
	schema := core.NewStateSchema().
		AddChannel("items", core.Append, func() core.StateValue { return []interface{}{} })

	state := schema.NewState()
	schema.ApplyUpdates(state, map[string]core.StateValue{"plain": "first", "items": []interface{}{1}})
	schema.ApplyUpdates(state, map[string]core.StateValue{"plain": "second", "items": []interface{}{2}})

	plain, _ := state.Get("plain")
	assert.Equal(t, "second", plain, "undeclared channels overwrite")
	items, _ := state.Get("items")
	assert.Equal(t, []interface{}{1, 2}, items, "declared channels reduce")
}

// Numeric and map reducers behave like operator.add and dict merge.
func TestConformance_NumericAndMapReducers(t *testing.T) {
	schema := core.NewStateSchema().
		AddChannel("total", core.SumInt, func() core.StateValue { return 0 }).
		AddChannel("ratio", core.SumFloat, func() core.StateValue { return 0.0 }).
		AddChannel("attrs", core.MergeMap, func() core.StateValue { return map[string]interface{}{} })

	state := schema.NewState()
	schema.ApplyUpdates(state, map[string]core.StateValue{
		"total": 5, "ratio": 1.5, "attrs": map[string]interface{}{"a": 1},
	})
	schema.ApplyUpdates(state, map[string]core.StateValue{
		"total": 7, "ratio": 2.25, "attrs": map[string]interface{}{"b": 2},
	})

	total, _ := state.Get("total")
	assert.Equal(t, 12, total)
	ratio, _ := state.Get("ratio")
	assert.InDelta(t, 3.75, ratio.(float64), 1e-9)
	attrs, _ := state.Get("attrs")
	assert.Equal(t, map[string]interface{}{"a": 1, "b": 2}, attrs)
}

// LangGraph: parallel branches in one super-step each see the same input state
// and their updates are combined through the channel reducers.
func TestConformance_ParallelBranchesMergeViaReducers(t *testing.T) {
	schema := core.NewStateSchema().
		AddChannel("log", core.Append, func() core.StateValue { return []interface{}{} }).
		AddChannel("total", core.SumInt, func() core.StateValue { return 0 })

	g := core.NewGraph("fanout").WithStateSchema(schema)
	for _, name := range []string{"alpha", "beta", "gamma"} {
		n := name
		g.AddUpdateNode(n, n, func(ctx context.Context, s *core.BaseState) (map[string]core.StateValue, error) {
			// Every branch must observe the same unmodified input.
			seen, _ := s.Get("total")
			if seen != 0 {
				return nil, errors.New("branch observed another branch's write")
			}
			return map[string]core.StateValue{
				"log":   []interface{}{n},
				"total": 1,
			}, nil
		})
	}
	require.NoError(t, g.SetStartNode("alpha"))

	in := schema.NewState()
	out, err := g.ExecuteParallelUpdates(context.Background(), []string{"alpha", "beta", "gamma"}, in)
	require.NoError(t, err)

	total, _ := out.Get("total")
	assert.Equal(t, 3, total, "each branch contributes through the reducer")
	log, _ := out.Get("log")
	assert.Equal(t, []interface{}{"alpha", "beta", "gamma"}, log,
		"merge order follows the declared branch order, not completion order")
}

// Merging must stay deterministic even when branches finish out of order.
func TestConformance_ParallelMergeIsOrderIndependent(t *testing.T) {
	schema := core.NewStateSchema().
		AddChannel("log", core.Append, func() core.StateValue { return []interface{}{} })

	g := core.NewGraph("fanout-timing").WithStateSchema(schema)
	delays := map[string]time.Duration{"first": 30 * time.Millisecond, "second": 10 * time.Millisecond, "third": 0}
	for name, d := range delays {
		n, delay := name, d
		g.AddUpdateNode(n, n, func(ctx context.Context, s *core.BaseState) (map[string]core.StateValue, error) {
			time.Sleep(delay)
			return map[string]core.StateValue{"log": []interface{}{n}}, nil
		})
	}
	require.NoError(t, g.SetStartNode("first"))

	for i := 0; i < 10; i++ {
		out, err := g.ExecuteParallelUpdates(context.Background(), []string{"first", "second", "third"}, schema.NewState())
		require.NoError(t, err)
		log, _ := out.Get("log")
		require.Equal(t, []interface{}{"first", "second", "third"}, log)
	}
}

// A failing branch must not discard the successful branches' work.
func TestConformance_ParallelPartialFailure(t *testing.T) {
	schema := core.NewStateSchema().
		AddChannel("log", core.Append, func() core.StateValue { return []interface{}{} })

	g := core.NewGraph("fanout-fail").WithStateSchema(schema)
	g.AddUpdateNode("good", "good", func(ctx context.Context, s *core.BaseState) (map[string]core.StateValue, error) {
		return map[string]core.StateValue{"log": []interface{}{"good"}}, nil
	})
	g.AddUpdateNode("bad", "bad", func(ctx context.Context, s *core.BaseState) (map[string]core.StateValue, error) {
		return nil, errors.New("branch failed")
	})
	g.AddUpdateNode("panicky", "panicky", func(ctx context.Context, s *core.BaseState) (map[string]core.StateValue, error) {
		panic("branch exploded")
	})
	require.NoError(t, g.SetStartNode("good"))

	out, err := g.ExecuteParallelUpdates(context.Background(), []string{"good", "bad", "panicky"}, schema.NewState())
	require.Error(t, err)
	assert.Contains(t, err.Error(), "branch failed")
	assert.True(t, errors.Is(err, core.ErrNodePanic), "a panicking branch is reported, not fatal")

	require.NotNil(t, out)
	log, _ := out.Get("log")
	assert.Equal(t, []interface{}{"good"}, log, "successful branch work is preserved")
}

// State values must be isolated between the caller and the graph: mutating a
// state after handing it to the engine must not change the run.
func TestConformance_StateIsDeepCopied(t *testing.T) {
	in := core.NewBaseState()
	nested := map[string]interface{}{"list": []interface{}{1, 2}}
	in.Set("nested", nested)

	clone := in.Clone()
	nested["list"] = append(nested["list"].([]interface{}), 3)
	nested["added"] = true

	got, _ := clone.Get("nested")
	gotMap := got.(map[string]interface{})
	assert.Equal(t, []interface{}{1, 2}, gotMap["list"], "clone must not alias caller data")
	_, added := gotMap["added"]
	assert.False(t, added)
}

// Values that reflection cannot rebuild (structs with unexported fields such as
// time.Time) must round-trip rather than panic.
func TestConformance_CloneHandlesOpaqueValues(t *testing.T) {
	now := time.Now()
	in := core.NewBaseState()
	in.Set("ts", now)
	in.Set("dur", 5*time.Second)
	in.Set("nested", map[string]interface{}{"at": now})

	clone := in.Clone()
	ts, ok := clone.Get("ts")
	require.True(t, ok)
	assert.True(t, now.Equal(ts.(time.Time)))
	nested, _ := clone.Get("nested")
	assert.True(t, now.Equal(nested.(map[string]interface{})["at"].(time.Time)))
}

// Self-referential structures must not hang the copier.
func TestConformance_CloneHandlesCycles(t *testing.T) {
	cyclic := map[string]interface{}{"name": "root"}
	cyclic["self"] = cyclic

	in := core.NewBaseState()
	in.Set("cyclic", cyclic)

	done := make(chan struct{})
	go func() {
		defer close(done)
		_ = in.Clone()
	}()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("Clone did not terminate on a cyclic structure")
	}
}

// State must survive a JSON round-trip; anything less silently empties every
// checkpoint and API response.
func TestConformance_StateJSONRoundTrip(t *testing.T) {
	in := core.NewBaseState()
	in.Set("counter", 42)
	in.Set("messages", []interface{}{msg("1", "hi")})
	in.SetMetadata("thread", "t-1")

	encoded, err := in.ToJSON()
	require.NoError(t, err)
	assert.Contains(t, string(encoded), "counter")

	out := core.NewBaseState()
	require.NoError(t, out.FromJSON(encoded))

	counter, ok := out.Get("counter")
	require.True(t, ok)
	assert.EqualValues(t, 42, counter)
	thread, ok := out.GetMetadata("thread")
	require.True(t, ok)
	assert.Equal(t, "t-1", thread)

	// Writing after a round-trip must not panic on a nil map.
	out.Set("after", true)
}

// Concurrent readers and writers of one state must be race-free.
func TestConformance_StateConcurrentAccess(t *testing.T) {
	state := core.NewBaseState()
	var wg sync.WaitGroup
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			for j := 0; j < 50; j++ {
				state.Set("k", i*j)
				_, _ = state.Get("k")
				_ = state.Keys()
				_ = state.GetAll()
				_ = state.Clone()
			}
		}(i)
	}
	wg.Wait()
}
