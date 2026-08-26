// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

package core

import (
	"context"
	"encoding/json"
	"testing"
	"time"
)

// FuzzStateJSONRoundTrip checks that any payload the state accepts can be
// re-encoded and re-read without panicking or leaving the state unusable.
// State arrives from checkpoints and API requests, so it is untrusted input.
func FuzzStateJSONRoundTrip(f *testing.F) {
	f.Add(`{"data":{"a":1},"metadata":{}}`)
	f.Add(`{}`)
	f.Add(`{"data":null,"metadata":null}`)
	f.Add(`{"a":1,"b":[1,2,3]}`)
	f.Add(`{"data":{"nested":{"deep":[{"x":1}]}}}`)
	f.Add(`[]`)
	f.Add(`null`)
	f.Add(`{"data":{"big":1e309}}`)

	f.Fuzz(func(t *testing.T, payload string) {
		state := NewBaseState()
		if err := state.FromJSON([]byte(payload)); err != nil {
			return // rejecting bad input is the correct outcome
		}

		// A state that loaded must remain fully usable.
		state.Set("probe", "value")
		if v, ok := state.Get("probe"); !ok || v != "value" {
			t.Fatalf("state unusable after loading %q", payload)
		}
		_ = state.Keys()
		_ = state.GetAll()

		clone := state.Clone()
		if clone == nil {
			t.Fatalf("clone returned nil for %q", payload)
		}

		encoded, err := state.ToJSON()
		if err != nil {
			return
		}

		// The re-encoded form must load again.
		again := NewBaseState()
		if err := again.FromJSON(encoded); err != nil {
			t.Fatalf("re-encoded state failed to load: %v (original %q)", err, payload)
		}
	})
}

// FuzzStateMarshalUnmarshal exercises the json.Marshaler path used by every
// checkpoint and API response.
func FuzzStateMarshalUnmarshal(f *testing.F) {
	f.Add(`{"data":{"k":"v"},"metadata":{"m":1}}`)
	f.Add(`{"k":"v"}`)
	f.Add(``)

	f.Fuzz(func(t *testing.T, payload string) {
		var state BaseState
		if err := json.Unmarshal([]byte(payload), &state); err != nil {
			return
		}
		out, err := json.Marshal(&state)
		if err != nil {
			t.Fatalf("a state that unmarshalled must marshal again: %v", err)
		}
		var again BaseState
		if err := json.Unmarshal(out, &again); err != nil {
			t.Fatalf("round trip broke: %v (payload %q, encoded %q)", err, payload, out)
		}
	})
}

// FuzzDeepCopy checks the copier against arbitrary decoded JSON, which is the
// shape state values actually take.
func FuzzDeepCopy(f *testing.F) {
	f.Add(`{"a":[1,{"b":null}],"c":"x"}`)
	f.Add(`[[[[[1]]]]]`)
	f.Add(`{"n":1.5e10}`)

	f.Fuzz(func(t *testing.T, payload string) {
		var value interface{}
		if err := json.Unmarshal([]byte(payload), &value); err != nil {
			return
		}

		done := make(chan struct{})
		go func() {
			defer close(done)
			_ = deepCopy(value)
		}()

		select {
		case <-done:
		case <-time.After(10 * time.Second):
			t.Fatalf("deepCopy did not terminate for %q", payload)
		}
	})
}

// FuzzGraphRouting drives the router with arbitrary condition results. A
// routing function is user code and may return anything, including node IDs
// that do not exist; the engine must report that rather than misbehave.
func FuzzGraphRouting(f *testing.F) {
	f.Add("a", "b")
	f.Add("", "")
	f.Add("__end__", "nonexistent")
	f.Add("a", "__end__")

	f.Fuzz(func(t *testing.T, routeKey, target string) {
		g := NewGraph("fuzz")
		g.Config.MaxIterations = 8
		g.Config.EnableStreaming = false

		g.AddNode("start", "start", func(ctx context.Context, s *BaseState) (*BaseState, error) {
			return s, nil
		})
		g.AddNode("a", "a", func(ctx context.Context, s *BaseState) (*BaseState, error) {
			return s, nil
		})
		if err := g.SetStartNode("start"); err != nil {
			return
		}

		// Routes are only registered for targets the graph actually has, which
		// is what AddConditionalEdges enforces.
		routes := map[string]string{}
		if target == "a" || target == END {
			routes[routeKey] = target
		}
		if err := g.AddConditionalEdges("start",
			func(ctx context.Context, s *BaseState) (string, error) { return routeKey, nil },
			routes); err != nil {
			return
		}

		done := make(chan struct{})
		go func() {
			defer close(done)
			// Any outcome is acceptable except a panic or a hang.
			_, _ = g.Execute(context.Background(), NewBaseState())
		}()

		select {
		case <-done:
		case <-time.After(20 * time.Second):
			t.Fatalf("execution hung for routeKey=%q target=%q", routeKey, target)
		}
	})
}

// FuzzGraphConstruction builds graphs from arbitrary identifiers. Validation
// must either accept the graph or report why, never panic.
func FuzzGraphConstruction(f *testing.F) {
	f.Add("a", "b", "c")
	f.Add("", "", "")
	f.Add("__start__", "__end__", "x")
	f.Add("same", "same", "same")

	f.Fuzz(func(t *testing.T, nodeA, nodeB, start string) {
		g := NewGraph("fuzz-build")
		noop := func(ctx context.Context, s *BaseState) (*BaseState, error) { return s, nil }

		g.AddNode(nodeA, nodeA, noop)
		g.AddNode(nodeB, nodeB, noop)
		g.AddEdge(nodeA, nodeB, nil)
		_ = g.SetStartNode(start)
		_ = g.AddEndNode(nodeB)

		if err := g.Validate(); err != nil {
			return // reporting an invalid graph is correct
		}

		_ = g.GetTopology()
		done := make(chan struct{})
		go func() {
			defer close(done)
			_, _ = g.Execute(context.Background(), NewBaseState())
		}()
		select {
		case <-done:
		case <-time.After(20 * time.Second):
			t.Fatalf("execution hung for nodes %q/%q start %q", nodeA, nodeB, start)
		}
	})
}
