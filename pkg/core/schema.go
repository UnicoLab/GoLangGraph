// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package core

import (
	"context"
	"fmt"
	"sort"
	"sync"
)

// Reducer combines an existing channel value with an update, mirroring
// LangGraph's channel reducers (for example operator.add or add_messages).
//
// A reducer must not mutate its arguments; it returns the new value.
type Reducer func(existing, update StateValue) StateValue

// Channel describes one key of the graph state: how updates are combined and
// what the value is before anything has been written.
type Channel struct {
	Key     string
	Reducer Reducer
	Default func() StateValue
}

// StateSchema declares the channels of a graph state. It is the GoLangGraph
// equivalent of a LangGraph TypedDict state annotated with reducers.
//
// A nil schema, or a key with no declared channel, uses last-write-wins.
type StateSchema struct {
	mu       sync.RWMutex
	channels map[string]*Channel
	order    []string
}

// NewStateSchema creates an empty schema.
func NewStateSchema() *StateSchema {
	return &StateSchema{channels: make(map[string]*Channel)}
}

// AddChannel declares a channel with a reducer and optional default factory.
// Re-declaring a key replaces the previous channel.
func (s *StateSchema) AddChannel(key string, reducer Reducer, def func() StateValue) *StateSchema {
	if s == nil {
		return s
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.channels == nil {
		s.channels = make(map[string]*Channel)
	}
	if _, exists := s.channels[key]; !exists {
		s.order = append(s.order, key)
	}
	s.channels[key] = &Channel{Key: key, Reducer: reducer, Default: def}
	return s
}

// Reducer returns the reducer for a key, or nil when the key uses
// last-write-wins.
func (s *StateSchema) Reducer(key string) Reducer {
	if s == nil {
		return nil
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if ch, ok := s.channels[key]; ok {
		return ch.Reducer
	}
	return nil
}

// Default returns the zero value for a key before any write.
func (s *StateSchema) Default(key string) StateValue {
	if s == nil {
		return nil
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if ch, ok := s.channels[key]; ok && ch.Default != nil {
		return ch.Default()
	}
	return nil
}

// Keys returns declared channel keys in declaration order.
func (s *StateSchema) Keys() []string {
	if s == nil {
		return nil
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	return append([]string(nil), s.order...)
}

// NewState builds a state pre-populated with each channel's default value.
func (s *StateSchema) NewState() *BaseState {
	state := NewBaseState()
	if s == nil {
		return state
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	for _, key := range s.order {
		if ch := s.channels[key]; ch != nil && ch.Default != nil {
			state.Set(key, ch.Default())
		}
	}
	return state
}

// ApplyUpdates merges a map of channel updates into a state through reducers.
// Keys are applied in sorted order so the result is deterministic.
func (s *StateSchema) ApplyUpdates(state *BaseState, updates map[string]StateValue) {
	if state == nil || len(updates) == 0 {
		return
	}
	keys := make([]string, 0, len(updates))
	for k := range updates {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		state.Update(s, k, updates[k])
	}
}

// ---------------------------------------------------------------------------
// Built-in reducers
// ---------------------------------------------------------------------------

// LastValue overwrites the existing value. This is the default behavior for
// channels without a reducer.
func LastValue(existing, update StateValue) StateValue { return update }

// Append concatenates slice updates onto the existing slice, mirroring
// LangGraph's operator.add on list channels. Non-slice updates are appended as
// a single element.
func Append(existing, update StateValue) StateValue {
	out := toSlice(existing)
	switch u := update.(type) {
	case nil:
		return out
	case []interface{}:
		out = append(out, u...)
	default:
		if items, ok := asInterfaceSlice(update); ok {
			out = append(out, items...)
		} else {
			out = append(out, update)
		}
	}
	return out
}

// AddMessages appends messages and replaces any existing message that shares an
// "id" with an incoming one, matching LangGraph's add_messages reducer.
func AddMessages(existing, update StateValue) StateValue {
	current := toSlice(existing)
	incoming := toSlice(update)
	if update != nil && len(incoming) == 0 {
		incoming = []interface{}{update}
	}

	out := make([]interface{}, len(current))
	copy(out, current)

	for _, msg := range incoming {
		id, hasID := messageID(msg)
		if hasID {
			replaced := false
			for i, existingMsg := range out {
				if existingID, ok := messageID(existingMsg); ok && existingID == id {
					out[i] = msg
					replaced = true
					break
				}
			}
			if replaced {
				continue
			}
		}
		out = append(out, msg)
	}
	return out
}

// SumInt adds integer updates to the existing value.
func SumInt(existing, update StateValue) StateValue {
	return toInt(existing) + toInt(update)
}

// SumFloat adds float updates to the existing value.
func SumFloat(existing, update StateValue) StateValue {
	return toFloat(existing) + toFloat(update)
}

// MergeMap merges map updates key-by-key into the existing map.
func MergeMap(existing, update StateValue) StateValue {
	out := make(map[string]interface{})
	if em, ok := existing.(map[string]interface{}); ok {
		for k, v := range em {
			out[k] = v
		}
	}
	if um, ok := update.(map[string]interface{}); ok {
		for k, v := range um {
			out[k] = v
		}
	}
	return out
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

func toSlice(v StateValue) []interface{} {
	if v == nil {
		return nil
	}
	if s, ok := v.([]interface{}); ok {
		return s
	}
	if s, ok := asInterfaceSlice(v); ok {
		return s
	}
	return nil
}

// asInterfaceSlice converts typed slices (for example []map[string]any or
// []string) to []interface{} so reducers work with concrete Go slices too.
func asInterfaceSlice(v StateValue) ([]interface{}, bool) {
	switch s := v.(type) {
	case []interface{}:
		return s, true
	case []string:
		out := make([]interface{}, len(s))
		for i := range s {
			out[i] = s[i]
		}
		return out, true
	case []int:
		out := make([]interface{}, len(s))
		for i := range s {
			out[i] = s[i]
		}
		return out, true
	case []map[string]interface{}:
		out := make([]interface{}, len(s))
		for i := range s {
			out[i] = s[i]
		}
		return out, true
	}
	return nil, false
}

func messageID(msg interface{}) (string, bool) {
	m, ok := msg.(map[string]interface{})
	if !ok {
		return "", false
	}
	id, ok := m["id"]
	if !ok {
		return "", false
	}
	s, ok := id.(string)
	if !ok || s == "" {
		return "", false
	}
	return s, true
}

func toInt(v StateValue) int {
	switch n := v.(type) {
	case int:
		return n
	case int32:
		return int(n)
	case int64:
		return int(n)
	case float64:
		return int(n)
	case float32:
		return int(n)
	}
	return 0
}

func toFloat(v StateValue) float64 {
	switch n := v.(type) {
	case float64:
		return n
	case float32:
		return float64(n)
	case int:
		return float64(n)
	case int32:
		return float64(n)
	case int64:
		return float64(n)
	}
	return 0
}

// ---------------------------------------------------------------------------
// Update-style nodes
// ---------------------------------------------------------------------------

// UpdateFunc is a node that returns only the channels it changed, the way a
// LangGraph node returns a partial state dict. Returning nil means no update.
type UpdateFunc func(ctx context.Context, state *BaseState) (map[string]StateValue, error)

// WithStateSchema attaches a state schema whose reducers are applied to the
// updates returned by nodes registered with AddUpdateNode.
func (g *Graph) WithStateSchema(schema *StateSchema) *Graph {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.schema = schema
	return g
}

// StateSchema returns the graph's state schema, if any.
func (g *Graph) StateSchema() *StateSchema {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return g.schema
}

// AddUpdateNode registers a node that returns partial channel updates. The
// updates are merged into the running state using the graph's state schema, so
// reducers such as Append and AddMessages apply exactly as they do in LangGraph.
func (g *Graph) AddUpdateNode(id, name string, fn UpdateFunc) *Node {
	if fn == nil {
		node := g.AddNode(id, name, nil)
		return node
	}
	wrapped := func(ctx context.Context, state *BaseState) (*BaseState, error) {
		updates, err := fn(ctx, state)
		if err != nil {
			return nil, err
		}
		if len(updates) == 0 {
			return state, nil
		}
		out := state.Clone()
		g.StateSchema().ApplyUpdates(out, updates)
		return out, nil
	}
	node := g.AddNode(id, name, wrapped)
	node.updateFn = fn
	return node
}

// ExecuteParallelUpdates runs several update-style nodes concurrently against
// the same input state and merges their updates through the schema's reducers,
// implementing a LangGraph super-step over parallel branches.
//
// Branch updates are applied in the order the node IDs were supplied, so the
// merged result is deterministic regardless of completion order.
func (g *Graph) ExecuteParallelUpdates(ctx context.Context, nodeIDs []string, state *BaseState) (*BaseState, error) {
	if state == nil {
		state = NewBaseState()
	}
	if len(nodeIDs) == 0 {
		return state.Clone(), nil
	}

	g.mu.RLock()
	schema := g.schema
	nodes := make([]*Node, 0, len(nodeIDs))
	missing := ""
	for _, id := range nodeIDs {
		n, ok := g.Nodes[id]
		if !ok {
			missing = id
			break
		}
		nodes = append(nodes, n)
	}
	g.mu.RUnlock()

	if missing != "" {
		return nil, fmt.Errorf("node %s does not exist", missing)
	}

	updates := make([]map[string]StateValue, len(nodes))
	errs := make([]error, len(nodes))
	var wg sync.WaitGroup

	for i, node := range nodes {
		if node.updateFn == nil {
			return nil, fmt.Errorf("node %s was not registered with AddUpdateNode", node.ID)
		}
		wg.Add(1)
		go func(idx int, n *Node) {
			defer wg.Done()
			defer func() {
				if r := recover(); r != nil {
					errs[idx] = fmt.Errorf("%w: node %s: %v", ErrNodePanic, n.ID, r)
				}
			}()
			// Each branch sees an isolated copy of the input state.
			u, err := n.updateFn(ctx, state.Clone())
			if err != nil {
				errs[idx] = fmt.Errorf("node %s failed: %w", n.ID, err)
				return
			}
			updates[idx] = u
		}(i, node)
	}
	wg.Wait()

	merged := state.Clone()
	for i := range updates {
		if errs[i] != nil {
			continue
		}
		schema.ApplyUpdates(merged, updates[i])
	}

	if joined := joinErrors(errs); joined != nil {
		return merged, joined
	}
	return merged, nil
}
