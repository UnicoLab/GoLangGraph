// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package core

import (
	"context"
	"errors"
	"fmt"
)

// SubgraphOptions controls how a nested graph exchanges state with its parent.
type SubgraphOptions struct {
	// InputKeys restricts what the subgraph sees. Empty means the whole state.
	InputKeys []string
	// OutputKeys restricts what is written back to the parent. Empty means the
	// whole subgraph output state.
	OutputKeys []string
	// Namespace, when set, writes the subgraph output under this single key as
	// a map[string]interface{} instead of merging keys into the parent state.
	Namespace string
	// Schema applies reducers when merging subgraph output into parent state.
	// When nil, the parent graph's schema is used.
	Schema *StateSchema
	// PropagateInterrupts surfaces a subgraph interrupt to the parent instead of
	// treating it as an error.
	PropagateInterrupts bool
}

// ErrSubgraphInterrupted wraps an interrupt raised inside a nested graph.
var ErrSubgraphInterrupted = errors.New("subgraph interrupted")

// AddSubgraph registers a compiled graph as a node of this graph, mirroring
// LangGraph's ability to use a compiled graph as a node.
//
// The subgraph runs to completion with its own recursion limit and routing; its
// resulting state is merged back into the parent according to opts.
func (g *Graph) AddSubgraph(id, name string, sub *Graph, opts *SubgraphOptions) (*Node, error) {
	if sub == nil {
		return nil, fmt.Errorf("subgraph %s: nested graph must not be nil", id)
	}
	if sub == g {
		return nil, fmt.Errorf("subgraph %s: a graph cannot contain itself", id)
	}
	if err := sub.Validate(); err != nil {
		return nil, fmt.Errorf("subgraph %s: %w", id, err)
	}
	if contains, path := graphContains(sub, g, make(map[*Graph]bool)); contains {
		return nil, fmt.Errorf("subgraph %s would create a cycle of graphs: %s", id, path)
	}

	if opts == nil {
		opts = &SubgraphOptions{}
	}
	// Copy so later caller mutations cannot change node behaviour.
	local := *opts
	local.InputKeys = append([]string(nil), opts.InputKeys...)
	local.OutputKeys = append([]string(nil), opts.OutputKeys...)

	fn := func(ctx context.Context, state *BaseState) (*BaseState, error) {
		input := projectState(state, local.InputKeys)

		out, err := sub.Execute(ctx, input)
		if err != nil {
			var ie *InterruptError
			if errors.As(err, &ie) && local.PropagateInterrupts {
				return nil, fmt.Errorf("%w: %s: %w", ErrSubgraphInterrupted, id, err)
			}
			return nil, fmt.Errorf("subgraph %s failed: %w", id, err)
		}

		merged := state.Clone()
		schema := local.Schema
		if schema == nil {
			schema = g.StateSchema()
		}

		if local.Namespace != "" {
			merged.Set(local.Namespace, out.GetAll())
			return merged, nil
		}

		projected := projectState(out, local.OutputKeys)
		merged.MergeWithSchema(projected, schema)
		return merged, nil
	}

	node := g.AddNode(id, name, fn)
	node.Metadata["type"] = "subgraph"
	node.Metadata["subgraph_id"] = sub.ID
	node.Metadata["subgraph_name"] = sub.Name

	g.mu.Lock()
	if g.subgraphs == nil {
		g.subgraphs = make(map[string]*Graph)
	}
	g.subgraphs[id] = sub
	g.mu.Unlock()

	return node, nil
}

// Subgraph returns the nested graph registered under a node ID.
func (g *Graph) Subgraph(nodeID string) (*Graph, bool) {
	g.mu.RLock()
	defer g.mu.RUnlock()
	sub, ok := g.subgraphs[nodeID]
	return sub, ok
}

// Subgraphs returns nested graphs by node ID.
func (g *Graph) Subgraphs() map[string]*Graph {
	g.mu.RLock()
	defer g.mu.RUnlock()
	out := make(map[string]*Graph, len(g.subgraphs))
	for k, v := range g.subgraphs {
		out[k] = v
	}
	return out
}

// graphContains reports whether target is reachable from root through nested
// subgraphs, which would make the composition infinitely recursive.
func graphContains(root, target *Graph, seen map[*Graph]bool) (bool, string) {
	if root == nil || seen[root] {
		return false, ""
	}
	seen[root] = true

	root.mu.RLock()
	nested := make(map[string]*Graph, len(root.subgraphs))
	for k, v := range root.subgraphs {
		nested[k] = v
	}
	root.mu.RUnlock()

	for id, sub := range nested {
		if sub == target {
			return true, fmt.Sprintf("%s -> %s", root.Name, id)
		}
		if found, path := graphContains(sub, target, seen); found {
			return true, fmt.Sprintf("%s -> %s", root.Name, path)
		}
	}
	return false, ""
}

// projectState returns a state limited to the given keys. Empty keys means the
// whole state.
func projectState(state *BaseState, keys []string) *BaseState {
	if state == nil {
		return NewBaseState()
	}
	if len(keys) == 0 {
		return state.Clone()
	}
	out := NewBaseState()
	for _, k := range keys {
		if v, ok := state.Get(k); ok {
			out.Set(k, v)
		}
	}
	return out
}
