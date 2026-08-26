// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package server

import (
	"sort"
	"sync"

	"github.com/piotrlaczkowski/GoLangGraph/pkg/core"
)

// GraphManager holds the graphs a server exposes over the API. Registering a
// graph makes it listable, inspectable, executable and streamable, which is
// what GoLangGraph Studio needs to debug a workflow.
type GraphManager struct {
	mu     sync.RWMutex
	graphs map[string]*core.Graph
	order  []string
}

// NewGraphManager creates an empty graph manager.
func NewGraphManager() *GraphManager {
	return &GraphManager{graphs: make(map[string]*core.Graph)}
}

// Register adds a graph under an ID. Re-registering an ID replaces the graph.
func (gm *GraphManager) Register(id string, g *core.Graph) {
	if gm == nil || g == nil || id == "" {
		return
	}
	gm.mu.Lock()
	defer gm.mu.Unlock()
	if _, exists := gm.graphs[id]; !exists {
		gm.order = append(gm.order, id)
	}
	gm.graphs[id] = g
}

// Unregister removes a graph.
func (gm *GraphManager) Unregister(id string) {
	if gm == nil {
		return
	}
	gm.mu.Lock()
	defer gm.mu.Unlock()
	delete(gm.graphs, id)
	for i, existing := range gm.order {
		if existing == id {
			gm.order = append(gm.order[:i], gm.order[i+1:]...)
			break
		}
	}
}

// Get returns a graph by ID.
func (gm *GraphManager) Get(id string) (*core.Graph, bool) {
	if gm == nil {
		return nil, false
	}
	gm.mu.RLock()
	defer gm.mu.RUnlock()
	g, ok := gm.graphs[id]
	return g, ok
}

// List returns registered graph IDs in registration order.
func (gm *GraphManager) List() []string {
	if gm == nil {
		return nil
	}
	gm.mu.RLock()
	defer gm.mu.RUnlock()
	return append([]string(nil), gm.order...)
}

// GraphNodeView describes a node for API clients.
type GraphNodeView struct {
	ID       string                 `json:"id"`
	Name     string                 `json:"name"`
	Type     string                 `json:"type"`
	IsStart  bool                   `json:"is_start"`
	IsEnd    bool                   `json:"is_end"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// GraphEdgeView describes an edge for API clients.
type GraphEdgeView struct {
	From        string `json:"from"`
	To          string `json:"to"`
	Conditional bool   `json:"conditional"`
	// RouteKey is set for conditional edges and names the routing key that
	// selects this destination.
	RouteKey string `json:"route_key,omitempty"`
}

// GraphTopologyView is the serialisable topology of a graph. Studio renders
// this directly, so both nodes and edges are always present (never null).
type GraphTopologyView struct {
	Nodes []GraphNodeView `json:"nodes"`
	Edges []GraphEdgeView `json:"edges"`
}

// GraphSummaryView describes a graph without its topology.
type GraphSummaryView struct {
	ID        string   `json:"id"`
	Name      string   `json:"name"`
	StartNode string   `json:"start_node"`
	EndNodes  []string `json:"end_nodes"`
	NodeCount int      `json:"node_count"`
	EdgeCount int      `json:"edge_count"`
	Running   bool     `json:"running"`
}

// summariseGraph builds a summary view of a graph.
func summariseGraph(id string, g *core.Graph) GraphSummaryView {
	topo := describeTopology(g)
	endNodes := append([]string(nil), g.EndNodes...)
	if endNodes == nil {
		endNodes = []string{}
	}
	return GraphSummaryView{
		ID:        id,
		Name:      g.Name,
		StartNode: g.StartNode,
		EndNodes:  endNodes,
		NodeCount: len(topo.Nodes),
		EdgeCount: len(topo.Edges),
		Running:   g.IsRunning(),
	}
}

// describeTopology converts a graph into its serialisable topology, including
// conditional routes so a client sees every reachable path.
func describeTopology(g *core.Graph) GraphTopologyView {
	view := GraphTopologyView{Nodes: []GraphNodeView{}, Edges: []GraphEdgeView{}}
	if g == nil {
		return view
	}

	ids := make([]string, 0, len(g.Nodes))
	for id := range g.Nodes {
		ids = append(ids, id)
	}
	sort.Strings(ids)

	for _, id := range ids {
		node := g.Nodes[id]
		if node == nil {
			continue
		}
		nodeType := "node"
		if t, ok := node.Metadata["type"].(string); ok && t != "" {
			nodeType = t
		}
		view.Nodes = append(view.Nodes, GraphNodeView{
			ID:       node.ID,
			Name:     node.Name,
			Type:     nodeType,
			IsStart:  node.ID == g.StartNode,
			IsEnd:    g.IsEndNode(node.ID),
			Metadata: node.Metadata,
		})
	}

	// Static edges, ordered for a stable rendering.
	static := make([]GraphEdgeView, 0, len(g.Edges))
	for _, edge := range g.Edges {
		if edge == nil {
			continue
		}
		static = append(static, GraphEdgeView{
			From:        edge.From,
			To:          edge.To,
			Conditional: edge.Condition != nil,
		})
	}
	sort.Slice(static, func(i, j int) bool {
		if static[i].From != static[j].From {
			return static[i].From < static[j].From
		}
		return static[i].To < static[j].To
	})
	view.Edges = append(view.Edges, static...)

	// Conditional routes registered via AddConditionalEdges.
	for _, id := range ids {
		ce, ok := g.GetConditionalEdge(id)
		if !ok || ce == nil {
			continue
		}
		keys := make([]string, 0, len(ce.Routes))
		for k := range ce.Routes {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, key := range keys {
			view.Edges = append(view.Edges, GraphEdgeView{
				From:        id,
				To:          ce.Routes[key],
				Conditional: true,
				RouteKey:    key,
			})
		}
	}

	return view
}

// ExecutionStepView is a single node execution, as sent to clients.
type ExecutionStepView struct {
	NodeID    string                     `json:"node_id"`
	Step      int                        `json:"step"`
	Success   bool                       `json:"success"`
	Error     string                     `json:"error,omitempty"`
	DurationM float64                    `json:"duration_ms"`
	Attempts  int                        `json:"attempts"`
	State     map[string]core.StateValue `json:"state,omitempty"`
}

// describeStep converts an engine result into its wire representation.
func describeStep(r *core.ExecutionResult) ExecutionStepView {
	view := ExecutionStepView{
		NodeID:    r.NodeID,
		Step:      r.Step,
		Success:   r.Success,
		Error:     r.ErrorMessage,
		DurationM: float64(r.Duration.Microseconds()) / 1000.0,
		Attempts:  r.Attempts,
	}
	if r.State != nil {
		view.State = r.State.GetAll()
	}
	return view
}
