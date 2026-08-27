// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package server

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"sync"

	"github.com/UnicoLab/GoLangGraph/pkg/core"
)

// AgentPipelineNode is a safe, declarative pipeline step.  It refers to an
// already-registered agent rather than accepting a function or source code,
// making it suitable for authoring from Studio without turning the API into a
// remote-code-execution surface.
type AgentPipelineNode struct {
	ID      string `json:"id"`
	AgentID string `json:"agent_id"`
	Name    string `json:"name,omitempty"`
}

// AgentPipelineDefinition is the Studio authoring contract for an executable
// sequential multi-agent pipeline.  The output of each step becomes the input
// of the next one and the full output remains available in graph state.
type AgentPipelineDefinition struct {
	ID    string              `json:"id"`
	Name  string              `json:"name"`
	Nodes []AgentPipelineNode `json:"nodes"`
}

// BuildAgentPipeline compiles a data-only Studio definition into a core graph.
// It intentionally supports a sequential topology only.  Conditional routing
// and arbitrary custom nodes require application code, where their behavior
// can be reviewed and tested, instead of being faked by the visual editor.
func BuildAgentPipeline(def AgentPipelineDefinition, agents *AgentManager) (*core.Graph, error) {
	def.ID = strings.TrimSpace(def.ID)
	def.Name = strings.TrimSpace(def.Name)
	if def.ID == "" {
		return nil, fmt.Errorf("pipeline id is required")
	}
	if def.Name == "" {
		return nil, fmt.Errorf("pipeline name is required")
	}
	if len(def.Nodes) == 0 {
		return nil, fmt.Errorf("pipeline must contain at least one agent node")
	}
	if agents == nil {
		return nil, fmt.Errorf("agent manager not available")
	}

	g := core.NewGraph(def.Name)
	g.Metadata["studio_pipeline"] = true
	g.Metadata["pipeline_id"] = def.ID
	seen := make(map[string]struct{}, len(def.Nodes))
	for index, pipelineNode := range def.Nodes {
		pipelineNode.ID = strings.TrimSpace(pipelineNode.ID)
		pipelineNode.AgentID = strings.TrimSpace(pipelineNode.AgentID)
		if pipelineNode.ID == "" || pipelineNode.AgentID == "" {
			return nil, fmt.Errorf("pipeline node %d requires id and agent_id", index+1)
		}
		if _, duplicate := seen[pipelineNode.ID]; duplicate {
			return nil, fmt.Errorf("duplicate pipeline node id %q", pipelineNode.ID)
		}
		seen[pipelineNode.ID] = struct{}{}

		instance, exists := agents.GetAgent(pipelineNode.AgentID)
		if !exists {
			return nil, fmt.Errorf("agent %q for pipeline node %q was not found", pipelineNode.AgentID, pipelineNode.ID)
		}
		name := strings.TrimSpace(pipelineNode.Name)
		if name == "" {
			name = instance.Name()
		}
		agentID := pipelineNode.AgentID // capture a distinct value for each closure
		nodeID := pipelineNode.ID
		node := g.AddNode(nodeID, name, func(ctx context.Context, state *core.BaseState) (*core.BaseState, error) {
			current, ok := agents.GetAgent(agentID)
			if !ok {
				return state, fmt.Errorf("agent %q is no longer registered", agentID)
			}
			input := pipelineInput(state)
			execution, err := current.Execute(ctx, input)
			if err != nil {
				return state, fmt.Errorf("agent %q: %w", agentID, err)
			}
			if execution == nil {
				return state, fmt.Errorf("agent %q returned no execution", agentID)
			}
			if !execution.Success {
				message := execution.ErrorMessage
				if message == "" {
					message = "execution failed"
				}
				return state, fmt.Errorf("agent %q: %s", agentID, message)
			}
			state.Set("last_output", execution.Output)
			state.Set("agent."+nodeID+".output", execution.Output)
			state.Set("input", pipelineOutput(execution.Output))
			return state, nil
		})
		node.Metadata["type"] = "agent"
		node.Metadata["agent_id"] = pipelineNode.AgentID
		if index > 0 {
			g.AddEdge(def.Nodes[index-1].ID, pipelineNode.ID, nil)
		}
	}
	if err := g.SetStartNode(def.Nodes[0].ID); err != nil {
		return nil, err
	}
	if err := g.AddEndNode(def.Nodes[len(def.Nodes)-1].ID); err != nil {
		return nil, err
	}
	if err := g.Validate(); err != nil {
		return nil, err
	}
	return g, nil
}

func pipelineInput(state *core.BaseState) string {
	if state == nil {
		return ""
	}
	value, _ := state.Get("input")
	return pipelineOutput(value)
}

func pipelineOutput(value interface{}) string {
	if text, ok := value.(string); ok {
		return text
	}
	if value == nil {
		return ""
	}
	encoded, err := json.Marshal(value)
	if err != nil {
		return fmt.Sprint(value)
	}
	return string(encoded)
}

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
