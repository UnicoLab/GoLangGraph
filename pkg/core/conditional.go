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

// ConditionalEdge represents a conditional edge that can route to different nodes
type ConditionalEdge struct {
	ID        string                 `json:"id"`
	From      string                 `json:"from"`
	Condition EdgeCondition          `json:"-"`
	Routes    map[string]string      `json:"routes"` // condition result -> target node
	Metadata  map[string]interface{} `json:"metadata"`
}

// RouterFunction represents a function that determines the next node based on state
type RouterFunction func(ctx context.Context, state *BaseState) (string, error)

// ConditionalRouter manages conditional routing logic
type ConditionalRouter struct {
	mu       sync.RWMutex
	routes   map[string]RouterFunction
	order    []string
	fallback string
}

// NewConditionalRouter creates a new conditional router
func NewConditionalRouter(fallback string) *ConditionalRouter {
	return &ConditionalRouter{
		routes:   make(map[string]RouterFunction),
		fallback: fallback,
	}
}

// AddRoute adds a route with a condition. Routes are evaluated in the order
// they were added, which keeps routing deterministic.
func (cr *ConditionalRouter) AddRoute(condition string, router RouterFunction) {
	cr.mu.Lock()
	defer cr.mu.Unlock()
	if _, exists := cr.routes[condition]; !exists {
		cr.order = append(cr.order, condition)
	}
	cr.routes[condition] = router
}

// Route determines the next node based on state. Conditions are evaluated in
// insertion order; the first non-empty result wins. A condition that returns an
// error is skipped, and the fallback is used when nothing matches.
func (cr *ConditionalRouter) Route(ctx context.Context, state *BaseState) (string, error) {
	cr.mu.RLock()
	order := append([]string(nil), cr.order...)
	routes := make(map[string]RouterFunction, len(cr.routes))
	for k, v := range cr.routes {
		routes[k] = v
	}
	fallback := cr.fallback
	cr.mu.RUnlock()

	for _, key := range order {
		router := routes[key]
		if router == nil {
			continue
		}
		result, err := router(ctx, state)
		if err != nil {
			continue // Try next condition
		}
		if result != "" {
			return result, nil
		}
	}

	// Return fallback if no conditions match
	return fallback, nil
}

// AddConditionalEdges adds conditional edges to the graph, mirroring
// LangGraph's add_conditional_edges: a single path function is evaluated once
// per visit and its result is mapped through the route table.
//
// An empty routes map means the condition returns the destination node ID
// directly. END is accepted as a destination.
func (g *Graph) AddConditionalEdges(from string, condition EdgeCondition, routes map[string]string) error {
	g.mu.Lock()
	defer g.mu.Unlock()

	if condition == nil {
		return fmt.Errorf("conditional edge from %s requires a condition function", from)
	}

	// Verify source node exists
	if _, exists := g.Nodes[from]; !exists {
		return fmt.Errorf("source node %s does not exist", from)
	}

	// Verify target nodes exist
	for _, to := range routes {
		if to != END && to != START {
			if _, exists := g.Nodes[to]; !exists {
				return fmt.Errorf("target node %s does not exist", to)
			}
		}
	}

	if _, exists := g.condEdges[from]; exists {
		return fmt.Errorf("conditional edges already defined for node %s", from)
	}

	copied := make(map[string]string, len(routes))
	for k, v := range routes {
		copied[k] = v
	}

	edge := &ConditionalEdge{
		ID:        fmt.Sprintf("conditional_%s", from),
		From:      from,
		Condition: condition,
		Routes:    copied,
		Metadata:  make(map[string]interface{}),
	}

	if g.condEdges == nil {
		g.condEdges = make(map[string]*ConditionalEdge)
	}
	g.condEdges[from] = edge

	return nil
}

// GetConditionalEdge retrieves a conditional edge for a node
func (g *Graph) GetConditionalEdge(nodeID string) (*ConditionalEdge, bool) {
	g.mu.RLock()
	defer g.mu.RUnlock()

	edge, exists := g.condEdges[nodeID]
	return edge, exists
}

// sortedValues returns a map's values ordered by key, for deterministic output.
func sortedValues(m map[string]string) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	values := make([]string, 0, len(keys))
	for _, k := range keys {
		values = append(values, m[k])
	}
	return values
}

// Common routing functions

// RouteByMessageType routes based on the type of the last message
func RouteByMessageType(ctx context.Context, state *BaseState) (string, error) {
	messages, exists := state.Get("messages")
	if !exists {
		return "", fmt.Errorf("no messages in state")
	}

	messageList, ok := messages.([]interface{})
	if !ok || len(messageList) == 0 {
		return "", fmt.Errorf("invalid or empty message list")
	}

	lastMessage := messageList[len(messageList)-1]
	messageMap, ok := lastMessage.(map[string]interface{})
	if !ok {
		return "", fmt.Errorf("invalid message format")
	}

	messageType, exists := messageMap["type"]
	if !exists {
		return "", fmt.Errorf("message has no type")
	}

	switch messageType {
	case "human":
		return "process_human_message", nil
	case "ai":
		return "process_ai_message", nil
	case "tool":
		return "process_tool_message", nil
	default:
		return "default_processor", nil
	}
}

// RouteByToolCalls routes based on whether tool calls are present
func RouteByToolCalls(ctx context.Context, state *BaseState) (string, error) {
	toolCalls, exists := state.Get("tool_calls")
	if !exists {
		return "no_tools", nil
	}

	toolCallList, ok := toolCalls.([]interface{})
	if !ok {
		return "no_tools", nil
	}

	if len(toolCallList) > 0 {
		return "execute_tools", nil
	}

	return "no_tools", nil
}

// RouteByCondition routes based on a boolean condition in state
func RouteByCondition(conditionKey string, trueRoute string, falseRoute string) RouterFunction {
	return func(ctx context.Context, state *BaseState) (string, error) {
		condition, exists := state.Get(conditionKey)
		if !exists {
			return falseRoute, nil
		}

		conditionBool, ok := condition.(bool)
		if !ok {
			return falseRoute, nil
		}

		if conditionBool {
			return trueRoute, nil
		}

		return falseRoute, nil
	}
}

// RouteByCounter routes based on a counter value
func RouteByCounter(counterKey string, maxCount int, continueRoute string, exitRoute string) RouterFunction {
	return func(ctx context.Context, state *BaseState) (string, error) {
		counter, exists := state.Get(counterKey)
		if !exists {
			return continueRoute, nil
		}

		counterInt, ok := counter.(int)
		if !ok {
			return continueRoute, nil
		}

		if counterInt >= maxCount {
			return exitRoute, nil
		}

		return continueRoute, nil
	}
}

// RouteByStateValue routes based on a specific state value
func RouteByStateValue(key string, routes map[interface{}]string, defaultRoute string) RouterFunction {
	return func(ctx context.Context, state *BaseState) (string, error) {
		value, exists := state.Get(key)
		if !exists {
			return defaultRoute, nil
		}

		if route, exists := routes[value]; exists {
			return route, nil
		}

		return defaultRoute, nil
	}
}

// START and END constants for graph flow control
const (
	START = "__start__"
	END   = "__end__"
)

// IsStartNode checks if a node is the start node
func (g *Graph) IsStartNode(nodeID string) bool {
	if nodeID == START {
		return true
	}
	g.mu.RLock()
	defer g.mu.RUnlock()
	return nodeID == g.StartNode
}

// IsEndNode checks if a node is an end node
func (g *Graph) IsEndNode(nodeID string) bool {
	if nodeID == END {
		return true
	}

	g.mu.RLock()
	defer g.mu.RUnlock()
	for _, endNode := range g.EndNodes {
		if endNode == nodeID {
			return true
		}
	}

	return false
}

// GetNextNodes determines the next nodes to execute based on current node and
// state. It delegates to the same routing logic the engine uses, so callers and
// the executor can never disagree about where a node leads.
func (g *Graph) GetNextNodes(ctx context.Context, currentNodeID string, state *BaseState) ([]string, error) {
	if state == nil {
		state = NewBaseState()
	}
	next, err := g.routeFrom(ctx, currentNodeID, state)
	if err != nil {
		return nil, err
	}
	if next == "" {
		return nil, nil
	}
	return []string{next}, nil
}
