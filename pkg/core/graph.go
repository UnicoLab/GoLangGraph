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
	"runtime/debug"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/sirupsen/logrus"
)

// Sentinel errors returned by graph execution. Callers should use errors.Is to
// classify failures rather than matching on message text.
var (
	// ErrGraphInvalid indicates the graph structure failed validation.
	ErrGraphInvalid = errors.New("graph validation failed")
	// ErrRecursionLimit indicates execution exceeded GraphConfig.MaxIterations.
	// This mirrors LangGraph's GraphRecursionError.
	ErrRecursionLimit = errors.New("recursion limit exceeded")
	// ErrInterrupted indicates execution was stopped via Interrupt.
	ErrInterrupted = errors.New("execution interrupted")
	// ErrNodePanic indicates a node function panicked. The panic is recovered
	// and converted to this error; the engine never leaves locks held.
	ErrNodePanic = errors.New("node panicked")
	// ErrNoRoute indicates no outgoing edge matched from a node.
	ErrNoRoute = errors.New("no valid next node")
	// ErrGraphClosed indicates the graph has been closed via Close.
	ErrGraphClosed = errors.New("graph is closed")
)

// NodeFunc represents a function that can be executed as a node.
//
// Returning (nil, nil) means "no state update" and mirrors LangGraph's
// behaviour when a node returns None: the incoming state is carried forward
// unchanged.
type NodeFunc func(ctx context.Context, state *BaseState) (*BaseState, error)

// EdgeCondition represents a condition function for conditional edges.
//
// For per-edge conditions (AddEdge), the function returns the target node ID to
// take that edge, or "" to decline it. For routed conditional edges
// (AddConditionalEdges), the function returns a routing key that is mapped
// through the route table.
type EdgeCondition func(ctx context.Context, state *BaseState) (string, error)

// RetryPolicy controls per-node retry behaviour.
type RetryPolicy struct {
	// MaxAttempts is the number of *additional* attempts after the first.
	MaxAttempts int `json:"max_attempts"`
	// Delay is the wait between attempts.
	Delay time.Duration `json:"delay"`
	// Backoff multiplies Delay after each failed attempt. Values <= 1 mean a
	// constant delay.
	Backoff float64 `json:"backoff"`
	// RetryIf decides whether an error is retryable. Nil means "retry all".
	RetryIf func(error) bool `json:"-"`
}

// Node represents a node in the graph
type Node struct {
	ID       string                 `json:"id"`
	Name     string                 `json:"name"`
	Function NodeFunc               `json:"-"`
	Metadata map[string]interface{} `json:"metadata"`
	// Retry, when non-nil, overrides GraphConfig retry settings for this node.
	Retry *RetryPolicy `json:"retry,omitempty"`

	// updateFn is set for nodes registered via AddUpdateNode and lets the
	// engine collect partial channel updates for reducer-based merging.
	updateFn UpdateFunc
}

// Edge represents an edge in the graph
type Edge struct {
	ID        string                 `json:"id"`
	From      string                 `json:"from"`
	To        string                 `json:"to"`
	Condition EdgeCondition          `json:"-"`
	Metadata  map[string]interface{} `json:"metadata"`
}

// ExecutionResult represents the result of node execution.
//
// Error holds the Go error and is not serialisable; ErrorMessage carries the
// same information over JSON/WebSocket so clients such as GoLangGraph Studio
// can display failures.
type ExecutionResult struct {
	NodeID       string        `json:"node_id"`
	Success      bool          `json:"success"`
	Error        error         `json:"-"`
	ErrorMessage string        `json:"error,omitempty"`
	Duration     time.Duration `json:"duration"`
	Timestamp    time.Time     `json:"timestamp"`
	State        *BaseState    `json:"state,omitempty"`
	// Step is the 0-based index of this node execution within the run.
	Step int `json:"step"`
	// Attempts is the number of attempts made (1 when the node succeeded first try).
	Attempts int `json:"attempts"`
}

// GraphConfig represents configuration for graph execution
type GraphConfig struct {
	// MaxIterations bounds the number of node executions in a single run,
	// mirroring LangGraph's recursion_limit. Exceeding it returns ErrRecursionLimit.
	MaxIterations int `json:"max_iterations"`
	// Timeout bounds total run duration. Zero means no timeout.
	Timeout           time.Duration `json:"timeout"`
	EnableStreaming   bool          `json:"enable_streaming"`
	EnableCheckpoints bool          `json:"enable_checkpoints"`
	ParallelExecution bool          `json:"parallel_execution"`
	// RetryAttempts is the number of additional attempts after the first for
	// every node. It defaults to 0: node functions frequently perform
	// non-idempotent work (LLM calls, tool side effects), so silent retries are
	// opt-in rather than the default. Set a per-node RetryPolicy for finer control.
	RetryAttempts int           `json:"retry_attempts"`
	RetryDelay    time.Duration `json:"retry_delay"`
	// InterruptBefore pauses execution before the listed nodes run.
	InterruptBefore []string `json:"interrupt_before,omitempty"`
	// InterruptAfter pauses execution after the listed nodes run.
	InterruptAfter []string `json:"interrupt_after,omitempty"`
}

// DefaultGraphConfig returns default configuration
func DefaultGraphConfig() *GraphConfig {
	return &GraphConfig{
		MaxIterations:     100,
		Timeout:           30 * time.Minute,
		EnableStreaming:   true,
		EnableCheckpoints: true,
		ParallelExecution: true,
		RetryAttempts:     0,
		RetryDelay:        1 * time.Second,
	}
}

// Clone returns a deep copy of the configuration.
func (c *GraphConfig) Clone() *GraphConfig {
	if c == nil {
		return nil
	}
	cp := *c
	cp.InterruptBefore = append([]string(nil), c.InterruptBefore...)
	cp.InterruptAfter = append([]string(nil), c.InterruptAfter...)
	return &cp
}

// InterruptError is returned when execution pauses at an interrupt point. It
// carries the state at the pause and the node that would run next, so the run
// can be resumed with Resume.
type InterruptError struct {
	NodeID   string
	Before   bool
	State    *BaseState
	Step     int
	ThreadID string
}

func (e *InterruptError) Error() string {
	when := "after"
	if e.Before {
		when = "before"
	}
	return fmt.Sprintf("execution interrupted %s node %s at step %d", when, e.NodeID, e.Step)
}

// Is lets errors.Is(err, ErrInterrupted) match interrupt pauses.
func (e *InterruptError) Is(target error) bool { return target == ErrInterrupted }

// StateSaver is the minimal checkpointing hook the engine needs. The
// persistence package provides an adapter implementing it, which keeps core
// free of a dependency on any storage backend.
type StateSaver interface {
	SaveState(ctx context.Context, threadID, nodeID string, step int, state *BaseState) error
}

// runHandle tracks a single in-flight execution so Interrupt can signal every
// active run without racing on shared fields.
type runHandle struct {
	interrupt chan struct{}
	once      sync.Once
}

func (h *runHandle) signal() { h.once.Do(func() { close(h.interrupt) }) }

// Graph represents the execution graph.
//
// A Graph is safe for concurrent use: Execute keeps all mutable run state in a
// per-invocation structure, so multiple goroutines may execute the same graph
// simultaneously without interfering with each other.
type Graph struct {
	ID        string                 `json:"id"`
	Name      string                 `json:"name"`
	Nodes     map[string]*Node       `json:"nodes"`
	Edges     map[string]*Edge       `json:"edges"`
	StartNode string                 `json:"start_node"`
	EndNodes  []string               `json:"end_nodes"`
	Config    *GraphConfig           `json:"config"`
	Metadata  map[string]interface{} `json:"metadata"`

	// Deterministic ordering: Go map iteration is randomised, so edge and node
	// order is tracked explicitly to make routing reproducible.
	nodeOrder []string
	edgeOrder []string

	// Conditional edges are held in a typed field rather than Metadata so that
	// JSON round-tripping the graph cannot corrupt them.
	condEdges map[string]*ConditionalEdge

	// Errors accumulated by builder methods that cannot return one; surfaced by Validate.
	buildErrors []error

	// Observability mirrors of the most recent execution.
	currentState     *BaseState
	executionHistory []*ExecutionResult
	running          int

	mu sync.RWMutex

	// Streaming and interrupts
	streamChan chan *ExecutionResult
	closed     bool
	closeOnce  sync.Once
	active     map[*runHandle]struct{}

	// Optional state schema providing channel reducers.
	schema *StateSchema

	// Nested graphs registered via AddSubgraph, keyed by node ID.
	subgraphs map[string]*Graph

	// Optional checkpointing
	saver    StateSaver
	threadID string

	logger *logrus.Logger
}

// NewGraph creates a new graph
func NewGraph(name string) *Graph {
	return &Graph{
		ID:               uuid.New().String(),
		Name:             name,
		Nodes:            make(map[string]*Node),
		Edges:            make(map[string]*Edge),
		EndNodes:         make([]string, 0),
		Config:           DefaultGraphConfig(),
		Metadata:         make(map[string]interface{}),
		condEdges:        make(map[string]*ConditionalEdge),
		executionHistory: make([]*ExecutionResult, 0),
		streamChan:       make(chan *ExecutionResult, 100),
		active:           make(map[*runHandle]struct{}),
		logger:           logrus.New(),
	}
}

// SetLogger replaces the graph logger. A nil logger is ignored.
func (g *Graph) SetLogger(l *logrus.Logger) {
	if l == nil {
		return
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	g.logger = l
}

// WithCheckpointer attaches a state saver and thread ID used to persist state
// after every node execution, enabling durable execution and resume.
func (g *Graph) WithCheckpointer(saver StateSaver, threadID string) *Graph {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.saver = saver
	g.threadID = threadID
	return g
}

// AddNode adds a node to the graph. Adding a node with an existing ID or an
// empty ID records a build error surfaced by Validate.
func (g *Graph) AddNode(id, name string, fn NodeFunc) *Node {
	g.mu.Lock()
	defer g.mu.Unlock()

	if id == "" {
		g.buildErrors = append(g.buildErrors, errors.New("node ID must not be empty"))
	}
	if id == START || id == END {
		g.buildErrors = append(g.buildErrors, fmt.Errorf("node ID %q is reserved", id))
	}
	if fn == nil {
		g.buildErrors = append(g.buildErrors, fmt.Errorf("node %s has a nil function", id))
	}
	if _, exists := g.Nodes[id]; exists {
		g.buildErrors = append(g.buildErrors, fmt.Errorf("node %s already exists", id))
	} else {
		g.nodeOrder = append(g.nodeOrder, id)
	}

	node := &Node{
		ID:       id,
		Name:     name,
		Function: fn,
		Metadata: make(map[string]interface{}),
	}

	g.Nodes[id] = node
	return node
}

// AddEdge adds an edge to the graph. Edges are followed in insertion order,
// which makes routing deterministic.
func (g *Graph) AddEdge(from, to string, condition EdgeCondition) *Edge {
	g.mu.Lock()
	defer g.mu.Unlock()

	if from == "" || to == "" {
		g.buildErrors = append(g.buildErrors, fmt.Errorf("edge endpoints must not be empty (from=%q to=%q)", from, to))
	}

	edge := &Edge{
		ID:        uuid.New().String(),
		From:      from,
		To:        to,
		Condition: condition,
		Metadata:  make(map[string]interface{}),
	}

	g.Edges[edge.ID] = edge
	g.edgeOrder = append(g.edgeOrder, edge.ID)
	return edge
}

// SetStartNode sets the starting node for execution
func (g *Graph) SetStartNode(nodeID string) error {
	g.mu.Lock()
	defer g.mu.Unlock()

	if _, exists := g.Nodes[nodeID]; !exists {
		return fmt.Errorf("node %s does not exist", nodeID)
	}

	g.StartNode = nodeID
	return nil
}

// AddEndNode adds an end node to the graph
func (g *Graph) AddEndNode(nodeID string) error {
	g.mu.Lock()
	defer g.mu.Unlock()

	if _, exists := g.Nodes[nodeID]; !exists {
		return fmt.Errorf("node %s does not exist", nodeID)
	}
	for _, existing := range g.EndNodes {
		if existing == nodeID {
			return nil
		}
	}

	g.EndNodes = append(g.EndNodes, nodeID)
	return nil
}

// Validate validates the graph structure
func (g *Graph) Validate() error {
	g.mu.RLock()
	defer g.mu.RUnlock()

	if len(g.buildErrors) > 0 {
		return fmt.Errorf("%w: %w", ErrGraphInvalid, errors.Join(g.buildErrors...))
	}

	if g.StartNode == "" {
		return fmt.Errorf("%w: start node is not set", ErrGraphInvalid)
	}

	if _, exists := g.Nodes[g.StartNode]; !exists {
		return fmt.Errorf("%w: start node %s does not exist", ErrGraphInvalid, g.StartNode)
	}

	for _, endNode := range g.EndNodes {
		if _, exists := g.Nodes[endNode]; !exists {
			return fmt.Errorf("%w: end node %s does not exist", ErrGraphInvalid, endNode)
		}
	}

	for _, id := range g.edgeOrder {
		edge := g.Edges[id]
		if edge == nil {
			continue
		}
		if _, exists := g.Nodes[edge.From]; !exists {
			return fmt.Errorf("%w: edge %s references non-existent from node %s", ErrGraphInvalid, edge.ID, edge.From)
		}
		if !g.isTerminal(edge.To) {
			if _, exists := g.Nodes[edge.To]; !exists {
				return fmt.Errorf("%w: edge %s references non-existent to node %s", ErrGraphInvalid, edge.ID, edge.To)
			}
		}
	}

	for from, ce := range g.condEdges {
		if _, exists := g.Nodes[from]; !exists {
			return fmt.Errorf("%w: conditional edge references non-existent source node %s", ErrGraphInvalid, from)
		}
		if ce.Condition == nil {
			return fmt.Errorf("%w: conditional edge from %s has a nil condition", ErrGraphInvalid, from)
		}
		for key, to := range ce.Routes {
			if g.isTerminal(to) {
				continue
			}
			if _, exists := g.Nodes[to]; !exists {
				return fmt.Errorf("%w: conditional route %s->%s (key %q) targets non-existent node", ErrGraphInvalid, from, to, key)
			}
		}
	}

	return nil
}

// isTerminal reports whether a target refers to graph termination. Callers must
// hold at least a read lock.
func (g *Graph) isTerminal(target string) bool {
	return target == END || target == ""
}

// ExecuteOptions customises a single run.
type ExecuteOptions struct {
	// ThreadID scopes checkpoints for this run. Empty uses the graph default.
	ThreadID string
	// StartNode overrides the entry point (used by Resume).
	StartNode string
	// Stream, when non-nil, receives per-step results for this run only.
	// It is closed when the run finishes.
	Stream chan<- *ExecutionResult
	// ResumeStep sets the starting step counter (used by Resume).
	ResumeStep int
}

// Execute executes the graph with the given initial state.
//
// Execute is safe for concurrent use; each call carries its own state and
// history. On failure it returns the last known good state alongside the error
// so callers can inspect partial progress.
func (g *Graph) Execute(ctx context.Context, initialState *BaseState) (*BaseState, error) {
	return g.ExecuteWithOptions(ctx, initialState, nil)
}

// Resume continues a run from a previously interrupted point.
func (g *Graph) Resume(ctx context.Context, ie *InterruptError) (*BaseState, error) {
	if ie == nil {
		return nil, errors.New("resume requires a non-nil interrupt")
	}
	start := ie.NodeID
	step := ie.Step
	if !ie.Before {
		// The node already ran; continue from the node that follows it.
		next, err := g.routeFrom(ctx, ie.NodeID, ie.State)
		if err != nil {
			return ie.State, err
		}
		if next == "" || next == END {
			return ie.State, nil
		}
		start = next
		step = ie.Step + 1
	}
	return g.ExecuteWithOptions(ctx, ie.State, &ExecuteOptions{
		ThreadID:   ie.ThreadID,
		StartNode:  start,
		ResumeStep: step,
	})
}

// ExecuteWithOptions runs the graph with per-run options.
func (g *Graph) ExecuteWithOptions(ctx context.Context, initialState *BaseState, opts *ExecuteOptions) (*BaseState, error) {
	if opts == nil {
		opts = &ExecuteOptions{}
	}
	if err := g.Validate(); err != nil {
		return nil, err
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if initialState == nil {
		initialState = NewBaseState()
	}

	g.mu.RLock()
	closed := g.closed
	cfg := g.Config.Clone()
	logger := g.logger
	saver := g.saver
	threadID := g.threadID
	startNode := g.StartNode
	endNodes := make(map[string]struct{}, len(g.EndNodes))
	for _, id := range g.EndNodes {
		endNodes[id] = struct{}{}
	}
	g.mu.RUnlock()

	if closed {
		return nil, ErrGraphClosed
	}
	if cfg == nil {
		cfg = DefaultGraphConfig()
	}
	if opts.ThreadID != "" {
		threadID = opts.ThreadID
	}
	if opts.StartNode != "" {
		startNode = opts.StartNode
	}
	interruptBefore := toSet(cfg.InterruptBefore)
	interruptAfter := toSet(cfg.InterruptAfter)

	handle := &runHandle{interrupt: make(chan struct{})}
	g.mu.Lock()
	g.active[handle] = struct{}{}
	g.running++
	g.currentState = initialState.Clone()
	g.executionHistory = make([]*ExecutionResult, 0)
	g.mu.Unlock()

	defer func() {
		g.mu.Lock()
		delete(g.active, handle)
		if g.running > 0 {
			g.running--
		}
		g.mu.Unlock()
		if opts.Stream != nil {
			close(opts.Stream)
		}
	}()

	execCtx := ctx
	if cfg.Timeout > 0 {
		var cancel context.CancelFunc
		execCtx, cancel = context.WithTimeout(ctx, cfg.Timeout)
		defer cancel()
	}

	state := initialState.Clone()
	currentNode := startNode
	step := opts.ResumeStep

	for {
		select {
		case <-execCtx.Done():
			return state, fmt.Errorf("graph execution stopped: %w", contextError(execCtx))
		case <-handle.interrupt:
			return state, ErrInterrupted
		default:
		}

		if cfg.MaxIterations > 0 && step >= cfg.MaxIterations {
			return state, fmt.Errorf("%w: %d node executions without reaching an end node", ErrRecursionLimit, cfg.MaxIterations)
		}

		if _, pause := interruptBefore[currentNode]; pause {
			return state, &InterruptError{NodeID: currentNode, Before: true, State: state, Step: step, ThreadID: threadID}
		}

		result, err := g.executeNodeStep(execCtx, currentNode, state, cfg, logger, step)

		// Record the step (success or failure) so failures are observable.
		g.recordResult(result, state)
		g.emit(cfg, opts.Stream, result)

		if err != nil {
			return state, fmt.Errorf("node %s failed: %w", currentNode, err)
		}

		if result.State != nil {
			state = result.State
		}
		g.mu.Lock()
		g.currentState = state.Clone()
		g.mu.Unlock()

		if saver != nil && cfg.EnableCheckpoints && threadID != "" {
			if serr := saver.SaveState(execCtx, threadID, currentNode, step, state); serr != nil {
				logger.WithError(serr).WithField("node_id", currentNode).Warn("checkpoint save failed")
			}
		}

		step++

		if _, pause := interruptAfter[currentNode]; pause {
			return state, &InterruptError{NodeID: currentNode, Before: false, State: state, Step: step - 1, ThreadID: threadID}
		}

		if _, isEnd := endNodes[currentNode]; isEnd {
			break
		}

		nextNode, err := g.routeFrom(execCtx, currentNode, state)
		if err != nil {
			return state, err
		}
		if nextNode == "" || nextNode == END {
			break
		}
		currentNode = nextNode
	}

	return state, nil
}

// joinErrors collapses a slice of errors into one, or nil when all are nil.
func joinErrors(errs []error) error {
	return errors.Join(errs...)
}

func toSet(items []string) map[string]struct{} {
	set := make(map[string]struct{}, len(items))
	for _, i := range items {
		set[i] = struct{}{}
	}
	return set
}

// contextError returns the most specific cause available for a done context.
func contextError(ctx context.Context) error {
	if cause := context.Cause(ctx); cause != nil {
		return cause
	}
	return ctx.Err()
}

// recordResult appends to the observability mirror of the last run.
func (g *Graph) recordResult(result *ExecutionResult, fallback *BaseState) {
	if result == nil {
		return
	}
	g.mu.Lock()
	g.executionHistory = append(g.executionHistory, result)
	g.mu.Unlock()
}

// emit publishes a step result to the graph-wide stream and the per-run stream.
// It never sends on a closed channel and never blocks.
func (g *Graph) emit(cfg *GraphConfig, runStream chan<- *ExecutionResult, result *ExecutionResult) {
	if result == nil {
		return
	}
	if runStream != nil {
		select {
		case runStream <- result:
		default:
		}
	}
	if !cfg.EnableStreaming {
		return
	}
	g.mu.RLock()
	defer g.mu.RUnlock()
	if g.closed {
		return
	}
	select {
	case g.streamChan <- result:
	default:
		// Consumer is slow; drop rather than stall graph execution.
	}
}

// executeNodeStep runs one node with retry and panic protection. No graph lock
// is held while user code runs.
func (g *Graph) executeNodeStep(ctx context.Context, nodeID string, state *BaseState, cfg *GraphConfig, logger *logrus.Logger, step int) (*ExecutionResult, error) {
	g.mu.RLock()
	node, exists := g.Nodes[nodeID]
	g.mu.RUnlock()

	if !exists {
		err := fmt.Errorf("node %s does not exist", nodeID)
		return &ExecutionResult{NodeID: nodeID, Success: false, Error: err, ErrorMessage: err.Error(), Timestamp: time.Now(), Step: step}, err
	}

	policy := node.Retry
	if policy == nil {
		policy = &RetryPolicy{MaxAttempts: cfg.RetryAttempts, Delay: cfg.RetryDelay}
	}

	logger.WithFields(logrus.Fields{
		"node_id": nodeID, "node_name": node.Name, "graph_id": g.ID, "step": step,
	}).Debug("Executing node")

	start := time.Now()
	delay := policy.Delay
	var resultState *BaseState
	var err error
	attempts := 0

	for attempt := 0; ; attempt++ {
		attempts = attempt + 1
		// Each attempt starts from a pristine copy so a partially-mutated state
		// from a failed attempt cannot leak into the retry.
		resultState, err = callNode(ctx, node, state.Clone())
		if err == nil {
			break
		}
		if attempt >= policy.MaxAttempts {
			break
		}
		if policy.RetryIf != nil && !policy.RetryIf(err) {
			break
		}
		logger.WithFields(logrus.Fields{"node_id": nodeID, "attempt": attempts, "error": err}).Warn("Node execution failed, retrying")
		if delay > 0 {
			select {
			case <-ctx.Done():
				cerr := contextError(ctx)
				return &ExecutionResult{NodeID: nodeID, Success: false, Error: cerr, ErrorMessage: cerr.Error(), Duration: time.Since(start), Timestamp: time.Now(), Step: step, Attempts: attempts}, cerr
			case <-time.After(delay):
			}
		}
		if policy.Backoff > 1 {
			delay = time.Duration(float64(delay) * policy.Backoff)
		}
	}

	duration := time.Since(start)
	result := &ExecutionResult{
		NodeID:    nodeID,
		Success:   err == nil,
		Error:     err,
		Duration:  duration,
		Timestamp: time.Now(),
		State:     resultState,
		Step:      step,
		Attempts:  attempts,
	}
	if err != nil {
		result.ErrorMessage = err.Error()
		result.State = nil
		fields := logrus.Fields{"node_id": nodeID, "duration": duration, "error": err}
		var pe *PanicError
		if errors.As(err, &pe) {
			fields["stack"] = string(pe.Stack)
		}
		logger.WithFields(fields).Error("Node execution failed")
	}
	return result, err
}

// PanicError reports a recovered panic from user code. The stack is kept in a
// separate field rather than in the message so that error strings surfaced to
// API clients do not leak internal paths and goroutine dumps.
type PanicError struct {
	// Where identifies the node or condition that panicked.
	Where string
	// Value is the recovered panic value.
	Value interface{}
	// Stack is the goroutine stack captured at recovery time, for logs only.
	Stack []byte
}

func (e *PanicError) Error() string {
	return fmt.Sprintf("%s: %s: %v", ErrNodePanic.Error(), e.Where, e.Value)
}

// Unwrap lets errors.Is(err, ErrNodePanic) succeed.
func (e *PanicError) Unwrap() error { return ErrNodePanic }

// callNode invokes a node function, converting panics into errors so a faulty
// node can never crash the process or wedge the engine.
func callNode(ctx context.Context, node *Node, state *BaseState) (out *BaseState, err error) {
	defer func() {
		if r := recover(); r != nil {
			out = nil
			err = &PanicError{Where: "node " + node.ID, Value: r, Stack: debug.Stack()}
		}
	}()
	return node.Function(ctx, state)
}

// callCondition invokes an edge condition with panic protection.
func callCondition(ctx context.Context, from string, fn EdgeCondition, state *BaseState) (out string, err error) {
	defer func() {
		if r := recover(); r != nil {
			out = ""
			err = &PanicError{Where: "condition on " + from, Value: r, Stack: debug.Stack()}
		}
	}()
	return fn(ctx, state)
}

// routeFrom determines the next node, honouring routed conditional edges first
// (LangGraph's add_conditional_edges) and then per-edge conditions in insertion
// order. User code is called without holding the graph lock.
func (g *Graph) routeFrom(ctx context.Context, currentNodeID string, state *BaseState) (string, error) {
	g.mu.RLock()
	condEdge := g.condEdges[currentNodeID]
	outgoing := make([]*Edge, 0, 4)
	for _, id := range g.edgeOrder {
		if e := g.Edges[id]; e != nil && e.From == currentNodeID {
			outgoing = append(outgoing, e)
		}
	}
	g.mu.RUnlock()

	// Routed conditional edges take precedence and are evaluated exactly once.
	if condEdge != nil && condEdge.Condition != nil {
		key, err := callCondition(ctx, currentNodeID, condEdge.Condition, state.Clone())
		if err != nil {
			return "", fmt.Errorf("conditional edge from %s failed: %w", currentNodeID, err)
		}
		if target, ok := condEdge.Routes[key]; ok {
			return normalizeTarget(target), nil
		}
		if len(condEdge.Routes) == 0 {
			// No route table: the condition returns the destination directly.
			return normalizeTarget(key), nil
		}
		if key == END || key == "" {
			return END, nil
		}
		return "", fmt.Errorf("%w: conditional edge from %s returned key %q which is not in its route table", ErrNoRoute, currentNodeID, key)
	}

	if len(outgoing) == 0 {
		return "", nil
	}

	if len(outgoing) == 1 && outgoing[0].Condition == nil {
		return normalizeTarget(outgoing[0].To), nil
	}

	// Per-edge conditions, evaluated in insertion order for determinism.
	for _, edge := range outgoing {
		if edge.Condition == nil {
			continue
		}
		target, err := callCondition(ctx, currentNodeID, edge.Condition, state.Clone())
		if err != nil {
			return "", fmt.Errorf("edge condition evaluation failed: %w", err)
		}
		if target == edge.To {
			return normalizeTarget(edge.To), nil
		}
	}

	for _, edge := range outgoing {
		if edge.Condition == nil {
			return normalizeTarget(edge.To), nil
		}
	}

	return "", fmt.Errorf("%w from %s", ErrNoRoute, currentNodeID)
}

func normalizeTarget(target string) string {
	if target == END {
		return END
	}
	return target
}

// Stream returns a channel for streaming execution results from any run.
// Results are dropped rather than blocking execution if the consumer is slow;
// use ExecuteWithOptions with a per-run Stream for lossless streaming.
func (g *Graph) Stream() <-chan *ExecutionResult {
	return g.streamChan
}

// Interrupt interrupts all in-flight executions. It is safe to call at any
// time, including after Close and when nothing is running.
func (g *Graph) Interrupt() {
	g.mu.RLock()
	handles := make([]*runHandle, 0, len(g.active))
	for h := range g.active {
		handles = append(handles, h)
	}
	g.mu.RUnlock()

	for _, h := range handles {
		h.signal()
	}
}

// IsRunning returns whether the graph is currently executing
func (g *Graph) IsRunning() bool {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return g.running > 0
}

// GetExecutionHistory returns the execution history of the most recent run.
func (g *Graph) GetExecutionHistory() []*ExecutionResult {
	g.mu.RLock()
	defer g.mu.RUnlock()

	history := make([]*ExecutionResult, len(g.executionHistory))
	copy(history, g.executionHistory)
	return history
}

// GetCurrentState returns the state of the most recent run.
func (g *Graph) GetCurrentState() *BaseState {
	g.mu.RLock()
	defer g.mu.RUnlock()

	if g.currentState == nil {
		return nil
	}
	return g.currentState.Clone()
}

// Reset resets the graph observability state.
func (g *Graph) Reset() {
	g.mu.Lock()
	defer g.mu.Unlock()

	g.currentState = nil
	g.executionHistory = make([]*ExecutionResult, 0)
}

// ExecuteParallel executes multiple nodes concurrently against a shared input
// state (a LangGraph super-step). Each node receives its own copy of the state;
// merge the results with MergeResults or a StateSchema reducer.
func (g *Graph) ExecuteParallel(ctx context.Context, nodeIDs []string, state *BaseState) (map[string]*ExecutionResult, error) {
	if len(nodeIDs) == 0 {
		return make(map[string]*ExecutionResult), nil
	}
	if state == nil {
		state = NewBaseState()
	}

	g.mu.RLock()
	cfg := g.Config.Clone()
	logger := g.logger
	g.mu.RUnlock()
	if cfg == nil {
		cfg = DefaultGraphConfig()
	}

	results := make(map[string]*ExecutionResult, len(nodeIDs))
	var resultsMu sync.Mutex
	var wg sync.WaitGroup
	errs := make([]error, len(nodeIDs))

	for i, nodeID := range nodeIDs {
		wg.Add(1)
		go func(idx int, nID string) {
			defer wg.Done()
			result, err := g.executeNodeStep(ctx, nID, state, cfg, logger, idx)
			resultsMu.Lock()
			if result != nil {
				results[nID] = result
			}
			resultsMu.Unlock()
			if err != nil {
				errs[idx] = fmt.Errorf("node %s failed: %w", nID, err)
			}
		}(i, nodeID)
	}

	wg.Wait()

	// Always return the partial results so callers can see which branches
	// succeeded even when one fails.
	if joined := errors.Join(errs...); joined != nil {
		return results, joined
	}
	return results, nil
}

// GetNodesByType returns nodes filtered by metadata type
func (g *Graph) GetNodesByType(nodeType string) []*Node {
	g.mu.RLock()
	defer g.mu.RUnlock()

	var nodes []*Node
	for _, id := range g.nodeOrder {
		node := g.Nodes[id]
		if node == nil {
			continue
		}
		if nodeTypeValue, exists := node.Metadata["type"]; exists {
			if nodeTypeValue == nodeType {
				nodes = append(nodes, node)
			}
		}
	}
	return nodes
}

// GetTopology returns the graph topology as adjacency list, including
// conditional routes so visualisers and Studio see the full reachable graph.
func (g *Graph) GetTopology() map[string][]string {
	g.mu.RLock()
	defer g.mu.RUnlock()

	topology := make(map[string][]string)
	for nodeID := range g.Nodes {
		topology[nodeID] = make([]string, 0)
	}

	for _, id := range g.edgeOrder {
		edge := g.Edges[id]
		if edge == nil {
			continue
		}
		topology[edge.From] = append(topology[edge.From], edge.To)
	}

	for _, from := range g.nodeOrder {
		ce := g.condEdges[from]
		if ce == nil {
			continue
		}
		topology[from] = append(topology[from], sortedValues(ce.Routes)...)
	}

	return topology
}

// Close closes the graph and cleans up resources. It is idempotent and safe to
// call concurrently with execution: in-flight runs are interrupted first.
func (g *Graph) Close() {
	g.Interrupt()

	g.closeOnce.Do(func() {
		g.mu.Lock()
		g.closed = true
		close(g.streamChan)
		g.mu.Unlock()
	})
}
