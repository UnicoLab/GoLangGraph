// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package agent

import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/mux"
	"github.com/sirupsen/logrus"
	yaml "gopkg.in/yaml.v3"

	"github.com/UnicoLab/GoLangGraph/pkg/llm"
	"github.com/UnicoLab/GoLangGraph/pkg/tools"
)

// MaxRequestBodyBytes caps how much of a request body the agent handlers will
// read. Without a cap a single client could stream an unbounded body into the
// JSON decoder and exhaust the process's memory.
const MaxRequestBodyBytes = 1 << 20 // 1 MiB

// DefaultAgentExecutionTimeout is used when an agent config sets no timeout.
const DefaultAgentExecutionTimeout = 5 * time.Minute

// MultiAgentManager manages multiple agents with routing and deployment capabilities
type MultiAgentManager struct {
	config          *MultiAgentConfig
	agents          map[string]Agent // Changed from map[string]Agent to map[string]Agent
	llmManager      *llm.ProviderManager
	toolRegistry    *tools.ToolRegistry
	router          *mux.Router
	middleware      []MiddlewareFunc
	deploymentState *DeploymentState
	logger          *logrus.Logger
	mu              sync.RWMutex

	// Health monitoring
	healthCheckers map[string]*HealthChecker
	healthMu       sync.RWMutex

	// Health checker lifecycle. The checkers used to be started from the
	// constructor as bare goroutines with no cancellation and an endless
	// "for range ticker.C" loop, so every manager ever built leaked one
	// goroutine per agent for the lifetime of the process and Stop had no way
	// to reclaim them.
	lifecycleMu   sync.Mutex
	healthCancel  context.CancelFunc
	healthWG      sync.WaitGroup
	healthRunning bool

	// Metrics and monitoring
	metrics *MultiAgentMetrics

	// Request limiting
	limiter *rateLimiter
}

// MiddlewareFunc defines middleware function signature
type MiddlewareFunc func(next http.Handler) http.Handler

// DeploymentState tracks the deployment state of agents
type DeploymentState struct {
	Status      string                 `json:"status"`
	StartedAt   time.Time              `json:"started_at"`
	UpdatedAt   time.Time              `json:"updated_at"`
	AgentStates map[string]*AgentState `json:"agent_states"`
	ErrorCount  int                    `json:"error_count"`
	LastError   string                 `json:"last_error"`
	Metadata    map[string]interface{} `json:"metadata"`
}

// Clone returns a deep copy of the deployment state.
//
// GetDeploymentState and the /deployment/status handler used to dereference the
// struct while holding the read lock and hand the result out - but the copy
// shares the AgentStates map and every *AgentState in it, so callers (and the
// JSON encoder, after the lock was released) read fields that request handlers
// were concurrently writing. The race detector flags it on any concurrent load.
func (ds *DeploymentState) Clone() *DeploymentState {
	if ds == nil {
		return nil
	}
	clone := *ds
	clone.AgentStates = make(map[string]*AgentState, len(ds.AgentStates))
	for id, state := range ds.AgentStates {
		clone.AgentStates[id] = state.Clone()
	}
	clone.Metadata = make(map[string]interface{}, len(ds.Metadata))
	for k, v := range ds.Metadata {
		clone.Metadata[k] = v
	}
	return &clone
}

// AgentState tracks the state of individual agents
type AgentState struct {
	ID           string                 `json:"id"`
	Status       string                 `json:"status"` // "starting", "running", "stopping", "stopped", "error"
	StartedAt    time.Time              `json:"started_at"`
	UpdatedAt    time.Time              `json:"updated_at"`
	RequestCount int64                  `json:"request_count"`
	ErrorCount   int64                  `json:"error_count"`
	LastRequest  time.Time              `json:"last_request"`
	LastError    string                 `json:"last_error"`
	HealthStatus string                 `json:"health_status"`
	Metadata     map[string]interface{} `json:"metadata"`
}

// Clone returns a deep copy of the agent state.
func (as *AgentState) Clone() *AgentState {
	if as == nil {
		return nil
	}
	clone := *as
	clone.Metadata = make(map[string]interface{}, len(as.Metadata))
	for k, v := range as.Metadata {
		clone.Metadata[k] = v
	}
	return &clone
}

// HealthChecker performs health checks for agents.
//
// The mutable fields are guarded by mu: they are written by the checker's own
// goroutine and read by callers of Snapshot (and by the /health handlers), so
// leaving them bare was a data race waiting for the first reader.
type HealthChecker struct {
	AgentID string
	Config  *HealthCheckConfig
	Logger  *logrus.Logger

	mu               sync.RWMutex
	lastCheck        time.Time
	status           string
	consecutiveFails int
	lastError        string
}

// HealthCheckerSnapshot is an immutable view of a checker's state.
type HealthCheckerSnapshot struct {
	AgentID          string    `json:"agent_id"`
	Status           string    `json:"status"`
	LastCheck        time.Time `json:"last_check"`
	ConsecutiveFails int       `json:"consecutive_fails"`
	LastError        string    `json:"last_error,omitempty"`
}

// Snapshot returns the checker's current state under its lock.
func (hc *HealthChecker) Snapshot() HealthCheckerSnapshot {
	hc.mu.RLock()
	defer hc.mu.RUnlock()
	return HealthCheckerSnapshot{
		AgentID:          hc.AgentID,
		Status:           hc.status,
		LastCheck:        hc.lastCheck,
		ConsecutiveFails: hc.consecutiveFails,
		LastError:        hc.lastError,
	}
}

// record stores the outcome of one check and reports the resulting state plus
// whether the status changed.
func (hc *HealthChecker) record(healthy bool, status string, err error) (HealthCheckerSnapshot, bool) {
	hc.mu.Lock()
	defer hc.mu.Unlock()

	previous := hc.status
	hc.lastCheck = time.Now()
	hc.status = status
	if healthy {
		hc.consecutiveFails = 0
		hc.lastError = ""
	} else {
		hc.consecutiveFails++
		if err != nil {
			hc.lastError = err.Error()
		}
	}

	return HealthCheckerSnapshot{
		AgentID:          hc.AgentID,
		Status:           hc.status,
		LastCheck:        hc.lastCheck,
		ConsecutiveFails: hc.consecutiveFails,
		LastError:        hc.lastError,
	}, previous != status
}

// MultiAgentMetrics tracks metrics for multi-agent system
type MultiAgentMetrics struct {
	TotalRequests  int64                    `json:"total_requests"`
	TotalErrors    int64                    `json:"total_errors"`
	AgentMetrics   map[string]*AgentMetrics `json:"agent_metrics"`
	RoutingMetrics *RoutingMetrics          `json:"routing_metrics"`
	LastUpdated    time.Time                `json:"last_updated"`
	mu             sync.RWMutex
}

// AgentMetrics tracks metrics for individual agents
type AgentMetrics struct {
	RequestCount   int64         `json:"request_count"`
	ErrorCount     int64         `json:"error_count"`
	AverageLatency time.Duration `json:"average_latency"`
	LastRequest    time.Time     `json:"last_request"`
	TotalLatency   time.Duration `json:"total_latency"`
}

// RoutingMetrics tracks routing statistics
type RoutingMetrics struct {
	RoutingDecisions map[string]int64 `json:"routing_decisions"`
	DefaultRoutes    int64            `json:"default_routes"`
	FailedRoutes     int64            `json:"failed_routes"`
}

// NewMultiAgentManager creates a new multi-agent manager
func NewMultiAgentManager(config *MultiAgentConfig, llmManager *llm.ProviderManager, toolRegistry *tools.ToolRegistry) (*MultiAgentManager, error) {
	if config == nil {
		return nil, fmt.Errorf("multi-agent config is required")
	}

	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("invalid multi-agent config: %w", err)
	}

	manager := &MultiAgentManager{
		config:         config,
		agents:         make(map[string]Agent),
		llmManager:     llmManager,
		toolRegistry:   toolRegistry,
		router:         mux.NewRouter(),
		middleware:     []MiddlewareFunc{},
		healthCheckers: make(map[string]*HealthChecker),
		logger:         logrus.New(),
		deploymentState: &DeploymentState{
			Status:      "initialized",
			StartedAt:   time.Now(),
			UpdatedAt:   time.Now(),
			AgentStates: make(map[string]*AgentState),
			Metadata:    make(map[string]interface{}),
		},
		metrics: &MultiAgentMetrics{
			AgentMetrics: make(map[string]*AgentMetrics),
			RoutingMetrics: &RoutingMetrics{
				RoutingDecisions: make(map[string]int64),
			},
			LastUpdated: time.Now(),
		},
	}

	// Initialize agents
	if err := manager.initializeAgents(); err != nil {
		return nil, fmt.Errorf("failed to initialize agents: %w", err)
	}

	// Setup routing
	if err := manager.setupRouting(); err != nil {
		return nil, fmt.Errorf("failed to setup routing: %w", err)
	}

	// Build the health checkers. They are only *started* by Start, so a manager
	// that is constructed and dropped does not leave goroutines behind.
	if err := manager.setupHealthChecking(); err != nil {
		return nil, fmt.Errorf("failed to setup health checking: %w", err)
	}

	return manager, nil
}

// initializeAgents creates and initializes all agents.
//
// The agents, deployment state and per-agent metrics are rebuilt into fresh
// maps so the function is safe to run again from Restart. Agent construction
// happens outside the manager lock: it calls into user-supplied definitions and
// factories, and holding mam.mu across that invites a deadlock if any of them
// reaches back into the manager.
func (mam *MultiAgentManager) initializeAgents() error {
	registry := GetGlobalRegistry()

	agents := make(map[string]Agent)
	states := make(map[string]*AgentState)
	agentMetrics := make(map[string]*AgentMetrics)

	// Only enabled agents are created; a disabled agent must not be started,
	// routed to or health checked.
	enabled := mam.config.GetEnabledAgents()
	agentIDs := make([]string, 0, len(enabled))
	for agentID := range enabled {
		agentIDs = append(agentIDs, agentID)
	}
	sort.Strings(agentIDs)

	for _, agentID := range agentIDs {
		agentConfig := enabled[agentID]
		// Ensure agent has an ID
		if agentConfig == nil {
			return fmt.Errorf("agent %s: configuration is empty", agentID)
		}
		if agentConfig.ID == "" {
			agentConfig.ID = agentID
		}

		// Create agent instance (as Agent interface)
		var agent Agent
		var err error

		// Check if agent is defined programmatically first
		if _, exists := registry.GetDefinition(agentID); exists {
			agent, err = registry.CreateAgentFromDefinition(agentID, mam.llmManager, mam.toolRegistry)
			if err != nil {
				return fmt.Errorf("failed to create agent %s from definition: %w", agentID, err)
			}
			mam.logger.WithField("agent_id", agentID).Info("Agent created from definition")
		} else {
			// Check for factory-based creation
			factoryIDs := registry.ListFactories()

			isFactory := false
			for _, id := range factoryIDs {
				if id == agentID {
					isFactory = true
					break
				}
			}

			if isFactory {
				// CreateAgentFromFactory returns Agent interface
				agent, err = registry.CreateAgentFromFactory(agentID, mam.llmManager, mam.toolRegistry)
				if err != nil {
					return fmt.Errorf("failed to create agent %s from factory: %w", agentID, err)
				}
				mam.logger.WithField("agent_id", agentID).Info("Agent created from factory")
			} else {
				// Fall back to config-based agent creation
				// NewAgent returns *BaseAgent which implements Agent interface
				agent = NewAgent(agentConfig, mam.llmManager, mam.toolRegistry)
				mam.logger.WithField("agent_id", agentID).Info("Agent created from config")
			}
		}

		if agent == nil {
			return fmt.Errorf("agent %s: builder returned no agent", agentID)
		}

		agents[agentID] = agent

		// Initialize agent state
		states[agentID] = &AgentState{
			ID:           agentID,
			Status:       "initialized",
			StartedAt:    time.Now(),
			UpdatedAt:    time.Now(),
			RequestCount: 0,
			ErrorCount:   0,
			HealthStatus: "unknown",
			Metadata:     make(map[string]interface{}),
		}

		// Initialize agent metrics
		agentMetrics[agentID] = &AgentMetrics{
			RequestCount:   0,
			ErrorCount:     0,
			AverageLatency: 0,
			TotalLatency:   0,
		}

		mam.logger.WithField("agent_id", agentID).Info("Agent initialized")
	}

	mam.mu.Lock()
	mam.agents = agents
	mam.deploymentState.AgentStates = states
	mam.deploymentState.UpdatedAt = time.Now()
	mam.mu.Unlock()

	mam.metrics.mu.Lock()
	mam.metrics.AgentMetrics = agentMetrics
	mam.metrics.LastUpdated = time.Now()
	mam.metrics.mu.Unlock()

	return nil
}

// setupRouting configures HTTP routing for multi-agent requests
func (mam *MultiAgentManager) setupRouting() error {
	// A config without a routing block is valid (Validate only checks routing
	// when it is present), but this used to dereference it unconditionally and
	// panic before the manager was ever returned.
	routing := mam.config.Routing
	if routing == nil {
		routing = &RoutingConfig{}
	}

	// Setup global middleware
	for _, middlewareConfig := range routing.Middleware {
		if !middlewareConfig.Enabled {
			continue
		}
		middleware, err := mam.createMiddleware(middlewareConfig)
		if err != nil {
			return fmt.Errorf("middleware %q: %w", middlewareConfig.Type, err)
		}
		mam.middleware = append(mam.middleware, middleware)
	}

	// Apply middleware to router
	for _, middleware := range mam.middleware {
		mam.router.Use(mux.MiddlewareFunc(middleware))
	}

	// Add metrics middleware
	mam.router.Use(mux.MiddlewareFunc(mam.metricsMiddleware))

	// Setup management endpoints FIRST so they don't get caught by other routes
	mam.setupManagementEndpoints()

	// Setup routing rules, highest priority first.
	for _, rule := range mam.config.SortedRules() {
		if err := mam.setupRoutingRule(routing, rule); err != nil {
			return fmt.Errorf("routing rule %q: %w", rule.ID, err)
		}
	}

	// Setup default route if configured (this should be LAST)
	if routing.DefaultAgent != "" {
		mam.router.PathPrefix("/").HandlerFunc(mam.createAgentHandler(routing.DefaultAgent, true))
	}

	return nil
}

// setupRoutingRule sets up a single routing rule.
//
// Every failure path here used to leave `route` nil and return silently, so a
// pattern the router could not express - a header pattern whose value contains
// a colon, for instance - produced a config that loaded cleanly and an agent
// that was simply unreachable. Failures are now reported and abort startup.
func (mam *MultiAgentManager) setupRoutingRule(routing *RoutingConfig, rule RoutingRule) error {
	handler := mam.createAgentHandler(rule.AgentID, false)

	conditions, err := compileConditions(rule.Conditions)
	if err != nil {
		return err
	}

	var route *mux.Route
	switch strings.ToLower(strings.TrimSpace(routing.Type)) {
	case "", "path":
		matcher, err := pathMatcher(rule)
		if err != nil {
			return err
		}
		if matcher != nil {
			route = mam.router.MatcherFunc(func(r *http.Request, rm *mux.RouteMatch) bool {
				return matcher(r.URL.Path)
			})
		} else {
			route = mam.router.Path(rule.Pattern)
		}
	case "host":
		route = mam.router.Host(rule.Pattern)
	case "header":
		// SplitN keeps colons that belong to the value ("Authorization: Bearer
		// x"); plain Split rejected any such pattern by producing three parts.
		parts := strings.SplitN(rule.Pattern, ":", 2)
		if len(parts) != 2 {
			return fmt.Errorf("header pattern %q must be \"Header-Name: value\"", rule.Pattern)
		}
		name := strings.TrimSpace(parts[0])
		value := strings.TrimSpace(parts[1])
		if name == "" {
			return fmt.Errorf("header pattern %q has an empty header name", rule.Pattern)
		}
		route = mam.router.MatcherFunc(func(r *http.Request, rm *mux.RouteMatch) bool {
			return r.Header.Get(name) == value
		})
	case "query":
		parts := strings.SplitN(rule.Pattern, "=", 2)
		if len(parts) != 2 {
			return fmt.Errorf("query pattern %q must be \"key=value\"", rule.Pattern)
		}
		key := strings.TrimSpace(parts[0])
		value := parts[1]
		if key == "" {
			return fmt.Errorf("query pattern %q has an empty key", rule.Pattern)
		}
		route = mam.router.MatcherFunc(func(r *http.Request, rm *mux.RouteMatch) bool {
			return r.URL.Query().Get(key) == value
		})
	default:
		return fmt.Errorf("unsupported routing type %q", routing.Type)
	}

	if route == nil {
		return fmt.Errorf("pattern %q could not be installed", rule.Pattern)
	}

	// Conditions are part of the rule's match: a rule that declares them must
	// not fire for a request that fails them.
	if len(conditions) > 0 {
		route = route.MatcherFunc(func(r *http.Request, rm *mux.RouteMatch) bool {
			return matchConditions(r, conditions)
		})
	}

	if rule.Method != "" {
		route = route.Methods(rule.Method)
	}
	route.Handler(handler)

	mam.logger.WithFields(logrus.Fields{
		"rule_id":    rule.ID,
		"pattern":    rule.Pattern,
		"match":      rule.MatchMode(),
		"agent_id":   rule.AgentID,
		"method":     rule.Method,
		"priority":   rule.Priority,
		"conditions": len(conditions),
	}).Info("Routing rule configured")

	return nil
}

// pathMatcher returns a path predicate for rules that need a match mode mux
// cannot express directly. It returns nil for plain exact matching so mux keeps
// handling its own path templates.
func pathMatcher(rule RoutingRule) (func(string) bool, error) {
	switch rule.MatchMode() {
	case MatchExact:
		return nil, nil
	case MatchPrefix:
		return func(path string) bool { return strings.HasPrefix(path, rule.Pattern) }, nil
	case MatchSuffix:
		return func(path string) bool { return strings.HasSuffix(path, rule.Pattern) }, nil
	case MatchContains:
		return func(path string) bool { return strings.Contains(path, rule.Pattern) }, nil
	case MatchRegex:
		re, err := regexp.Compile(rule.Pattern)
		if err != nil {
			return nil, fmt.Errorf("invalid regex pattern %q: %w", rule.Pattern, err)
		}
		return re.MatchString, nil
	default:
		return nil, fmt.Errorf("unsupported match mode %q", rule.Match)
	}
}

// compiledCondition pairs a condition with the request accessor it reads.
type compiledCondition struct {
	condition RoutingCondition
	value     func(*http.Request) string
}

// compileConditions resolves each routing condition to a request accessor,
// rejecting the ones the router cannot evaluate instead of ignoring them.
func compileConditions(conditions []RoutingCondition) ([]compiledCondition, error) {
	compiled := make([]compiledCondition, 0, len(conditions))
	for _, cond := range conditions {
		if err := validateCondition(cond); err != nil {
			return nil, err
		}
		condition := cond
		switch strings.ToLower(strings.TrimSpace(cond.Type)) {
		case ConditionHeader:
			compiled = append(compiled, compiledCondition{condition, func(r *http.Request) string {
				return r.Header.Get(condition.Key)
			}})
		case ConditionQuery:
			compiled = append(compiled, compiledCondition{condition, func(r *http.Request) string {
				return r.URL.Query().Get(condition.Key)
			}})
		case ConditionIP:
			compiled = append(compiled, compiledCondition{condition, clientIP})
		case ConditionMethod:
			compiled = append(compiled, compiledCondition{condition, func(r *http.Request) string {
				return r.Method
			}})
		}
	}
	return compiled, nil
}

func matchConditions(r *http.Request, conditions []compiledCondition) bool {
	for _, compiled := range conditions {
		if !compiled.condition.Evaluate(compiled.value(r)) {
			return false
		}
	}
	return true
}

// clientIP extracts the caller's address, preferring X-Forwarded-For when the
// server sits behind a proxy.
func clientIP(r *http.Request) string {
	if forwarded := r.Header.Get("X-Forwarded-For"); forwarded != "" {
		if first, _, found := strings.Cut(forwarded, ","); found {
			return strings.TrimSpace(first)
		}
		return strings.TrimSpace(forwarded)
	}
	if host, _, err := net.SplitHostPort(r.RemoteAddr); err == nil {
		return host
	}
	return r.RemoteAddr
}

// createAgentHandler creates an HTTP handler for a specific agent
func (mam *MultiAgentManager) createAgentHandler(agentID string, isDefault bool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()

		// Get agent
		agent, exists := mam.getAgent(agentID)
		if !exists {
			mam.recordMetrics(agentID, time.Since(start), true)
			mam.updateRoutingMetrics(agentID, isDefault)
			mam.recordFailedRoute()
			http.Error(w, fmt.Sprintf("Agent %s not found", agentID), http.StatusNotFound)
			return
		}

		// Update routing metrics
		mam.updateRoutingMetrics(agentID, isDefault)

		// Per-agent budgets can only be applied here, where the agent is known.
		if allowed, retry := mam.limiter.allowAgent(agentID); !allowed {
			mam.recordMetrics(agentID, time.Since(start), true)
			writeRateLimited(w, retry)
			return
		}

		// Parse request
		var input string
		switch r.Method {
		case http.MethodGet:
			input = r.URL.Query().Get("input")
			if input == "" {
				input = r.URL.Query().Get("q")
			}
		case http.MethodPost:
			var requestData struct {
				Input string `json:"input"`
			}
			// Cap the body: the decoder used to read whatever the client sent.
			decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, MaxRequestBodyBytes))
			if err := decoder.Decode(&requestData); err != nil {
				mam.recordMetrics(agentID, time.Since(start), true)
				http.Error(w, "Invalid request body", http.StatusBadRequest)
				return
			}
			input = requestData.Input
		default:
			mam.recordMetrics(agentID, time.Since(start), true)
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		if input == "" {
			mam.recordMetrics(agentID, time.Since(start), true)
			http.Error(w, "Input is required", http.StatusBadRequest)
			return
		}

		// Execute the agent under the timeout its own config asks for. The
		// handler used to hard-code five minutes and ignore AgentConfig.Timeout
		// entirely, so a config that promised a 100ms budget still let a slow
		// provider hold the request open for minutes.
		ctx, cancel := context.WithTimeout(r.Context(), mam.executionTimeout(agent))
		defer cancel()

		execution, err := agent.Execute(ctx, input)
		if err != nil {
			mam.recordMetrics(agentID, time.Since(start), true)
			mam.updateAgentError(agentID, err)
			http.Error(w, fmt.Sprintf("Agent execution failed: %v", err), executionErrorStatus(r.Context(), err))
			return
		}

		// Defensive: Execute currently reports every failure through err, but
		// an execution that comes back with Success=false is a failure whether
		// or not err was set, and counting it as a success would understate the
		// error rate.
		if execution != nil && !execution.Success {
			mam.recordMetrics(agentID, time.Since(start), true)
			if execution.Error != nil {
				mam.updateAgentError(agentID, execution.Error)
			} else if execution.ErrorMessage != "" {
				mam.updateAgentError(agentID, errors.New(execution.ErrorMessage))
			} else {
				mam.updateAgentError(agentID, errors.New("agent execution did not succeed"))
			}
		} else {
			// Record successful metrics
			mam.recordMetrics(agentID, time.Since(start), false)
			mam.updateAgentSuccess(agentID)
		}

		// Return response
		w.Header().Set("Content-Type", "application/json")
		response := map[string]interface{}{
			"agent_id":  agentID,
			"execution": execution,
			"timestamp": time.Now(),
		}
		_ = json.NewEncoder(w).Encode(response)
	}
}

// executionTimeout returns the deadline to apply to one agent run.
func (mam *MultiAgentManager) executionTimeout(agent Agent) time.Duration {
	if agent != nil {
		if config := agent.GetConfig(); config != nil && config.Timeout > 0 {
			return config.Timeout
		}
	}
	return DefaultAgentExecutionTimeout
}

// executionErrorStatus distinguishes "we ran out of time" from "the agent
// failed", which used to be reported identically as a 500.
func executionErrorStatus(requestCtx context.Context, err error) int {
	if errors.Is(err, context.DeadlineExceeded) {
		return http.StatusGatewayTimeout
	}
	if errors.Is(err, context.Canceled) {
		if requestCtx.Err() != nil {
			// The client went away; nothing will read the response anyway.
			return http.StatusRequestTimeout
		}
		return http.StatusGatewayTimeout
	}
	return http.StatusInternalServerError
}

// setupManagementEndpoints sets up management and monitoring endpoints
func (mam *MultiAgentManager) setupManagementEndpoints() {
	// Health check endpoint
	mam.router.HandleFunc("/health", mam.handleHealth).Methods("GET")
	mam.router.HandleFunc("/health/{agent_id}", mam.handleAgentHealth).Methods("GET")

	// Metrics endpoint
	mam.router.HandleFunc("/metrics", mam.handleMetrics).Methods("GET")

	// Agent management endpoints
	mam.router.HandleFunc("/agents", mam.handleListAgents).Methods("GET")
	mam.router.HandleFunc("/agents/{agent_id}", mam.handleGetAgent).Methods("GET")
	mam.router.HandleFunc("/agents/{agent_id}/status", mam.handleAgentStatus).Methods("GET")

	// Configuration endpoints
	mam.router.HandleFunc("/config", mam.handleGetConfig).Methods("GET")
	mam.router.HandleFunc("/routing", mam.handleGetRouting).Methods("GET")

	// Deployment endpoints
	mam.router.HandleFunc("/deployment/status", mam.handleDeploymentStatus).Methods("GET")
	mam.router.HandleFunc("/deployment/restart", mam.handleRestart).Methods("POST")
}

// setupHealthChecking builds the per-agent health checkers.
//
// It deliberately does not start them: the goroutines belong to the manager's
// running lifetime and are launched by Start and reclaimed by Stop.
func (mam *MultiAgentManager) setupHealthChecking() error {
	mam.healthMu.Lock()
	defer mam.healthMu.Unlock()

	mam.healthCheckers = make(map[string]*HealthChecker)

	if mam.config.Deployment == nil || mam.config.Deployment.HealthCheck == nil || !mam.config.Deployment.HealthCheck.Enabled {
		return nil
	}

	for agentID := range mam.config.GetEnabledAgents() {
		healthConfig := mam.config.Deployment.HealthCheck

		// Check for agent-specific health check config
		if agentSpecific, exists := healthConfig.AgentSpecific[agentID]; exists && agentSpecific != nil {
			healthConfig = agentSpecific
		}

		mam.healthCheckers[agentID] = &HealthChecker{
			AgentID: agentID,
			Config:  healthConfig,
			Logger:  logrus.New(),
			status:  "unknown",
		}
	}

	return nil
}

// startHealthCheckers launches one goroutine per checker, all cancellable.
func (mam *MultiAgentManager) startHealthCheckers() {
	mam.lifecycleMu.Lock()
	defer mam.lifecycleMu.Unlock()

	if mam.healthRunning {
		return
	}

	mam.healthMu.RLock()
	checkers := make([]*HealthChecker, 0, len(mam.healthCheckers))
	for _, checker := range mam.healthCheckers {
		checkers = append(checkers, checker)
	}
	mam.healthMu.RUnlock()

	if len(checkers) == 0 {
		return
	}

	ctx, cancel := context.WithCancel(context.Background())
	mam.healthCancel = cancel
	mam.healthRunning = true

	for _, checker := range checkers {
		mam.healthWG.Add(1)
		go func(c *HealthChecker) {
			defer mam.healthWG.Done()
			mam.runHealthChecker(ctx, c)
		}(checker)
	}
}

// stopHealthCheckers cancels the checkers and waits for them to exit, giving up
// when ctx expires so a caller with a deadline is never blocked forever.
func (mam *MultiAgentManager) stopHealthCheckers(ctx context.Context) error {
	mam.lifecycleMu.Lock()
	if !mam.healthRunning {
		mam.lifecycleMu.Unlock()
		return nil
	}
	cancel := mam.healthCancel
	mam.healthCancel = nil
	mam.healthRunning = false
	mam.lifecycleMu.Unlock()

	if cancel != nil {
		cancel()
	}

	done := make(chan struct{})
	go func() {
		mam.healthWG.Wait()
		close(done)
	}()

	if ctx == nil {
		<-done
		return nil
	}

	select {
	case <-done:
		return nil
	case <-ctx.Done():
		return fmt.Errorf("timed out waiting for health checkers to stop: %w", ctx.Err())
	}
}

// runHealthChecker runs health checks for an agent until ctx is cancelled.
func (mam *MultiAgentManager) runHealthChecker(ctx context.Context, checker *HealthChecker) {
	// Config.Period substitutes a default for a missing period_seconds. The
	// raw value was handed to time.NewTicker, and time.NewTicker(0) panics -
	// on a background goroutine, so an enabled health check with no interval
	// crashed the whole process at startup.
	ticker := time.NewTicker(checker.Config.Period())
	defer ticker.Stop()

	// Initial delay, interruptible: an unconditional Sleep here meant Stop had
	// to wait out initial_delay_seconds (30s in the shipped default config)
	// before the goroutine would even look at cancellation.
	if delay := checker.Config.InitialDelay(); delay > 0 {
		timer := time.NewTimer(delay)
		select {
		case <-ctx.Done():
			timer.Stop()
			return
		case <-timer.C:
		}
	}

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			mam.performHealthCheck(ctx, checker)
		}
	}
}

// performHealthCheck performs a single health check.
//
// The old check asked agent.IsRunning(), which reports whether the agent is
// *mid-execution*: an idle, perfectly healthy agent was therefore marked
// unhealthy on every tick and a busy one looked healthy. The real question is
// whether the agent exists and its LLM provider is reachable.
func (mam *MultiAgentManager) performHealthCheck(ctx context.Context, checker *HealthChecker) {
	agent, exists := mam.getAgent(checker.AgentID)
	if !exists {
		snapshot, changed := checker.record(false, "not_found", fmt.Errorf("agent %s is not registered", checker.AgentID))
		mam.updateAgentHealthStatus(checker.AgentID, "unhealthy")
		mam.logHealth(checker, snapshot, changed)
		return
	}

	checkCtx, cancel := context.WithTimeout(ctx, checker.Config.Timeout())
	defer cancel()

	if err := mam.checkAgentProvider(checkCtx, agent); err != nil {
		snapshot, changed := checker.record(false, "unhealthy", err)
		if snapshot.ConsecutiveFails >= checker.Config.Failures() {
			mam.updateAgentHealthStatus(checker.AgentID, "unhealthy")
		} else {
			mam.updateAgentHealthStatus(checker.AgentID, "degraded")
		}
		mam.logHealth(checker, snapshot, changed)
		return
	}

	snapshot, changed := checker.record(true, "healthy", nil)
	mam.updateAgentHealthStatus(checker.AgentID, "healthy")
	mam.logHealth(checker, snapshot, changed)
}

// checkAgentProvider verifies that the agent's configured LLM provider exists
// and answers a health probe.
func (mam *MultiAgentManager) checkAgentProvider(ctx context.Context, agent Agent) error {
	config := agent.GetConfig()
	if config == nil {
		return fmt.Errorf("agent has no configuration")
	}
	if mam.llmManager == nil {
		return fmt.Errorf("no LLM provider manager configured")
	}
	provider, err := mam.llmManager.GetProvider(config.Provider)
	if err != nil {
		return fmt.Errorf("provider %s unavailable: %w", config.Provider, err)
	}
	if err := provider.IsHealthy(ctx); err != nil {
		return fmt.Errorf("provider %s unhealthy: %w", config.Provider, err)
	}
	return nil
}

func (mam *MultiAgentManager) logHealth(checker *HealthChecker, snapshot HealthCheckerSnapshot, changed bool) {
	if checker.Logger == nil {
		return
	}
	fields := logrus.Fields{
		"agent_id":          snapshot.AgentID,
		"status":            snapshot.Status,
		"consecutive_fails": snapshot.ConsecutiveFails,
	}
	switch {
	case snapshot.ConsecutiveFails == checker.Config.Failures():
		checker.Logger.WithFields(fields).Warn("Agent health check failing consistently")
	case changed && snapshot.Status == "healthy":
		checker.Logger.WithFields(fields).Info("Agent health check recovered")
	}
}

// HealthCheckerStatus returns the latest health checker state for an agent.
func (mam *MultiAgentManager) HealthCheckerStatus(agentID string) (HealthCheckerSnapshot, bool) {
	mam.healthMu.RLock()
	checker, exists := mam.healthCheckers[agentID]
	mam.healthMu.RUnlock()
	if !exists {
		return HealthCheckerSnapshot{}, false
	}
	return checker.Snapshot(), true
}

// CheckHealthNow runs one health check per agent synchronously and returns the
// resulting snapshots. It exists so callers (and tests) can observe health
// without waiting a full tick.
func (mam *MultiAgentManager) CheckHealthNow(ctx context.Context) map[string]HealthCheckerSnapshot {
	mam.healthMu.RLock()
	checkers := make([]*HealthChecker, 0, len(mam.healthCheckers))
	for _, checker := range mam.healthCheckers {
		checkers = append(checkers, checker)
	}
	mam.healthMu.RUnlock()

	results := make(map[string]HealthCheckerSnapshot, len(checkers))
	for _, checker := range checkers {
		mam.performHealthCheck(ctx, checker)
		results[checker.AgentID] = checker.Snapshot()
	}
	return results
}

// Middleware creation.
//
// An unrecognized or unusable middleware entry is now an error rather than a
// warning: "enabled: true" that quietly installs nothing is exactly the failure
// mode that let auth and rate limiting ship as no-ops.
func (mam *MultiAgentManager) createMiddleware(config MiddlewareConfig) (MiddlewareFunc, error) {
	switch strings.ToLower(strings.TrimSpace(config.Type)) {
	case "cors":
		return mam.corsMiddleware, nil
	case "auth":
		keys, err := mam.resolveAPIKeys(config)
		if err != nil {
			return nil, err
		}
		return mam.newAuthMiddleware(keys), nil
	case "logging":
		return mam.loggingMiddleware, nil
	case "rate_limit":
		limiter, err := mam.newRateLimiter(config)
		if err != nil {
			return nil, err
		}
		mam.limiter = limiter
		return limiter.middleware, nil
	default:
		return nil, fmt.Errorf("unknown middleware type %q", config.Type)
	}
}

// corsConfig returns the effective CORS settings, or nil when CORS is off.
// Every level of this chain is optional in a config file, and dereferencing it
// blindly panicked inside the HTTP handler - after the manager had started and
// started serving.
func (mam *MultiAgentManager) corsConfig() *CORSConfig {
	if mam.config == nil || mam.config.Shared == nil || mam.config.Shared.Security == nil {
		return nil
	}
	cors := mam.config.Shared.Security.CORS
	if cors == nil || !cors.Enabled {
		return nil
	}
	return cors
}

// Middleware implementations
func (mam *MultiAgentManager) corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if cors := mam.corsConfig(); cors != nil {
			origin := r.Header.Get("Origin")
			if origin != "" && mam.isAllowedOrigin(origin, cors.AllowedOrigins) {
				w.Header().Set("Access-Control-Allow-Origin", origin)
			}

			w.Header().Set("Access-Control-Allow-Methods", strings.Join(cors.AllowedMethods, ", "))
			w.Header().Set("Access-Control-Allow-Headers", strings.Join(cors.AllowedHeaders, ", "))
			w.Header().Set("Access-Control-Expose-Headers", strings.Join(cors.ExposedHeaders, ", "))
			w.Header().Set("Access-Control-Max-Age", fmt.Sprintf("%d", cors.MaxAge))

			if cors.AllowCredentials {
				w.Header().Set("Access-Control-Allow-Credentials", "true")
			}

			if r.Method == http.MethodOptions {
				w.WriteHeader(http.StatusOK)
				return
			}
		}

		next.ServeHTTP(w, r)
	})
}

// resolveAPIKeys collects the accepted API keys from the middleware entry and
// from shared.security.authentication.
//
// Enabling auth without keys is refused. The old middleware accepted *any*
// non-empty key with the comment "in a real implementation, validate the API
// key", so a deployment that believed it was authenticated was wide open.
func (mam *MultiAgentManager) resolveAPIKeys(config MiddlewareConfig) ([]string, error) {
	keys := extractKeys(config.Config)

	if mam.config != nil && mam.config.Shared != nil && mam.config.Shared.Security != nil {
		if auth := mam.config.Shared.Security.Authentication; auth != nil {
			keys = append(keys, extractKeys(auth.Config)...)
		}
	}

	// Deduplicate and drop blanks.
	seen := make(map[string]bool, len(keys))
	unique := make([]string, 0, len(keys))
	for _, key := range keys {
		key = strings.TrimSpace(key)
		if key == "" || seen[key] {
			continue
		}
		seen[key] = true
		unique = append(unique, key)
	}

	if len(unique) == 0 {
		return nil, fmt.Errorf("auth middleware is enabled but no API keys are configured (set config.api_keys or shared.security.authentication.config.api_keys)")
	}
	return unique, nil
}

// extractKeys reads an API key list from a middleware config map, accepting the
// spellings YAML and JSON configs use.
func extractKeys(config map[string]interface{}) []string {
	var keys []string
	for _, field := range []string{"api_keys", "apiKeys", "keys"} {
		raw, exists := config[field]
		if !exists {
			continue
		}
		switch v := raw.(type) {
		case []string:
			keys = append(keys, v...)
		case []interface{}:
			for _, item := range v {
				if str, ok := item.(string); ok {
					keys = append(keys, str)
				}
			}
		case string:
			for _, part := range strings.Split(v, ",") {
				keys = append(keys, strings.TrimSpace(part))
			}
		}
	}
	for _, field := range []string{"api_key", "apiKey", "key"} {
		if str, ok := config[field].(string); ok {
			keys = append(keys, str)
		}
	}
	return keys
}

func (mam *MultiAgentManager) newAuthMiddleware(keys []string) MiddlewareFunc {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Preflight requests never carry credentials.
			if r.Method == http.MethodOptions {
				next.ServeHTTP(w, r)
				return
			}

			apiKey := r.Header.Get("X-API-Key")
			if apiKey == "" {
				if bearer := r.Header.Get("Authorization"); strings.HasPrefix(bearer, "Bearer ") {
					apiKey = strings.TrimSpace(strings.TrimPrefix(bearer, "Bearer "))
				}
			}
			if apiKey == "" {
				apiKey = r.URL.Query().Get("api_key")
			}

			if apiKey == "" {
				http.Error(w, "API key required", http.StatusUnauthorized)
				return
			}

			if !matchesAnyKey(apiKey, keys) {
				mam.logger.WithField("path", r.URL.Path).Warn("Rejected request with invalid API key")
				http.Error(w, "Invalid API key", http.StatusUnauthorized)
				return
			}

			next.ServeHTTP(w, r)
		})
	}
}

// matchesAnyKey compares in constant time so the endpoint does not leak key
// material through response timing.
func matchesAnyKey(presented string, keys []string) bool {
	match := false
	for _, key := range keys {
		if subtle.ConstantTimeCompare([]byte(presented), []byte(key)) == 1 {
			match = true
		}
	}
	return match
}

func (mam *MultiAgentManager) loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		startTime := time.Now()

		next.ServeHTTP(w, r)

		mam.logger.WithFields(logrus.Fields{
			"method":   r.Method,
			"path":     r.URL.Path,
			"duration": time.Since(startTime),
			"remote":   r.RemoteAddr,
		}).Info("HTTP request")
	})
}

// maxRateLimitKeys caps how many per-IP / per-agent buckets are retained. A
// per-IP limiter keyed by a value the caller controls is an unbounded map: one
// bucket per source address, kept for the life of the process. Idle buckets are
// swept and, past this ceiling, the oldest are dropped.
const maxRateLimitKeys = 10000

// tokenBucket is a standard token bucket: capacity tokens, refilled at rate
// tokens per second.
type tokenBucket struct {
	capacity float64
	rate     float64
	tokens   float64
	last     time.Time
}

func newTokenBucket(requests int, period time.Duration, burst int, now time.Time) *tokenBucket {
	if period <= 0 {
		period = time.Minute
	}
	capacity := float64(burst)
	if capacity < float64(requests) {
		capacity = float64(requests)
	}
	if capacity <= 0 {
		capacity = 1
	}
	return &tokenBucket{
		capacity: capacity,
		rate:     float64(requests) / period.Seconds(),
		tokens:   capacity,
		last:     now,
	}
}

// allow consumes a token if one is available and otherwise reports how long the
// caller must wait.
func (b *tokenBucket) allow(now time.Time) (bool, time.Duration) {
	if elapsed := now.Sub(b.last); elapsed > 0 {
		b.tokens += elapsed.Seconds() * b.rate
		if b.tokens > b.capacity {
			b.tokens = b.capacity
		}
		b.last = now
	}

	if b.tokens >= 1 {
		b.tokens--
		return true, 0
	}

	if b.rate <= 0 {
		return false, time.Minute
	}
	wait := time.Duration(((1 - b.tokens) / b.rate) * float64(time.Second))
	if wait < time.Second {
		wait = time.Second
	}
	return false, wait
}

// rateLimitRule is a resolved budget.
type rateLimitRule struct {
	requests int
	period   time.Duration
	burst    int
}

// rateLimiter enforces the configured request budgets.
//
// The old rateLimitMiddleware called next.ServeHTTP and nothing else, with a
// comment saying a real limiter should go here, while the config carried a
// complete RateLimitConfig. Every configured limit was silently ignored.
type rateLimiter struct {
	logger *logrus.Logger

	global    *rateLimitRule
	perIP     *rateLimitRule
	perAgent  map[string]*rateLimitRule
	skipPaths map[string]bool

	// now is injectable so tests can advance time without sleeping.
	now func() time.Time

	mu           sync.Mutex
	globalBucket *tokenBucket
	buckets      map[string]*tokenBucket
	lastSeen     map[string]time.Time
}

func ruleFromConfig(limit *RateLimit, fallbackBurst int) *rateLimitRule {
	if limit == nil || limit.Requests <= 0 {
		return nil
	}
	period := limit.Period
	if period <= 0 {
		period = time.Minute
	}
	burst := limit.Burst
	if burst <= 0 {
		burst = fallbackBurst
	}
	return &rateLimitRule{requests: limit.Requests, period: period, burst: burst}
}

// ruleFromMiddleware reads the inline middleware spelling
// (requests_per_minute / burst_limit) used by the shipped example config.
func ruleFromMiddleware(config map[string]interface{}) *rateLimitRule {
	requests := intFromConfig(config, "requests_per_minute", "requestsPerMinute", "requests")
	if requests <= 0 {
		return nil
	}
	period := time.Minute
	if raw, ok := config["period"].(string); ok {
		if parsed, err := time.ParseDuration(raw); err == nil && parsed > 0 {
			period = parsed
		}
	}
	return &rateLimitRule{
		requests: requests,
		period:   period,
		burst:    intFromConfig(config, "burst_limit", "burstLimit", "burst"),
	}
}

func intFromConfig(config map[string]interface{}, fields ...string) int {
	for _, field := range fields {
		switch v := config[field].(type) {
		case int:
			return v
		case int64:
			return int(v)
		case float64:
			return int(v)
		}
	}
	return 0
}

func (mam *MultiAgentManager) newRateLimiter(config MiddlewareConfig) (*rateLimiter, error) {
	limiter := &rateLimiter{
		logger:    mam.logger,
		now:       time.Now,
		perAgent:  make(map[string]*rateLimitRule),
		skipPaths: make(map[string]bool),
		buckets:   make(map[string]*tokenBucket),
		lastSeen:  make(map[string]time.Time),
	}

	limiter.global = ruleFromMiddleware(config.Config)

	var shared *RateLimitConfig
	if mam.config != nil && mam.config.Shared != nil && mam.config.Shared.Security != nil {
		shared = mam.config.Shared.Security.RateLimit
	}

	if shared != nil && shared.Enabled {
		if rule := ruleFromConfig(shared.Global, shared.BurstLimit); rule != nil {
			limiter.global = rule
		}
		if rule := ruleFromConfig(shared.PerIP, shared.BurstLimit); rule != nil {
			limiter.perIP = rule
		}
		for agentID, limit := range shared.PerAgent {
			if rule := ruleFromConfig(limit, shared.BurstLimit); rule != nil {
				limiter.perAgent[agentID] = rule
			}
		}
		for _, path := range skipPathsOf(shared) {
			limiter.skipPaths[path] = true
		}
		if shared.PerUser != nil && shared.PerUser.Requests > 0 {
			// Saying so beats silently ignoring it: nothing in the request
			// carries a user identity, so this budget cannot be enforced.
			mam.logger.Warn("rate_limit: per_user limits are configured but not enforced (no user identity source)")
		}
	}

	if limiter.global == nil && limiter.perIP == nil && len(limiter.perAgent) == 0 {
		return nil, fmt.Errorf("rate_limit middleware is enabled but no limits are configured (set config.requests_per_minute or shared.security.rate_limit)")
	}

	if limiter.global != nil {
		limiter.globalBucket = newTokenBucket(limiter.global.requests, limiter.global.period, limiter.global.burst, limiter.now())
	}

	return limiter, nil
}

func skipPathsOf(config *RateLimitConfig) []string {
	var paths []string
	for _, limit := range []*RateLimit{config.Global, config.PerIP, config.PerUser} {
		if limit != nil {
			paths = append(paths, limit.SkipPaths...)
		}
	}
	return paths
}

// allowKeyed applies a per-key budget, creating the bucket on first use.
func (rl *rateLimiter) allowKeyed(key string, rule *rateLimitRule) (bool, time.Duration) {
	now := rl.now()

	rl.mu.Lock()
	defer rl.mu.Unlock()

	bucket, exists := rl.buckets[key]
	if !exists {
		bucket = newTokenBucket(rule.requests, rule.period, rule.burst, now)
		rl.buckets[key] = bucket
	}
	rl.lastSeen[key] = now
	allowed, retry := bucket.allow(now)
	rl.evictLocked(now)
	return allowed, retry
}

// evictLocked keeps the keyed bucket maps bounded. Must be called with mu held.
func (rl *rateLimiter) evictLocked(now time.Time) {
	if len(rl.buckets) <= maxRateLimitKeys {
		return
	}
	// Drop everything untouched in the last minute first, then, if that was not
	// enough, the least recently used keys.
	for key, seen := range rl.lastSeen {
		if now.Sub(seen) > time.Minute {
			delete(rl.buckets, key)
			delete(rl.lastSeen, key)
		}
	}
	for len(rl.buckets) > maxRateLimitKeys {
		oldestKey := ""
		var oldest time.Time
		for key, seen := range rl.lastSeen {
			if oldestKey == "" || seen.Before(oldest) {
				oldestKey, oldest = key, seen
			}
		}
		if oldestKey == "" {
			return
		}
		delete(rl.buckets, oldestKey)
		delete(rl.lastSeen, oldestKey)
	}
}

// allowRequest applies the global and per-IP budgets.
func (rl *rateLimiter) allowRequest(r *http.Request) (bool, time.Duration) {
	if rl.skipPaths[r.URL.Path] {
		return true, 0
	}

	if rl.globalBucket != nil {
		rl.mu.Lock()
		allowed, retry := rl.globalBucket.allow(rl.now())
		rl.mu.Unlock()
		if !allowed {
			return false, retry
		}
	}

	if rl.perIP != nil {
		return rl.allowKeyed("ip:"+clientIP(r), rl.perIP)
	}

	return true, 0
}

// allowAgent applies a per-agent budget, if one is configured for that agent.
func (rl *rateLimiter) allowAgent(agentID string) (bool, time.Duration) {
	if rl == nil {
		return true, 0
	}
	rule, exists := rl.perAgent[agentID]
	if !exists {
		return true, 0
	}
	return rl.allowKeyed("agent:"+agentID, rule)
}

func (rl *rateLimiter) middleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if allowed, retry := rl.allowRequest(r); !allowed {
			writeRateLimited(w, retry)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func writeRateLimited(w http.ResponseWriter, retry time.Duration) {
	seconds := int(retry.Seconds())
	if seconds < 1 {
		seconds = 1
	}
	w.Header().Set("Retry-After", fmt.Sprintf("%d", seconds))
	http.Error(w, "Rate limit exceeded", http.StatusTooManyRequests)
}

func (mam *MultiAgentManager) metricsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		next.ServeHTTP(w, r)

		// Update global metrics
		mam.metrics.mu.Lock()
		mam.metrics.TotalRequests++
		mam.metrics.LastUpdated = time.Now()
		mam.metrics.mu.Unlock()
	})
}

// Helper methods
func (mam *MultiAgentManager) getAgent(agentID string) (Agent, bool) {
	mam.mu.RLock()
	defer mam.mu.RUnlock()

	agent, exists := mam.agents[agentID]
	return agent, exists
}

func (mam *MultiAgentManager) isAllowedOrigin(origin string, allowedOrigins []string) bool {
	for _, allowed := range allowedOrigins {
		if allowed == "*" || allowed == origin {
			return true
		}
	}
	return false
}

// recordMetrics accounts for one request against an agent.
//
// The whole body used to sit inside "if the agent has a metrics entry", so a
// request routed to an agent that does not exist - the 404 path, which is
// precisely a failure worth counting - incremented nothing at all and
// TotalErrors permanently understated the error rate.
func (mam *MultiAgentManager) recordMetrics(agentID string, duration time.Duration, isError bool) {
	mam.metrics.mu.Lock()
	defer mam.metrics.mu.Unlock()

	agentMetrics, exists := mam.metrics.AgentMetrics[agentID]
	if !exists {
		agentMetrics = &AgentMetrics{}
		mam.metrics.AgentMetrics[agentID] = agentMetrics
	}

	agentMetrics.RequestCount++
	agentMetrics.LastRequest = time.Now()
	agentMetrics.TotalLatency += duration
	agentMetrics.AverageLatency = agentMetrics.TotalLatency / time.Duration(agentMetrics.RequestCount)

	if isError {
		agentMetrics.ErrorCount++
		mam.metrics.TotalErrors++
	}

	mam.metrics.LastUpdated = time.Now()
}

func (mam *MultiAgentManager) updateRoutingMetrics(agentID string, isDefault bool) {
	mam.metrics.mu.Lock()
	defer mam.metrics.mu.Unlock()

	if isDefault {
		mam.metrics.RoutingMetrics.DefaultRoutes++
	} else {
		mam.metrics.RoutingMetrics.RoutingDecisions[agentID]++
	}
}

// recordFailedRoute counts a request that reached a route whose agent is
// missing. RoutingMetrics.FailedRoutes was declared, serialized and never once
// incremented, so the metric was always zero.
func (mam *MultiAgentManager) recordFailedRoute() {
	mam.metrics.mu.Lock()
	defer mam.metrics.mu.Unlock()
	mam.metrics.RoutingMetrics.FailedRoutes++
}

// updateAgentError records a failure against both the agent and the deployment.
//
// DeploymentState.ErrorCount and LastError were declared and serialized to
// /deployment/status but never written, so the deployment always looked clean
// no matter how many agent executions failed.
func (mam *MultiAgentManager) updateAgentError(agentID string, err error) {
	if err == nil {
		return
	}

	mam.mu.Lock()
	defer mam.mu.Unlock()

	mam.deploymentState.ErrorCount++
	mam.deploymentState.LastError = err.Error()
	mam.deploymentState.UpdatedAt = time.Now()

	if state, exists := mam.deploymentState.AgentStates[agentID]; exists {
		state.ErrorCount++
		state.LastError = err.Error()
		state.UpdatedAt = time.Now()
	}
}

func (mam *MultiAgentManager) updateAgentSuccess(agentID string) {
	mam.mu.Lock()
	defer mam.mu.Unlock()

	if state, exists := mam.deploymentState.AgentStates[agentID]; exists {
		state.RequestCount++
		state.LastRequest = time.Now()
		state.UpdatedAt = time.Now()
		state.Status = "running"
	}
}

func (mam *MultiAgentManager) updateAgentHealthStatus(agentID, status string) {
	mam.mu.Lock()
	defer mam.mu.Unlock()

	if state, exists := mam.deploymentState.AgentStates[agentID]; exists {
		state.HealthStatus = status
		state.UpdatedAt = time.Now()
	}
}

// HTTP Handlers

// OverallHealth summarizes agent health: "healthy" when every agent is, and
// "unhealthy" as soon as one is not.
func (mam *MultiAgentManager) OverallHealth() (string, map[string]string) {
	mam.mu.RLock()
	defer mam.mu.RUnlock()

	agents := make(map[string]string, len(mam.deploymentState.AgentStates))
	status := "healthy"
	for agentID, state := range mam.deploymentState.AgentStates {
		agents[agentID] = state.HealthStatus
		switch state.HealthStatus {
		case "unhealthy", "not_found", "error":
			status = "unhealthy"
		case "degraded", "unknown":
			if status == "healthy" {
				status = "degraded"
			}
		}
	}
	if len(agents) == 0 {
		status = "unhealthy"
	}
	return status, agents
}

// handleHealth reports the real aggregate health.
//
// It used to hard-code "status": "healthy" and HTTP 200 while listing agents
// that said "unhealthy" right next to it, so every liveness probe pointed at
// this endpoint passed no matter what state the system was in.
func (mam *MultiAgentManager) handleHealth(w http.ResponseWriter, r *http.Request) {
	status, agents := mam.OverallHealth()

	health := map[string]interface{}{
		"status":    status,
		"timestamp": time.Now(),
		"agents":    agents,
	}

	w.Header().Set("Content-Type", "application/json")
	if status == "unhealthy" {
		w.WriteHeader(http.StatusServiceUnavailable)
	}
	_ = json.NewEncoder(w).Encode(health)
}

func (mam *MultiAgentManager) handleAgentHealth(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	agentID := vars["agent_id"]

	// Copy under the lock: the handler used to keep the live *AgentState and
	// read its fields after unlocking, racing every in-flight request.
	mam.mu.RLock()
	live, exists := mam.deploymentState.AgentStates[agentID]
	var state *AgentState
	if exists {
		state = live.Clone()
	}
	mam.mu.RUnlock()

	if !exists {
		http.Error(w, "Agent not found", http.StatusNotFound)
		return
	}

	health := map[string]interface{}{
		"agent_id":      agentID,
		"status":        state.HealthStatus,
		"timestamp":     time.Now(),
		"request_count": state.RequestCount,
		"error_count":   state.ErrorCount,
		"last_error":    state.LastError,
		"last_request":  state.LastRequest,
		"started_at":    state.StartedAt,
	}
	if snapshot, ok := mam.HealthCheckerStatus(agentID); ok {
		health["last_check"] = snapshot.LastCheck
		health["consecutive_fails"] = snapshot.ConsecutiveFails
		if snapshot.LastError != "" {
			health["check_error"] = snapshot.LastError
		}
	}

	w.Header().Set("Content-Type", "application/json")
	if state.HealthStatus == "unhealthy" || state.HealthStatus == "not_found" {
		w.WriteHeader(http.StatusServiceUnavailable)
	}
	_ = json.NewEncoder(w).Encode(health)
}

func (mam *MultiAgentManager) handleMetrics(w http.ResponseWriter, r *http.Request) {
	// Encode a snapshot rather than the live struct so the encoder never walks
	// maps another request is mutating.
	metrics := mam.GetMetrics()

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(metrics)
}

func (mam *MultiAgentManager) handleListAgents(w http.ResponseWriter, r *http.Request) {
	mam.mu.RLock()
	agents := make(map[string]interface{})
	for agentID, state := range mam.deploymentState.AgentStates {
		agents[agentID] = map[string]interface{}{
			"status":        state.Status,
			"health_status": state.HealthStatus,
			"request_count": state.RequestCount,
			"error_count":   state.ErrorCount,
			"last_error":    state.LastError,
			"started_at":    state.StartedAt,
		}
	}
	mam.mu.RUnlock()

	response := map[string]interface{}{
		"agents": agents,
		"count":  len(agents),
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(response)
}

func (mam *MultiAgentManager) handleGetAgent(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	agentID := vars["agent_id"]

	agent, exists := mam.getAgent(agentID)
	if !exists {
		http.Error(w, "Agent not found", http.StatusNotFound)
		return
	}

	mam.mu.RLock()
	state := mam.deploymentState.AgentStates[agentID].Clone()
	mam.mu.RUnlock()

	response := map[string]interface{}{
		"agent_id": agentID,
		"config":   agent.GetConfig(),
		"state":    state,
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(response)
}

func (mam *MultiAgentManager) handleAgentStatus(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	agentID := vars["agent_id"]

	// The live pointer used to be encoded after the lock was dropped; -race
	// flagged it against every concurrent request that touched the same agent.
	mam.mu.RLock()
	live, exists := mam.deploymentState.AgentStates[agentID]
	var state *AgentState
	if exists {
		state = live.Clone()
	}
	mam.mu.RUnlock()

	if !exists {
		http.Error(w, "Agent not found", http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(state)
}

// handleGetConfig serves the configuration with credentials removed.
//
// It used to encode the raw config, so an unauthenticated GET /config returned
// every LLM API key, the database and cache passwords and both secret maps.
func (mam *MultiAgentManager) handleGetConfig(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(mam.config.Redacted())
}

func (mam *MultiAgentManager) handleGetRouting(w http.ResponseWriter, r *http.Request) {
	routing := mam.config.Routing
	if routing == nil {
		routing = &RoutingConfig{}
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(routing)
}

func (mam *MultiAgentManager) handleDeploymentStatus(w http.ResponseWriter, r *http.Request) {
	state := mam.GetDeploymentState()

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(state)
}

// handleRestart actually restarts the agents.
//
// It used to log "Restart requested", answer "restart_initiated" and do
// nothing whatsoever - a caller had no way to tell the difference between a
// restart and a no-op.
func (mam *MultiAgentManager) handleRestart(w http.ResponseWriter, r *http.Request) {
	mam.logger.Info("Restart requested")

	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()

	w.Header().Set("Content-Type", "application/json")

	if err := mam.Restart(ctx); err != nil {
		mam.logger.WithError(err).Error("Restart failed")
		w.WriteHeader(http.StatusInternalServerError)
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"status":    "restart_failed",
			"error":     err.Error(),
			"timestamp": time.Now(),
		})
		return
	}

	_ = json.NewEncoder(w).Encode(map[string]interface{}{
		"status":    "restarted",
		"timestamp": time.Now(),
	})
}

// Public methods

// setStatus moves the deployment and every agent to a status.
func (mam *MultiAgentManager) setStatus(status string) {
	mam.mu.Lock()
	defer mam.mu.Unlock()

	now := time.Now()
	mam.deploymentState.Status = status
	mam.deploymentState.UpdatedAt = now

	for agentID := range mam.agents {
		if state, exists := mam.deploymentState.AgentStates[agentID]; exists {
			state.Status = status
			state.UpdatedAt = now
		}
	}
}

// Start starts the multi-agent manager.
//
// ctx is honored rather than ignored: a caller that hands in an already
// cancelled context gets an error instead of a manager that reports "running".
func (mam *MultiAgentManager) Start(ctx context.Context) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("cannot start multi-agent manager: %w", err)
	}

	mam.logger.Info("Starting multi-agent manager")

	mam.setStatus("starting")

	// Health checkers run for exactly as long as the manager does.
	mam.startHealthCheckers()

	mam.setStatus("running")

	mam.logger.Info("Multi-agent manager started successfully")
	return nil
}

// Stop stops the multi-agent manager and reclaims its goroutines.
//
// Stop is safe to call more than once and safe to call on a manager that was
// never started.
func (mam *MultiAgentManager) Stop(ctx context.Context) error {
	mam.logger.Info("Stopping multi-agent manager")

	mam.setStatus("stopping")

	// Cancel and join the health checkers. Without this they outlived the
	// manager entirely: one goroutine per agent, ticking forever.
	err := mam.stopHealthCheckers(ctx)

	mam.setStatus("stopped")

	if err != nil {
		mam.logger.WithError(err).Warn("Multi-agent manager stopped with pending health checkers")
		return err
	}

	mam.logger.Info("Multi-agent manager stopped")
	return nil
}

// Restart rebuilds every agent from configuration and brings the manager back
// up. Counters and error state are reset; routing is left alone because the
// router is wired to agent IDs, which do not change.
func (mam *MultiAgentManager) Restart(ctx context.Context) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("cannot restart multi-agent manager: %w", err)
	}

	if err := mam.Stop(ctx); err != nil {
		return fmt.Errorf("restart: %w", err)
	}

	if err := mam.initializeAgents(); err != nil {
		mam.setStatus("error")
		mam.mu.Lock()
		mam.deploymentState.LastError = err.Error()
		mam.deploymentState.ErrorCount++
		mam.mu.Unlock()
		return fmt.Errorf("restart: failed to reinitialize agents: %w", err)
	}

	mam.mu.Lock()
	mam.deploymentState.ErrorCount = 0
	mam.deploymentState.LastError = ""
	mam.mu.Unlock()

	if err := mam.setupHealthChecking(); err != nil {
		return fmt.Errorf("restart: failed to rebuild health checkers: %w", err)
	}

	return mam.Start(ctx)
}

// GetRouter returns the HTTP router
func (mam *MultiAgentManager) GetRouter() *mux.Router {
	return mam.router
}

// GetConfig returns the multi-agent configuration
func (mam *MultiAgentManager) GetConfig() *MultiAgentConfig {
	return mam.config
}

// GetMetrics returns current metrics
func (mam *MultiAgentManager) GetMetrics() *MultiAgentMetrics {
	mam.metrics.mu.RLock()
	defer mam.metrics.mu.RUnlock()

	// Create a copy of metrics without copying the mutex
	metricsCopy := MultiAgentMetrics{
		TotalRequests: mam.metrics.TotalRequests,
		TotalErrors:   mam.metrics.TotalErrors,
		AgentMetrics:  make(map[string]*AgentMetrics),
		RoutingMetrics: &RoutingMetrics{
			RoutingDecisions: make(map[string]int64),
			DefaultRoutes:    mam.metrics.RoutingMetrics.DefaultRoutes,
			FailedRoutes:     mam.metrics.RoutingMetrics.FailedRoutes,
		},
		LastUpdated: mam.metrics.LastUpdated,
	}

	// Copy agent metrics
	for k, v := range mam.metrics.AgentMetrics {
		metricsCopy.AgentMetrics[k] = &AgentMetrics{
			RequestCount:   v.RequestCount,
			ErrorCount:     v.ErrorCount,
			AverageLatency: v.AverageLatency,
			LastRequest:    v.LastRequest,
			TotalLatency:   v.TotalLatency,
		}
	}

	// Copy routing decisions
	for k, v := range mam.metrics.RoutingMetrics.RoutingDecisions {
		metricsCopy.RoutingMetrics.RoutingDecisions[k] = v
	}

	return &metricsCopy
}

// GetDeploymentState returns a deep copy of the current deployment state.
//
// It used to return a shallow copy sharing the live AgentStates map and every
// *AgentState in it, so the "snapshot" kept changing under the caller and
// reading it raced with request handlers.
func (mam *MultiAgentManager) GetDeploymentState() *DeploymentState {
	mam.mu.RLock()
	defer mam.mu.RUnlock()

	return mam.deploymentState.Clone()
}

// LoadMultiAgentConfigFromFile loads multi-agent configuration from a file.
//
// The extension chooses the decoder; the result is validated before it is
// returned, so a caller never receives a config the manager would reject.
func LoadMultiAgentConfigFromFile(filename string) (*MultiAgentConfig, error) {
	path, err := filepath.Abs(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to get absolute path: %w", err)
	}

	content, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	var config MultiAgentConfig
	ext := strings.ToLower(filepath.Ext(filename))

	switch ext {
	case ".yaml", ".yml":
		if err := yaml.Unmarshal(content, &config); err != nil {
			return nil, fmt.Errorf("failed to parse YAML config: %w", err)
		}
	case ".json":
		if err := json.Unmarshal(content, &config); err != nil {
			return nil, fmt.Errorf("failed to parse JSON config: %w", err)
		}
	default:
		return nil, fmt.Errorf("unsupported config file format: %s", ext)
	}

	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("invalid configuration: %w", err)
	}

	return &config, nil
}
