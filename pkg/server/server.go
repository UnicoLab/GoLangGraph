// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package server

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	"github.com/gorilla/mux"
	"github.com/gorilla/websocket"
	"github.com/sirupsen/logrus"

	"github.com/UnicoLab/GoLangGraph/pkg/agent"
	"github.com/UnicoLab/GoLangGraph/pkg/core"
	"github.com/UnicoLab/GoLangGraph/pkg/llm"
	"github.com/UnicoLab/GoLangGraph/pkg/persistence"
	"github.com/UnicoLab/GoLangGraph/pkg/tools"
)

// ServerConfig represents server configuration
type ServerConfig struct {
	Host           string        `json:"host"`
	Port           int           `json:"port"`
	ReadTimeout    time.Duration `json:"read_timeout"`
	WriteTimeout   time.Duration `json:"write_timeout"`
	MaxHeaderBytes int           `json:"max_header_bytes"`
	EnableCORS     bool          `json:"enable_cors"`
	StaticDir      string        `json:"static_dir"`
	DevMode        bool          `json:"dev_mode"`
	LogLevel       string        `json:"log_level"`

	// Security controls authentication, allowed origins and request limits.
	// Nil falls back to DefaultSecurityConfig.
	Security *SecurityConfig `json:"security"`
}

// DefaultServerConfig returns default server configuration
func DefaultServerConfig() *ServerConfig {
	return &ServerConfig{
		Host:           "0.0.0.0",
		Port:           8080,
		ReadTimeout:    30 * time.Second,
		WriteTimeout:   30 * time.Second,
		MaxHeaderBytes: 1 << 20, // 1MB
		EnableCORS:     true,
		StaticDir:      "./static",
		DevMode:        false,
		LogLevel:       "info",
		Security:       DefaultSecurityConfig(),
	}
}

// Server represents the HTTP server
type Server struct {
	config   *ServerConfig
	router   *mux.Router
	server   *http.Server
	serverMu sync.RWMutex
	logger   *logrus.Logger
	upgrader websocket.Upgrader

	// Core components
	llmManager     *llm.ProviderManager
	toolRegistry   *tools.ToolRegistry
	agentManager   *AgentManager
	sessionManager *persistence.SessionManager

	graphManager *GraphManager
	checkpointer persistence.Checkpointer

	// Request accounting for the metrics endpoint.
	requestsTotal  atomic.Uint64
	requestsFailed atomic.Uint64
	startedAt      time.Time

	// WebSocket connections, keyed by resource ID then by connection, so that
	// several clients can observe the same agent or graph at once.
	wsConnections   map[string]map[*websocket.Conn]struct{}
	wsConnectionsMu sync.RWMutex

	// httpConnections records connection state so Stop can close sockets that
	// have been accepted but have not yet reached a request. net/http otherwise
	// gives StateNew connections a five-second grace period during Shutdown.
	httpConnections   map[net.Conn]http.ConnState
	httpConnectionsMu sync.Mutex
}

// NewServer creates a new server
func NewServer(config *ServerConfig) *Server {
	if config == nil {
		config = DefaultServerConfig()
	}

	if config.Security == nil {
		config.Security = DefaultSecurityConfig()
	}

	server := &Server{
		config:          config,
		router:          mux.NewRouter(),
		logger:          logrus.New(),
		graphManager:    NewGraphManager(),
		wsConnections:   make(map[string]map[*websocket.Conn]struct{}),
		httpConnections: make(map[net.Conn]http.ConnState),
		startedAt:       time.Now(),
	}

	// Reject WebSocket upgrades from origins the API does not allow. Accepting
	// every origin permits cross-site WebSocket hijacking: any page a user
	// visits could open a socket to this server and drive it as that user.
	server.upgrader = websocket.Upgrader{
		CheckOrigin: func(r *http.Request) bool {
			return config.Security.originAllowed(r.Header.Get("Origin"))
		},
	}

	// LogLevel was declared in the configuration and never read, so setting it
	// had no effect at all.
	if config.LogLevel != "" {
		if level, err := logrus.ParseLevel(config.LogLevel); err == nil {
			server.logger.SetLevel(level)
		} else {
			server.logger.WithField("log_level", config.LogLevel).
				Warn("Unrecognized log level; keeping the default")
		}
	}

	server.setupRoutes()
	return server
}

func (s *Server) trackHTTPConnection(conn net.Conn, state http.ConnState) {
	s.httpConnectionsMu.Lock()
	defer s.httpConnectionsMu.Unlock()
	if state == http.StateClosed || state == http.StateHijacked {
		delete(s.httpConnections, conn)
		return
	}
	s.httpConnections[conn] = state
}

// SetCheckpointer attaches the checkpointer used to serve thread history.
func (s *Server) SetCheckpointer(cp persistence.Checkpointer) {
	s.checkpointer = cp
}

// SetGraphManager replaces the graph manager.
func (s *Server) SetGraphManager(manager *GraphManager) {
	s.graphManager = manager
}

// GraphManager returns the server's graph manager, used to register graphs
// that should be listable, inspectable and executable over the API.
func (s *Server) GraphManager() *GraphManager {
	return s.graphManager
}

// registerWSConn records a live WebSocket connection for a resource.
func (s *Server) registerWSConn(id string, conn *websocket.Conn) {
	s.wsConnectionsMu.Lock()
	defer s.wsConnectionsMu.Unlock()
	if s.wsConnections[id] == nil {
		s.wsConnections[id] = make(map[*websocket.Conn]struct{})
	}
	s.wsConnections[id][conn] = struct{}{}
}

// unregisterWSConn removes a connection, dropping the resource entry when the
// last connection for it closes.
func (s *Server) unregisterWSConn(id string, conn *websocket.Conn) {
	s.wsConnectionsMu.Lock()
	defer s.wsConnectionsMu.Unlock()
	conns := s.wsConnections[id]
	if conns == nil {
		return
	}
	delete(conns, conn)
	if len(conns) == 0 {
		delete(s.wsConnections, id)
	}
}

// wsConnectionCount reports how many live connections a resource has.
func (s *Server) wsConnectionCount(id string) int {
	s.wsConnectionsMu.RLock()
	defer s.wsConnectionsMu.RUnlock()
	return len(s.wsConnections[id])
}

// SetLLMManager sets the LLM provider manager
func (s *Server) SetLLMManager(manager *llm.ProviderManager) {
	s.llmManager = manager
}

// SetToolRegistry sets the tool registry
func (s *Server) SetToolRegistry(registry *tools.ToolRegistry) {
	s.toolRegistry = registry
}

// SetAgentManager sets the agent manager
func (s *Server) SetAgentManager(manager *AgentManager) {
	s.agentManager = manager
}

// SetSessionManager sets the session manager
func (s *Server) SetSessionManager(manager *persistence.SessionManager) {
	s.sessionManager = manager
}

// setupRoutes sets up HTTP routes
func (s *Server) setupRoutes() {
	// Middleware. gorilla/mux wraps from the last registered to the first, so
	// the first registered is the outermost. Recovery goes first so a panic in
	// any later middleware or handler still produces a response rather than
	// dropping the connection, and authentication goes last so a rejected
	// request still carries the CORS and security headers a browser needs to
	// read the 401.
	s.router.Use(recoveryMiddleware(s.logger))
	s.router.Use(bodyLimitMiddleware(s.config.Security.maxBytes()))
	s.router.Use(securityHeadersMiddleware)

	if s.config.EnableCORS {
		s.router.Use(s.corsMiddleware)
	}

	s.router.Use(s.loggingMiddleware)
	s.router.Use(s.authMiddleware)

	// API routes
	api := s.router.PathPrefix("/api/v1").Subrouter()

	// Health check
	api.HandleFunc("/health", s.handleHealth).Methods("GET", "OPTIONS")

	// LLM providers
	api.HandleFunc("/providers", s.handleListProviders).Methods("GET")
	api.HandleFunc("/providers/{name}/models", s.handleGetProviderModels).Methods("GET")
	api.HandleFunc("/providers/{name}/health", s.handleProviderHealth).Methods("GET")

	// Agents
	api.HandleFunc("/agents", s.handleListAgents).Methods("GET")
	api.HandleFunc("/agents", s.handleCreateAgent).Methods("POST")
	api.HandleFunc("/agents/{id}", s.handleGetAgent).Methods("GET")
	api.HandleFunc("/agents/{id}", s.handleUpdateAgent).Methods("PUT")
	api.HandleFunc("/agents/{id}", s.handleDeleteAgent).Methods("DELETE")
	api.HandleFunc("/agents/{id}/execute", s.handleExecuteAgent).Methods("POST")
	api.HandleFunc("/agents/{id}/history", s.handleGetAgentHistory).Methods("GET")

	// Graphs
	api.HandleFunc("/graphs", s.handleListGraphs).Methods("GET")
	api.HandleFunc("/graphs/{id}", s.handleGetGraph).Methods("GET")
	api.HandleFunc("/graphs/{id}/topology", s.handleGetGraphTopology).Methods("GET")
	api.HandleFunc("/graphs/{id}/execute", s.handleExecuteGraph).Methods("POST")
	api.HandleFunc("/graphs/{id}/interrupt", s.handleInterruptGraph).Methods("POST")

	// Sessions and threads
	api.HandleFunc("/sessions", s.handleCreateSession).Methods("POST")
	api.HandleFunc("/sessions/{id}", s.handleGetSession).Methods("GET")
	api.HandleFunc("/threads", s.handleCreateThread).Methods("POST")
	api.HandleFunc("/threads/{id}", s.handleGetThread).Methods("GET")
	api.HandleFunc("/threads/{id}/checkpoints", s.handleListCheckpoints).Methods("GET")

	// Tools
	api.HandleFunc("/tools", s.handleListTools).Methods("GET")
	api.HandleFunc("/tools/{name}", s.handleGetTool).Methods("GET")

	// WebSocket endpoints
	api.HandleFunc("/ws/agents/{id}/stream", s.handleAgentWebSocket)
	api.HandleFunc("/ws/graphs/{id}/stream", s.handleGraphWebSocket)

	// Dev mode specific routes
	if s.config.DevMode {
		debug := s.router.PathPrefix("/debug").Subrouter()
		debug.HandleFunc("/", s.handleDebugDashboard).Methods("GET")
		debug.HandleFunc("/agents", s.handleDebugAgents).Methods("GET")
		debug.HandleFunc("/config", s.handleDebugConfig).Methods("GET")
		debug.HandleFunc("/logs", s.handleDebugLogs).Methods("GET")
		debug.HandleFunc("/metrics", s.handleDebugMetrics).Methods("GET")
		debug.HandleFunc("/reload", s.handleDebugReload).Methods("POST")

		playground := s.router.PathPrefix("/playground").Subrouter()
		playground.HandleFunc("/", s.handlePlaygroundDashboard).Methods("GET")
		playground.HandleFunc("/test", s.handlePlaygroundTest).Methods("POST")
		playground.HandleFunc("/agents/{id}/test", s.handlePlaygroundAgentTest).Methods("POST")
	}

	// Cross-origin preflight. Routes declare concrete methods, so an OPTIONS
	// request would otherwise fall through to 404 and the browser would block
	// every cross-origin call. This catch-all gives the CORS middleware a
	// matched route to run on; it must be registered before the static handler.
	s.router.PathPrefix("/").Methods(http.MethodOptions).HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	})

	// Static files for UI
	if s.config.StaticDir != "" {
		s.router.PathPrefix("/").Handler(http.FileServer(http.Dir(s.config.StaticDir)))
	}
}

// Start starts the server
func (s *Server) Start() error {
	httpServer := &http.Server{
		Addr:           fmt.Sprintf("%s:%d", s.config.Host, s.config.Port),
		Handler:        s.router,
		ReadTimeout:    s.config.ReadTimeout,
		WriteTimeout:   s.config.WriteTimeout,
		MaxHeaderBytes: s.config.MaxHeaderBytes,
		ConnState:      s.trackHTTPConnection,
	}
	s.serverMu.Lock()
	s.server = httpServer
	s.serverMu.Unlock()

	s.logger.WithFields(logrus.Fields{
		"host": s.config.Host,
		"port": s.config.Port,
	}).Info("Starting GoLangGraph server")

	if err := httpServer.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	return nil
}

// Stop stops the server
func (s *Server) Stop(ctx context.Context) error {
	s.logger.Info("Stopping GoLangGraph server")

	// Close live WebSocket connections so Shutdown is not blocked by hijacked
	// connections, which http.Server does not wait for or close itself.
	s.wsConnectionsMu.Lock()
	for id, conns := range s.wsConnections {
		for conn := range conns {
			_ = conn.Close()
		}
		delete(s.wsConnections, id)
	}
	s.wsConnectionsMu.Unlock()

	// Shutdown intentionally waits for active requests. A connection that has
	// not reached a request is not active work, but net/http treats StateNew as
	// active for five seconds; close those sockets so a stop/restart is prompt.
	s.httpConnectionsMu.Lock()
	newConnections := make([]net.Conn, 0)
	for conn, state := range s.httpConnections {
		if state == http.StateNew {
			newConnections = append(newConnections, conn)
		}
	}
	s.httpConnectionsMu.Unlock()
	for _, conn := range newConnections {
		_ = conn.Close()
	}

	s.serverMu.RLock()
	httpServer := s.server
	s.serverMu.RUnlock()
	if httpServer == nil {
		return nil
	}
	return httpServer.Shutdown(ctx)
}

// Middleware

func (s *Server) corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		origin := r.Header.Get("Origin")

		// Responses vary by Origin, so caches must not share them across origins.
		w.Header().Add("Vary", "Origin")

		if !s.config.Security.originAllowed(origin) {
			// Omit the CORS headers entirely; the browser then blocks the read.
			if r.Method == http.MethodOptions {
				w.WriteHeader(http.StatusForbidden)
				return
			}
			next.ServeHTTP(w, r)
			return
		}

		if allow := s.config.Security.corsOrigin(origin); allow != "" {
			w.Header().Set("Access-Control-Allow-Origin", allow)
			// Credentials are only meaningful for a specific origin, never "*".
			if allow != "*" {
				w.Header().Set("Access-Control-Allow-Credentials", "true")
			}
		}
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-API-Key")
		w.Header().Set("Access-Control-Max-Age", "600")

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusOK)
			return
		}

		next.ServeHTTP(w, r)
	})
}

// statusRecorder captures the response status so metrics can count failures.
type statusRecorder struct {
	http.ResponseWriter
	status int
}

func (r *statusRecorder) WriteHeader(status int) {
	r.status = status
	r.ResponseWriter.WriteHeader(status)
}

// Flush and Hijack are forwarded so streaming and WebSocket upgrades still work
// through the recorder.
func (r *statusRecorder) Flush() {
	if f, ok := r.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

func (r *statusRecorder) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	h, ok := r.ResponseWriter.(http.Hijacker)
	if !ok {
		return nil, nil, fmt.Errorf("response writer does not support hijacking")
	}
	return h.Hijack()
}

func (s *Server) loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()

		s.requestsTotal.Add(1)
		recorder := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
		w = recorder

		next.ServeHTTP(w, r)

		if recorder.status >= 400 {
			s.requestsFailed.Add(1)
		}

		s.logger.WithFields(logrus.Fields{
			"method":   r.Method,
			"path":     r.URL.Path,
			"duration": time.Since(start),
			"remote":   r.RemoteAddr,
		}).Info("HTTP request")
	})
}

// authMiddleware enforces API key authentication when the server is configured
// to require it. Preflight requests and configured public paths (health checks)
// bypass the check so probes and browsers keep working.
func (s *Server) authMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sec := s.config.Security
		if sec == nil || !sec.RequireAuth {
			next.ServeHTTP(w, r)
			return
		}
		if r.Method == http.MethodOptions || sec.isPublic(r.URL.Path) {
			next.ServeHTTP(w, r)
			return
		}

		if !sec.authorized(r.Header.Get("X-API-Key")) {
			s.logger.WithFields(logrus.Fields{
				"path":   r.URL.Path,
				"remote": r.RemoteAddr,
			}).Warn("Rejected unauthenticated request")
			s.writeError(w, http.StatusUnauthorized, "missing or invalid API key")
			return
		}

		next.ServeHTTP(w, r)
	})
}

// Health check handler
func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	health := map[string]interface{}{
		"status":    "healthy",
		"timestamp": time.Now(),
		"version":   "1.0.0",
	}

	// Check component health
	if s.llmManager != nil {
		ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
		defer cancel()

		providerHealth := s.llmManager.HealthCheck(ctx)
		health["providers"] = providerHealth
	}

	s.writeJSON(w, http.StatusOK, health)
}

// Provider handlers
func (s *Server) handleListProviders(w http.ResponseWriter, r *http.Request) {
	if s.llmManager == nil {
		s.writeError(w, http.StatusServiceUnavailable, "LLM manager not available")
		return
	}

	// Describe each provider rather than returning bare names, so clients can
	// show its type, endpoint and model without a second round trip.
	names := s.llmManager.ListProviders()
	sort.Strings(names)

	infos := make([]map[string]interface{}, 0, len(names))
	for _, name := range names {
		info := map[string]interface{}{"name": name}
		if provider, err := s.llmManager.GetProvider(name); err == nil && provider != nil {
			for key, value := range provider.GetConfig() {
				// Never expose credentials over the API.
				if key == "api_key" || key == "apiKey" {
					continue
				}
				info[key] = value
			}
		}
		infos = append(infos, info)
	}

	s.writeJSON(w, http.StatusOK, map[string]interface{}{
		"providers": infos,
	})
}

func (s *Server) handleGetProviderModels(w http.ResponseWriter, r *http.Request) {
	if s.llmManager == nil {
		s.writeError(w, http.StatusServiceUnavailable, "LLM manager not available")
		return
	}

	vars := mux.Vars(r)
	providerName := vars["name"]

	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()

	models, err := s.llmManager.GetProviderModels(ctx, providerName)
	if err != nil {
		s.writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	s.writeJSON(w, http.StatusOK, map[string]interface{}{
		"provider": providerName,
		"models":   models,
	})
}

func (s *Server) handleProviderHealth(w http.ResponseWriter, r *http.Request) {
	if s.llmManager == nil {
		s.writeError(w, http.StatusServiceUnavailable, "LLM manager not available")
		return
	}

	vars := mux.Vars(r)
	providerName := vars["name"]

	provider, err := s.llmManager.GetProvider(providerName)
	if err != nil {
		s.writeError(w, http.StatusNotFound, err.Error())
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()

	err = provider.IsHealthy(ctx)
	status := "healthy"
	if err != nil {
		status = "unhealthy"
	}

	s.writeJSON(w, http.StatusOK, map[string]interface{}{
		"provider": providerName,
		"status":   status,
		"error":    err,
	})
}

// Agent handlers
func (s *Server) handleListAgents(w http.ResponseWriter, r *http.Request) {
	if s.agentManager == nil {
		s.writeError(w, http.StatusServiceUnavailable, "Agent manager not available")
		return
	}

	// Return full configurations rather than bare IDs: clients such as
	// GoLangGraph Studio render an agent's name, type, model and provider from
	// this list, and a list of strings leaves every field undefined.
	ids := s.agentManager.ListAgents()
	configs := make([]*agent.AgentConfig, 0, len(ids))
	for _, id := range ids {
		if instance, ok := s.agentManager.GetAgent(id); ok {
			configs = append(configs, instance.GetConfig())
		}
	}
	sort.Slice(configs, func(i, j int) bool { return configs[i].ID < configs[j].ID })

	s.writeJSON(w, http.StatusOK, map[string]interface{}{
		"agents": configs,
	})
}

func (s *Server) handleCreateAgent(w http.ResponseWriter, r *http.Request) {
	if s.agentManager == nil {
		s.writeError(w, http.StatusServiceUnavailable, "Agent manager not available")
		return
	}

	var config agent.AgentConfig
	if err := json.NewDecoder(r.Body).Decode(&config); err != nil {
		s.writeError(w, http.StatusBadRequest, "Invalid request body")
		return
	}

	agentInstance, err := s.agentManager.CreateAgent(&config)
	if err != nil {
		s.writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	s.writeJSON(w, http.StatusCreated, map[string]interface{}{
		"agent": agentInstance.GetConfig(),
	})
}

func (s *Server) handleGetAgent(w http.ResponseWriter, r *http.Request) {
	if s.agentManager == nil {
		s.writeError(w, http.StatusServiceUnavailable, "Agent manager not available")
		return
	}

	vars := mux.Vars(r)
	agentID := vars["id"]

	agentInstance, exists := s.agentManager.GetAgent(agentID)
	if !exists {
		s.writeError(w, http.StatusNotFound, "Agent not found")
		return
	}

	s.writeJSON(w, http.StatusOK, map[string]interface{}{
		"agent": agentInstance.GetConfig(),
	})
}

func (s *Server) handleExecuteAgent(w http.ResponseWriter, r *http.Request) {
	if s.agentManager == nil {
		s.writeError(w, http.StatusServiceUnavailable, "Agent manager not available")
		return
	}

	vars := mux.Vars(r)
	agentID := vars["id"]

	var request struct {
		Input  string `json:"input"`
		Stream bool   `json:"stream"`
	}

	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		s.writeError(w, http.StatusBadRequest, "Invalid request body")
		return
	}

	agentInstance, exists := s.agentManager.GetAgent(agentID)
	if !exists {
		s.writeError(w, http.StatusNotFound, "Agent not found")
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Minute)
	defer cancel()

	execution, err := agentInstance.Execute(ctx, request.Input)
	if err != nil {
		s.writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	s.writeJSON(w, http.StatusOK, map[string]interface{}{
		"execution": execution,
	})
}

func (s *Server) handleUpdateAgent(w http.ResponseWriter, r *http.Request) {
	if s.agentManager == nil {
		s.writeError(w, http.StatusServiceUnavailable, "Agent manager not available")
		return
	}

	vars := mux.Vars(r)
	agentID := vars["id"]

	var config agent.AgentConfig
	if err := json.NewDecoder(r.Body).Decode(&config); err != nil {
		s.writeError(w, http.StatusBadRequest, "Invalid request body")
		return
	}

	// Ensure the ID matches
	config.ID = agentID

	agentInstance, err := s.agentManager.CreateAgent(&config)
	if err != nil {
		s.writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	s.writeJSON(w, http.StatusOK, map[string]interface{}{
		"agent": agentInstance.GetConfig(),
	})
}

func (s *Server) handleDeleteAgent(w http.ResponseWriter, r *http.Request) {
	if s.agentManager == nil {
		s.writeError(w, http.StatusServiceUnavailable, "Agent manager not available")
		return
	}

	vars := mux.Vars(r)
	agentID := vars["id"]

	s.agentManager.DeleteAgent(agentID)
	s.writeJSON(w, http.StatusOK, map[string]interface{}{
		"message": "Agent deleted successfully",
	})
}

func (s *Server) handleGetAgentHistory(w http.ResponseWriter, r *http.Request) {
	if s.agentManager == nil {
		s.writeError(w, http.StatusServiceUnavailable, "Agent manager not available")
		return
	}

	vars := mux.Vars(r)
	agentID := vars["id"]

	agentInstance, exists := s.agentManager.GetAgent(agentID)
	if !exists {
		s.writeError(w, http.StatusNotFound, "Agent not found")
		return
	}

	// Get execution history from agent
	history := agentInstance.GetExecutionHistory()
	s.writeJSON(w, http.StatusOK, map[string]interface{}{
		"agent_id": agentID,
		"history":  history,
	})
}

func (s *Server) handleListGraphs(w http.ResponseWriter, r *http.Request) {
	summaries := []GraphSummaryView{}
	if s.graphManager != nil {
		for _, id := range s.graphManager.List() {
			if g, ok := s.graphManager.Get(id); ok {
				summaries = append(summaries, summariseGraph(id, g))
			}
		}
	}
	s.writeJSON(w, http.StatusOK, map[string]interface{}{
		"graphs": summaries,
	})
}

func (s *Server) handleGetGraph(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	graphID := vars["id"]

	graph, exists := s.lookupGraph(graphID)
	if !exists {
		s.writeError(w, http.StatusNotFound, "graph not found")
		return
	}

	topology := describeTopology(graph)
	s.writeJSON(w, http.StatusOK, map[string]interface{}{
		"graph_id": graphID,
		"graph":    summariseGraph(graphID, graph),
		"nodes":    topology.Nodes,
		"edges":    topology.Edges,
	})
}

func (s *Server) handleGetGraphTopology(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	graphID := vars["id"]

	graph, exists := s.lookupGraph(graphID)
	if !exists {
		s.writeError(w, http.StatusNotFound, "graph not found")
		return
	}

	s.writeJSON(w, http.StatusOK, map[string]interface{}{
		"graph_id": graphID,
		"topology": describeTopology(graph),
	})
}

func (s *Server) handleExecuteGraph(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	graphID := vars["id"]

	graph, exists := s.lookupGraph(graphID)
	if !exists {
		s.writeError(w, http.StatusNotFound, "graph not found")
		return
	}

	var request struct {
		Input    string                     `json:"input"`
		State    map[string]core.StateValue `json:"state"`
		ThreadID string                     `json:"thread_id"`
	}

	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		s.writeError(w, http.StatusBadRequest, "invalid request body: "+err.Error())
		return
	}

	steps := make(chan *core.ExecutionResult, 256)
	collected := make([]ExecutionStepView, 0, 8)
	drained := make(chan struct{})
	go func() {
		defer close(drained)
		for result := range steps {
			collected = append(collected, describeStep(result))
		}
	}()

	finalState, err := graph.ExecuteWithOptions(r.Context(), buildInitialState(request.Input, request.State), &core.ExecuteOptions{
		ThreadID: request.ThreadID,
		Stream:   steps,
	})
	<-drained

	response := map[string]interface{}{
		"graph_id": graphID,
		"steps":    collected,
	}
	if finalState != nil {
		response["state"] = finalState.GetAll()
	}

	if err != nil {
		response["error"] = err.Error()

		var ie *core.InterruptError
		switch {
		case errors.As(err, &ie):
			// A pause is a normal, resumable outcome, not a server error.
			response["status"] = "interrupted"
			response["interrupt"] = map[string]interface{}{
				"node_id": ie.NodeID,
				"before":  ie.Before,
				"step":    ie.Step,
			}
			s.writeJSON(w, http.StatusOK, response)
		case errors.Is(err, context.Canceled), errors.Is(err, context.DeadlineExceeded):
			response["status"] = "canceled"
			s.writeJSON(w, http.StatusRequestTimeout, response)
		case errors.Is(err, core.ErrGraphInvalid):
			response["status"] = "invalid"
			s.writeJSON(w, http.StatusBadRequest, response)
		default:
			response["status"] = "failed"
			s.writeJSON(w, http.StatusInternalServerError, response)
		}
		return
	}

	response["status"] = "completed"
	s.writeJSON(w, http.StatusOK, response)
}

func (s *Server) handleInterruptGraph(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	graphID := vars["id"]

	graph, exists := s.lookupGraph(graphID)
	if !exists {
		s.writeError(w, http.StatusNotFound, "graph not found")
		return
	}

	graph.Interrupt()
	s.writeJSON(w, http.StatusOK, map[string]interface{}{
		"graph_id": graphID,
		"status":   "interrupted",
	})
}

func (s *Server) handleCreateSession(w http.ResponseWriter, r *http.Request) {
	if s.sessionManager == nil {
		s.writeError(w, http.StatusServiceUnavailable, "Session manager not available")
		return
	}

	var request struct {
		UserID string `json:"user_id"`
	}

	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		s.writeError(w, http.StatusBadRequest, "Invalid request body")
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()

	session := &persistence.Session{
		ID:        uuid.New().String(),
		UserID:    request.UserID,
		Metadata:  make(map[string]interface{}),
		CreatedAt: time.Now(),
	}

	err := s.sessionManager.CreateSession(ctx, session)
	if err != nil {
		if errors.Is(err, persistence.ErrSessionStoreUnavailable) {
			s.writeError(w, http.StatusServiceUnavailable, "Session storage is not configured")
			return
		}
		s.writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	s.writeJSON(w, http.StatusCreated, map[string]interface{}{
		"session": session,
	})
}

func (s *Server) handleGetSession(w http.ResponseWriter, r *http.Request) {
	if s.sessionManager == nil {
		s.writeError(w, http.StatusServiceUnavailable, "Session manager not available")
		return
	}

	vars := mux.Vars(r)
	sessionID := vars["id"]

	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()

	session, err := s.sessionManager.GetSession(ctx, sessionID)
	if err != nil {
		if errors.Is(err, persistence.ErrSessionStoreUnavailable) {
			s.writeError(w, http.StatusServiceUnavailable, "Session storage is not configured")
			return
		}
		s.writeError(w, http.StatusNotFound, err.Error())
		return
	}

	s.writeJSON(w, http.StatusOK, map[string]interface{}{
		"session": session,
	})
}

func (s *Server) handleCreateThread(w http.ResponseWriter, r *http.Request) {
	if s.sessionManager == nil {
		s.writeError(w, http.StatusServiceUnavailable, "Session manager not available")
		return
	}

	var request struct {
		SessionID string `json:"session_id"`
	}

	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		s.writeError(w, http.StatusBadRequest, "Invalid request body")
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()

	thread := &persistence.Thread{
		ID:        uuid.New().String(),
		Name:      fmt.Sprintf("Thread-%s", time.Now().Format("2006-01-02-15-04-05")),
		Metadata:  make(map[string]interface{}),
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}

	err := s.sessionManager.CreateThread(ctx, thread)
	if err != nil {
		if errors.Is(err, persistence.ErrSessionStoreUnavailable) {
			s.writeError(w, http.StatusServiceUnavailable, "Thread storage is not configured")
			return
		}
		s.writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	s.writeJSON(w, http.StatusCreated, map[string]interface{}{
		"thread": thread,
	})
}

func (s *Server) handleGetThread(w http.ResponseWriter, r *http.Request) {
	if s.sessionManager == nil {
		s.writeError(w, http.StatusServiceUnavailable, "Session manager not available")
		return
	}

	vars := mux.Vars(r)
	threadID := vars["id"]

	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()

	thread, err := s.sessionManager.GetThread(ctx, threadID)
	if err != nil {
		if errors.Is(err, persistence.ErrSessionStoreUnavailable) {
			s.writeError(w, http.StatusServiceUnavailable, "Thread storage is not configured")
			return
		}
		s.writeError(w, http.StatusNotFound, err.Error())
		return
	}

	s.writeJSON(w, http.StatusOK, map[string]interface{}{
		"thread": thread,
	})
}

func (s *Server) handleListCheckpoints(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	threadID := vars["id"]

	// Previously this always reported an empty list, so a thread's history was
	// invisible even when checkpoints existed.
	if s.checkpointer == nil {
		s.writeError(w, http.StatusServiceUnavailable, "no checkpointer is configured")
		return
	}

	metadata, err := s.checkpointer.List(r.Context(), threadID)
	if err != nil {
		s.writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if metadata == nil {
		metadata = []*persistence.CheckpointMetadata{}
	}

	// Oldest first, so a client can replay a thread in order.
	sort.Slice(metadata, func(i, j int) bool {
		if metadata[i].StepID != metadata[j].StepID {
			return metadata[i].StepID < metadata[j].StepID
		}
		return metadata[i].CreatedAt.Before(metadata[j].CreatedAt)
	})

	s.writeJSON(w, http.StatusOK, map[string]interface{}{
		"thread_id":   threadID,
		"checkpoints": metadata,
	})
}

func (s *Server) handleListTools(w http.ResponseWriter, r *http.Request) {
	if s.toolRegistry == nil {
		s.writeError(w, http.StatusServiceUnavailable, "Tool registry not available")
		return
	}

	tools := s.toolRegistry.ListTools()
	s.writeJSON(w, http.StatusOK, map[string]interface{}{
		"tools": tools,
	})
}

func (s *Server) handleGetTool(w http.ResponseWriter, r *http.Request) {
	if s.toolRegistry == nil {
		s.writeError(w, http.StatusServiceUnavailable, "Tool registry not available")
		return
	}

	vars := mux.Vars(r)
	toolName := vars["name"]

	tool, exists := s.toolRegistry.GetTool(toolName)
	if !exists {
		s.writeError(w, http.StatusNotFound, "Tool not found")
		return
	}

	s.writeJSON(w, http.StatusOK, map[string]interface{}{
		"tool": tool,
	})
}

// handleGraphWebSocket streams a graph execution to a client.
//
// Each connection gets its own execution context, canceled when the client
// disconnects, so a closed tab cannot leave a graph running forever. All writes
// go through a serializing writer because the read loop and the streaming
// goroutine write concurrently.
func (s *Server) handleGraphWebSocket(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	graphID := vars["id"]

	conn, err := s.upgrader.Upgrade(w, r, nil)
	if err != nil {
		s.logger.WithError(err).Error("Failed to upgrade WebSocket")
		return
	}
	defer func() { _ = conn.Close() }()

	s.registerWSConn(graphID, conn)
	defer s.unregisterWSConn(graphID, conn)

	writer := newWSWriter(conn)

	// Canceled when this connection goes away.
	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()

	var running sync.WaitGroup
	defer running.Wait()

	for {
		var message struct {
			Type  string                     `json:"type"`
			Input string                     `json:"input"`
			State map[string]core.StateValue `json:"state"`
		}

		if err := conn.ReadJSON(&message); err != nil {
			if !isFatalWSError(err) {
				_ = writer.WriteJSON(map[string]interface{}{
					"type": "error", "graph_id": graphID,
					"error": "invalid message: " + err.Error(), "timestamp": time.Now(),
				})
				continue
			}
			if !websocket.IsCloseError(err, websocket.CloseNormalClosure, websocket.CloseGoingAway) {
				s.logger.WithError(err).Debug("WebSocket read ended")
			}
			cancel()
			break
		}

		switch message.Type {
		case "execute":
			graph, exists := s.lookupGraph(graphID)
			if !exists {
				_ = writer.WriteJSON(map[string]interface{}{
					"type": "error", "graph_id": graphID,
					"error": "graph not found", "timestamp": time.Now(),
				})
				continue
			}
			running.Add(1)
			go func() {
				defer running.Done()
				s.streamGraphExecution(ctx, writer, graphID, graph, message.Input, message.State)
			}()

		case "interrupt":
			if graph, exists := s.lookupGraph(graphID); exists {
				graph.Interrupt()
				_ = writer.WriteJSON(map[string]interface{}{
					"type": "interrupted", "graph_id": graphID, "timestamp": time.Now(),
				})
			}

		case "ping":
			_ = writer.WriteJSON(map[string]interface{}{"type": "pong", "timestamp": time.Now()})

		default:
			_ = writer.WriteJSON(map[string]interface{}{
				"type": "error", "error": "unknown message type: " + message.Type,
				"timestamp": time.Now(),
			})
		}
	}
}

// isFatalWSError reports whether a read error means the connection is gone, as
// opposed to a single malformed message. A client that sends one bad frame
// should get an error back and keep its session, not be disconnected.
func isFatalWSError(err error) bool {
	if err == nil {
		return false
	}
	var syntaxErr *json.SyntaxError
	var typeErr *json.UnmarshalTypeError
	if errors.As(err, &syntaxErr) || errors.As(err, &typeErr) {
		// The frame was received in full; only its contents were unusable.
		return false
	}
	return true
}

// lookupGraph resolves a graph by ID.
//
// Registered graphs win, then the execution graph of an agent with that ID.
// Clients such as Studio request a topology using the agent's ID, so without
// the fallback the graph view of every agent would be empty.
func (s *Server) lookupGraph(id string) (*core.Graph, bool) {
	if s.graphManager != nil {
		if g, ok := s.graphManager.Get(id); ok {
			return g, true
		}
	}
	if s.agentManager != nil {
		if instance, ok := s.agentManager.GetAgent(id); ok {
			if g := instance.GetGraph(); g != nil {
				return g, true
			}
		}
	}
	return nil, false
}

// buildInitialState turns a WebSocket or HTTP request payload into a state.
func buildInitialState(input string, data map[string]core.StateValue) *core.BaseState {
	state := core.NewBaseState()
	for k, v := range data {
		state.Set(k, v)
	}
	if input != "" {
		state.Set("input", input)
	}
	return state
}

// streamGraphExecution runs a graph and emits one message per node, then a
// terminal message. It never writes to the socket directly.
func (s *Server) streamGraphExecution(ctx context.Context, writer *wsWriter, graphID string, graph *core.Graph, input string, data map[string]core.StateValue) {
	_ = writer.WriteJSON(map[string]interface{}{
		"type": "start", "graph_id": graphID, "timestamp": time.Now(),
	})

	steps := make(chan *core.ExecutionResult, 64)
	done := make(chan struct{})

	go func() {
		defer close(done)
		for result := range steps {
			if err := writer.WriteJSON(map[string]interface{}{
				"type": "step", "graph_id": graphID,
				"step": describeStep(result), "timestamp": time.Now(),
			}); err != nil {
				// The client is gone; drain the rest so the run is not blocked.
				for range steps {
				}
				return
			}
		}
	}()

	finalState, err := graph.ExecuteWithOptions(ctx, buildInitialState(input, data), &core.ExecuteOptions{
		Stream: steps,
	})
	<-done

	if err != nil {
		payload := map[string]interface{}{
			"type": "error", "graph_id": graphID,
			"error": err.Error(), "timestamp": time.Now(),
		}
		var ie *core.InterruptError
		if errors.As(err, &ie) {
			payload["type"] = "interrupt"
			payload["node_id"] = ie.NodeID
			payload["before"] = ie.Before
			payload["step"] = ie.Step
		}
		if finalState != nil {
			payload["state"] = finalState.GetAll()
		}
		_ = writer.WriteJSON(payload)
		return
	}

	result := map[string]interface{}{
		"type": "complete", "graph_id": graphID, "timestamp": time.Now(),
	}
	if finalState != nil {
		result["state"] = finalState.GetAll()
	}
	_ = writer.WriteJSON(result)
}

// WebSocket handlers
func (s *Server) handleAgentWebSocket(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	agentID := vars["id"]

	conn, err := s.upgrader.Upgrade(w, r, nil)
	if err != nil {
		s.logger.WithError(err).Error("Failed to upgrade WebSocket")
		return
	}
	defer func() { _ = conn.Close() }()

	s.registerWSConn(agentID, conn)
	defer s.unregisterWSConn(agentID, conn)

	writer := newWSWriter(conn)

	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()

	var running sync.WaitGroup
	defer running.Wait()

	// Handle WebSocket messages
	for {
		var message struct {
			Type  string `json:"type"`
			Input string `json:"input"`
		}

		if err := conn.ReadJSON(&message); err != nil {
			if !isFatalWSError(err) {
				_ = writer.WriteJSON(map[string]interface{}{
					"type": "error", "error": "invalid message: " + err.Error(), "timestamp": time.Now(),
				})
				continue
			}
			if !websocket.IsCloseError(err, websocket.CloseNormalClosure, websocket.CloseGoingAway) {
				s.logger.WithError(err).Debug("WebSocket read ended")
			}
			cancel()
			break
		}

		switch message.Type {
		case "execute":
			if s.agentManager == nil {
				_ = writer.WriteJSON(map[string]interface{}{
					"type": "error", "error": "agent manager not available", "timestamp": time.Now(),
				})
				continue
			}
			agentInstance, exists := s.agentManager.GetAgent(agentID)
			if !exists {
				_ = writer.WriteJSON(map[string]interface{}{
					"type": "error", "error": "agent not found", "timestamp": time.Now(),
				})
				continue
			}
			running.Add(1)
			go func(input string) {
				defer running.Done()
				s.streamAgentExecution(ctx, writer, agentInstance, input)
			}(message.Input)

		case "ping":
			_ = writer.WriteJSON(map[string]interface{}{"type": "pong", "timestamp": time.Now()})

		default:
			_ = writer.WriteJSON(map[string]interface{}{
				"type": "error", "error": "unknown message type: " + message.Type, "timestamp": time.Now(),
			})
		}
	}
}

// streamAgentExecution runs an agent for a WebSocket client. The context is the
// connection's, so a disconnect cancels the run instead of leaving it (and any
// provider calls it makes) running unattended.
func (s *Server) streamAgentExecution(ctx context.Context, writer *wsWriter, agentInstance agent.Agent, input string) {
	_ = writer.WriteJSON(map[string]interface{}{
		"type":      "start",
		"timestamp": time.Now(),
	})

	execution, err := agentInstance.Execute(ctx, input)
	if err != nil {
		_ = writer.WriteJSON(map[string]interface{}{
			"type":      "error",
			"error":     err.Error(),
			"timestamp": time.Now(),
		})
		return
	}

	_ = writer.WriteJSON(map[string]interface{}{
		"type":      "result",
		"execution": execution,
		"timestamp": time.Now(),
	})
}

// Utility functions
func (s *Server) writeJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(data); err != nil {
		// The status line is already written, so the response cannot be
		// changed; record the failure instead of discarding it.
		s.logger.WithError(err).Error("Failed to encode JSON response")
	}
}

func (s *Server) writeError(w http.ResponseWriter, status int, message string) {
	s.writeJSON(w, status, map[string]interface{}{
		"error":     message,
		"timestamp": time.Now(),
	})
}

// Dev mode handlers
func (s *Server) handleDebugDashboard(w http.ResponseWriter, r *http.Request) {
	dashboardHTML := `
<!DOCTYPE html>
<html>
<head>
    <title>GoLangGraph Debug Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .card { border: 1px solid #ddd; padding: 20px; margin: 10px 0; border-radius: 5px; }
        .nav { margin-bottom: 20px; }
        .nav a { margin-right: 15px; text-decoration: none; color: #0066cc; }
        .nav a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <h1>GoLangGraph Debug Dashboard</h1>
    <div class="nav">
        <a href="/debug/agents">Agents</a>
        <a href="/debug/config">Configuration</a>
        <a href="/debug/logs">Logs</a>
        <a href="/debug/metrics">Metrics</a>
        <a href="/playground">Playground</a>
    </div>
    <div class="card">
        <h3>System Status</h3>
        <p>Development mode is active</p>
        <p>Server uptime: <span id="uptime">Loading...</span></p>
    </div>
    <div class="card">
        <h3>Quick Actions</h3>
        <button onclick="fetch('/debug/reload', {method: 'POST'}).then(r => r.json()).then(d => alert(JSON.stringify(d)))">Reload Configuration</button>
    </div>
</body>
</html>
`
	w.Header().Set("Content-Type", "text/html")
	_, _ = w.Write([]byte(dashboardHTML))
}

func (s *Server) handleDebugAgents(w http.ResponseWriter, r *http.Request) {
	if s.agentManager == nil {
		s.writeError(w, http.StatusServiceUnavailable, "Agent manager not available")
		return
	}

	agents := s.agentManager.ListAgents()
	var agentDetails []map[string]interface{}

	for _, agentID := range agents {
		if agentInstance, exists := s.agentManager.GetAgent(agentID); exists {
			config := agentInstance.GetConfig()
			agentDetails = append(agentDetails, map[string]interface{}{
				"id":     agentID,
				"config": config,
				"graph":  agentInstance.GetGraph(),
			})
		}
	}

	s.writeJSON(w, http.StatusOK, map[string]interface{}{
		"agents": agentDetails,
		"count":  len(agents),
	})
}

func (s *Server) handleDebugConfig(w http.ResponseWriter, r *http.Request) {
	s.writeJSON(w, http.StatusOK, map[string]interface{}{
		"server_config": s.config,
		"dev_mode":      s.config.DevMode,
	})
}

func (s *Server) handleDebugLogs(w http.ResponseWriter, r *http.Request) {
	// No log store is wired up. Returning a fabricated entry would suggest logs
	// are being served when none are; say so instead.
	s.writeJSON(w, http.StatusNotImplemented, map[string]interface{}{
		"error": "no log store is configured; logs are written to the process logger",
		"logs":  []map[string]interface{}{},
	})
}

func (s *Server) handleDebugMetrics(w http.ResponseWriter, r *http.Request) {
	// These were previously hardcoded, and the agent count dereferenced a nil
	// manager while the WebSocket count read a map without its mutex.
	agentsActive := 0
	if s.agentManager != nil {
		agentsActive = len(s.agentManager.ListAgents())
	}

	s.wsConnectionsMu.RLock()
	wsConnections := 0
	for _, conns := range s.wsConnections {
		wsConnections += len(conns)
	}
	s.wsConnectionsMu.RUnlock()

	graphs := 0
	if s.graphManager != nil {
		graphs = len(s.graphManager.List())
	}

	var mem runtime.MemStats
	runtime.ReadMemStats(&mem)

	s.writeJSON(w, http.StatusOK, map[string]interface{}{
		"metrics": map[string]interface{}{
			"requests_total":        s.requestsTotal.Load(),
			"requests_failed":       s.requestsFailed.Load(),
			"agents_active":         agentsActive,
			"graphs_registered":     graphs,
			"websocket_connections": wsConnections,
			"goroutines":            runtime.NumGoroutine(),
			"memory_alloc_bytes":    mem.Alloc,
			"memory_sys_bytes":      mem.Sys,
			"gc_cycles":             mem.NumGC,
			"uptime_seconds":        int64(time.Since(s.startedAt).Seconds()),
		},
	})
}

func (s *Server) handleDebugReload(w http.ResponseWriter, r *http.Request) {
	// This reported "Configuration reloaded successfully" without reloading
	// anything, which would lead an operator to believe a change had taken
	// effect. Report the truth until reloading is actually implemented.
	s.writeJSON(w, http.StatusNotImplemented, map[string]interface{}{
		"error":     "configuration reload is not supported; restart the server to apply changes",
		"timestamp": time.Now().Format(time.RFC3339),
	})
}

func (s *Server) handlePlaygroundDashboard(w http.ResponseWriter, r *http.Request) {
	playgroundHTML := `
<!DOCTYPE html>
<html>
<head>
    <title>GoLangGraph Playground</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .panel { border: 1px solid #ddd; padding: 20px; margin: 10px 0; border-radius: 5px; }
        textarea { width: 100%; height: 100px; }
        button { padding: 10px 20px; margin: 5px; background: #0066cc; color: white; border: none; border-radius: 3px; cursor: pointer; }
        button:hover { background: #0052a3; }
        .output { background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 3px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>GoLangGraph Playground</h1>

        <div class="panel">
            <h3>Test Agent</h3>
            <textarea id="input" placeholder="Enter your message here..."></textarea>
            <br>
            <button onclick="testAgent()">Test Agent</button>
            <div id="output" class="output"></div>
        </div>

        <div class="panel">
            <h3>Available Agents</h3>
            <div id="agents">Loading...</div>
        </div>
    </div>

    <script>
        async function testAgent() {
            const input = document.getElementById('input').value;
            const output = document.getElementById('output');

            try {
                const response = await fetch('/playground/test', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ input: input })
                });
                const result = await response.json();
                output.innerHTML = '<pre>' + JSON.stringify(result, null, 2) + '</pre>';
            } catch (error) {
                output.innerHTML = '<pre>Error: ' + error.message + '</pre>';
            }
        }

        // Load agents
        fetch('/api/v1/agents')
            .then(r => r.json())
            .then(data => {
                const agentsDiv = document.getElementById('agents');
                if (data.agents && data.agents.length > 0) {
                    agentsDiv.innerHTML = data.agents.map(agent =>
                        '<div>• ' + agent + '</div>'
                    ).join('');
                } else {
                    agentsDiv.innerHTML = 'No agents available';
                }
            });
    </script>
</body>
</html>
`
	w.Header().Set("Content-Type", "text/html")
	_, _ = w.Write([]byte(playgroundHTML))
}

func (s *Server) handlePlaygroundTest(w http.ResponseWriter, r *http.Request) {
	var request struct {
		Input string `json:"input"`
	}

	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		s.writeError(w, http.StatusBadRequest, "Invalid request body")
		return
	}

	// Test with the first available agent
	if s.agentManager == nil {
		s.writeError(w, http.StatusServiceUnavailable, "Agent manager not available")
		return
	}

	agents := s.agentManager.ListAgents()
	if len(agents) == 0 {
		s.writeError(w, http.StatusNotFound, "No agents available")
		return
	}

	agentInstance, exists := s.agentManager.GetAgent(agents[0])
	if !exists {
		s.writeError(w, http.StatusNotFound, "Agent not found")
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 2*time.Minute)
	defer cancel()

	execution, err := agentInstance.Execute(ctx, request.Input)
	if err != nil {
		s.writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	s.writeJSON(w, http.StatusOK, map[string]interface{}{
		"agent_id":  agents[0],
		"input":     request.Input,
		"execution": execution,
		"timestamp": time.Now().Format(time.RFC3339),
	})
}

func (s *Server) handlePlaygroundAgentTest(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	agentID := vars["id"]

	var request struct {
		Input string `json:"input"`
	}

	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		s.writeError(w, http.StatusBadRequest, "Invalid request body")
		return
	}

	if s.agentManager == nil {
		s.writeError(w, http.StatusServiceUnavailable, "Agent manager not available")
		return
	}

	agentInstance, exists := s.agentManager.GetAgent(agentID)
	if !exists {
		s.writeError(w, http.StatusNotFound, "Agent not found")
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 2*time.Minute)
	defer cancel()

	execution, err := agentInstance.Execute(ctx, request.Input)
	if err != nil {
		s.writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	s.writeJSON(w, http.StatusOK, map[string]interface{}{
		"agent_id":  agentID,
		"input":     request.Input,
		"execution": execution,
		"timestamp": time.Now().Format(time.RFC3339),
	})
}

// AgentManager manages multiple agents
type AgentManager struct {
	agents       map[string]agent.Agent
	llmManager   *llm.ProviderManager
	toolRegistry *tools.ToolRegistry
	mu           sync.RWMutex
}

// NewAgentManager creates a new agent manager
func NewAgentManager(llmManager *llm.ProviderManager, toolRegistry *tools.ToolRegistry) *AgentManager {
	return &AgentManager{
		agents:       make(map[string]agent.Agent),
		llmManager:   llmManager,
		toolRegistry: toolRegistry,
	}
}

// CreateAgent creates a new agent
func (am *AgentManager) CreateAgent(config *agent.AgentConfig) (agent.Agent, error) {
	am.mu.Lock()
	defer am.mu.Unlock()

	agentInstance := agent.NewAgent(config, am.llmManager, am.toolRegistry)
	am.agents[config.ID] = agentInstance

	return agentInstance, nil
}

// ReplaceAgents atomically replaces the agents managed by am. The replacement
// is built and validated before the live map is swapped, so a malformed reload
// leaves the currently serving agents available.
func (am *AgentManager) ReplaceAgents(configs []*agent.AgentConfig) error {
	replacement := make(map[string]agent.Agent, len(configs))
	for _, config := range configs {
		if config == nil {
			return fmt.Errorf("agent configuration is nil")
		}
		if err := config.Validate(); err != nil {
			return fmt.Errorf("invalid agent %q: %w", config.ID, err)
		}
		if config.ID == "" {
			return fmt.Errorf("agent %q has an empty id", config.Name)
		}
		if _, exists := replacement[config.ID]; exists {
			return fmt.Errorf("duplicate agent id %q", config.ID)
		}
		replacement[config.ID] = agent.NewAgent(config, am.llmManager, am.toolRegistry)
	}

	am.mu.Lock()
	am.agents = replacement
	am.mu.Unlock()
	return nil
}

// GetAgent retrieves an agent by ID
func (am *AgentManager) GetAgent(id string) (agent.Agent, bool) {
	am.mu.RLock()
	defer am.mu.RUnlock()

	agentInstance, exists := am.agents[id]
	return agentInstance, exists
}

// ListAgents returns all agent IDs
func (am *AgentManager) ListAgents() []string {
	am.mu.RLock()
	defer am.mu.RUnlock()

	ids := make([]string, 0, len(am.agents))
	for id := range am.agents {
		ids = append(ids, id)
	}
	return ids
}

// DeleteAgent removes an agent
func (am *AgentManager) DeleteAgent(id string) {
	am.mu.Lock()
	defer am.mu.Unlock()

	delete(am.agents, id)
}
