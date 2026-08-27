// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package server

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/mux"
	"github.com/sirupsen/logrus"

	"github.com/UnicoLab/GoLangGraph/pkg/agent"
	"github.com/UnicoLab/GoLangGraph/pkg/llm"
	"github.com/UnicoLab/GoLangGraph/pkg/tools"
)

// AutoServer automatically generates REST endpoints for agents
type AutoServer struct {
	registry     *agent.AgentRegistry
	llmManager   *llm.ProviderManager
	toolRegistry *tools.ToolRegistry
	router       *mux.Router
	config       *AutoServerConfig
	logger       *logrus.Logger

	// Dynamic agent instances
	agentInstances map[string]agent.Agent
	agentMetadata  map[string]map[string]interface{}

	// Metrics tracking
	startTime time.Time
	// requestCount is incremented from every request goroutine, so it must be
	// accessed atomically; a plain int64 here was a data race under any
	// concurrent load, which is to say always.
	requestCount atomic.Int64
	// agentMetrics stores execution statistics per agent. sync.Map allows a
	// request handler to update its own agent's counters without serializing
	// unrelated agents or racing with endpoint generation.
	agentMetrics sync.Map // map[string]*agentExecutionMetrics

	// agentsMu guards agentInstances and agentMetadata. GenerateEndpoints
	// writes them and every handler reads them, and the public API permits
	// registering an agent and regenerating after the server is serving.
	agentsMu sync.RWMutex

	// started records whether Start has been called, so endpoints cannot be
	// regenerated underneath live traffic.
	started atomic.Bool

	// mu guards boundAddr.
	mu sync.Mutex
	// boundAddr is the address actually listened on, which differs from the
	// configured one when port 0 is used.
	boundAddr string
}

// agentExecutionMetrics holds the execution measurements exposed by the
// per-agent metrics endpoint. Every field is updated from request goroutines.
type agentExecutionMetrics struct {
	requests     atomic.Int64
	errors       atomic.Int64
	totalLatency atomic.Int64 // nanoseconds
	lastActive   atomic.Int64 // Unix nanoseconds, zero until the first request
}

// AutoServerConfig configures the auto-generated server
type AutoServerConfig struct {
	Host             string                 `yaml:"host" json:"host"`
	Port             int                    `yaml:"port" json:"port"`
	BasePath         string                 `yaml:"base_path" json:"base_path"`
	EnableWebUI      bool                   `yaml:"enable_web_ui" json:"enable_web_ui"`
	EnablePlayground bool                   `yaml:"enable_playground" json:"enable_playground"`
	EnableSchemaAPI  bool                   `yaml:"enable_schema_api" json:"enable_schema_api"`
	EnableMetricsAPI bool                   `yaml:"enable_metrics_api" json:"enable_metrics_api"`
	EnableCORS       bool                   `yaml:"enable_cors" json:"enable_cors"`
	SchemaValidation bool                   `yaml:"schema_validation" json:"schema_validation"`
	OllamaEndpoint   string                 `yaml:"ollama_endpoint" json:"ollama_endpoint"`
	LLMProviders     map[string]interface{} `yaml:"llm_providers" json:"llm_providers"`
	ServerTimeout    time.Duration          `yaml:"server_timeout" json:"server_timeout"`
	MaxRequestSize   int64                  `yaml:"max_request_size" json:"max_request_size"`
	Middleware       []string               `yaml:"middleware" json:"middleware"`

	// Security controls authentication, allowed origins and request limits,
	// using the same configuration type as Server. Nil falls back to
	// DefaultSecurityConfig.
	Security *SecurityConfig `yaml:"security" json:"security"`
}

// DefaultAutoServerConfig returns default configuration
func DefaultAutoServerConfig() *AutoServerConfig {
	return &AutoServerConfig{
		Host:             "0.0.0.0",
		Port:             8080,
		BasePath:         "/api",
		EnableWebUI:      true,
		EnablePlayground: true,
		EnableSchemaAPI:  true,
		EnableMetricsAPI: true,
		EnableCORS:       true,
		SchemaValidation: true,
		OllamaEndpoint:   "http://localhost:11434",
		ServerTimeout:    30 * time.Second,
		MaxRequestSize:   10 * 1024 * 1024, // 10MB
		Middleware:       []string{"cors", "logging", "recovery"},
		Security:         DefaultSecurityConfig(),
	}
}

// NewAutoServer creates a new auto-server instance backed by the process-wide
// agent registry.
//
// Note that the registry is shared: two AutoServer instances in one process see
// each other's agents, so an agent registered for one is served by the other.
// That is rarely what you want when the two servers have different exposure or
// credentials. Use NewAutoServerWithRegistry to give a server its own registry.
func NewAutoServer(config *AutoServerConfig) *AutoServer {
	if config == nil {
		config = DefaultAutoServerConfig()
	}

	router := mux.NewRouter()
	logger := logrus.New()

	// Initialize managers
	llmManager := llm.NewProviderManager()
	toolRegistry := tools.NewToolRegistry()

	// Setup LLM providers from config
	setupLLMProviders(llmManager, config)

	return &AutoServer{
		registry:       agent.GetGlobalRegistry(),
		llmManager:     llmManager,
		toolRegistry:   toolRegistry,
		router:         router,
		config:         config,
		logger:         logger,
		agentInstances: make(map[string]agent.Agent),
		agentMetadata:  make(map[string]map[string]interface{}),
		startTime:      time.Now(),
	}
}

// LoadAgentsFromDirectory loads agent definitions from a directory
// LoadAgentsFromDirectory loads every agent configuration file in a directory.
//
// This previously scanned nothing: it listed whatever was already registered
// and logged that count as "Loaded agent definitions", so a caller pointing at
// a directory of configs got silence and no agents.
func (as *AutoServer) LoadAgentsFromDirectory(directory string) error {
	as.logger.WithField("directory", directory).Info("Loading agents from directory")

	entries, err := os.ReadDir(directory)
	if err != nil {
		return fmt.Errorf("failed to read agent directory %s: %w", directory, err)
	}

	names := make([]string, 0, len(entries))
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		switch strings.ToLower(filepath.Ext(entry.Name())) {
		case ".yaml", ".yml", ".json":
			names = append(names, entry.Name())
		}
	}
	sort.Strings(names)

	if len(names) == 0 {
		return fmt.Errorf("no agent configuration files (.yaml, .yml, .json) found in %s", directory)
	}

	loaded := 0
	for _, name := range names {
		path := filepath.Join(directory, name)
		if err := as.LoadAgentsFromConfig(path); err != nil {
			return fmt.Errorf("failed to load %s: %w", path, err)
		}
		loaded++
	}

	as.logger.WithFields(logrus.Fields{
		"directory": directory,
		"files":     loaded,
	}).Info("Loaded agent definitions from directory")
	return nil
}

// LoadAgentsFromConfig loads agents from a multi-agent config file
func (as *AutoServer) LoadAgentsFromConfig(configPath string) error {
	config, err := agent.LoadMultiAgentConfigFromFile(configPath)
	if err != nil {
		return fmt.Errorf("failed to load multi-agent config: %w", err)
	}

	// Register agent configs as definitions
	for agentID, agentConfig := range config.Agents {
		definition := agent.NewBaseAgentDefinition(agentConfig)
		if err := as.registry.RegisterDefinition(agentID, definition); err != nil {
			as.logger.WithError(err).WithField("agent_id", agentID).Warn("Failed to register agent definition")
		}
	}

	as.logger.WithField("agents", len(config.Agents)).Info("Loaded agents from config")
	return nil
}

// NewAutoServerWithRegistry creates an auto-server with its own agent registry,
// isolated from the process-wide one and from any other server.
func NewAutoServerWithRegistry(config *AutoServerConfig, registry *agent.AgentRegistry) *AutoServer {
	as := NewAutoServer(config)
	if registry != nil {
		as.registry = registry
	}
	return as
}

// agentInstance returns a registered agent instance.
func (as *AutoServer) agentInstance(id string) (agent.Agent, bool) {
	as.agentsMu.RLock()
	defer as.agentsMu.RUnlock()
	instance, ok := as.agentInstances[id]
	return instance, ok
}

// agentMeta returns a registered agent's metadata.
func (as *AutoServer) agentMeta(id string) (map[string]interface{}, bool) {
	as.agentsMu.RLock()
	defer as.agentsMu.RUnlock()
	metadata, ok := as.agentMetadata[id]
	return metadata, ok
}

// agentMetaOrNil returns an agent's metadata, or nil when it is not registered.
func (as *AutoServer) agentMetaOrNil(id string) map[string]interface{} {
	metadata, _ := as.agentMeta(id)
	return metadata
}

// agentCount returns how many agents are being served.
func (as *AutoServer) agentCount() int {
	as.agentsMu.RLock()
	defer as.agentsMu.RUnlock()
	return len(as.agentInstances)
}

// agentIDs returns the served agent IDs.
func (as *AutoServer) agentIDs() []string {
	as.agentsMu.RLock()
	defer as.agentsMu.RUnlock()
	ids := make([]string, 0, len(as.agentInstances))
	for id := range as.agentInstances {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return ids
}

func (as *AutoServer) metricsForAgent(agentID string) *agentExecutionMetrics {
	metrics, _ := as.agentMetrics.LoadOrStore(agentID, &agentExecutionMetrics{})
	return metrics.(*agentExecutionMetrics)
}

// recordAgentExecution records an execution attempt, including requests that
// fail validation before an agent can run. That makes the error count useful
// when clients send malformed payloads instead of silently under-reporting it.
func (as *AutoServer) recordAgentExecution(agentID string, latency time.Duration, failed bool) {
	metrics := as.metricsForAgent(agentID)
	metrics.requests.Add(1)
	metrics.totalLatency.Add(latency.Nanoseconds())
	metrics.lastActive.Store(time.Now().UnixNano())
	if failed {
		metrics.errors.Add(1)
	}
}

// Address returns the address the server is listening on, or an empty string
// before Start. With port 0 configured this reports the port actually chosen.
func (as *AutoServer) Address() string {
	as.mu.Lock()
	defer as.mu.Unlock()
	return as.boundAddr
}

// Registry returns the agent registry this server serves from.
func (as *AutoServer) Registry() *agent.AgentRegistry {
	return as.registry
}

// RegisterAgent registers a single agent programmatically
func (as *AutoServer) RegisterAgent(id string, definition agent.AgentDefinition) error {
	return as.registry.RegisterDefinition(id, definition)
}

// GenerateEndpoints automatically generates REST endpoints for all registered agents
func (as *AutoServer) GenerateEndpoints() error {
	// Regenerating on a live server would mutate the route table and the agent
	// maps while handlers are reading them.
	if as.started.Load() {
		return fmt.Errorf("cannot generate endpoints while the server is running")
	}

	as.logger.Info("Generating dynamic endpoints for agents")

	// Apply middleware
	as.applyMiddleware()

	// Generate core system endpoints
	as.generateSystemEndpoints()

	// Generate agent-specific endpoints
	if err := as.generateAgentEndpoints(); err != nil {
		return fmt.Errorf("failed to generate agent endpoints: %w", err)
	}

	// Generate web interfaces if enabled
	if as.config.EnableWebUI {
		as.generateWebInterfaces()
	}

	// Generate schema endpoints if enabled
	if as.config.EnableSchemaAPI {
		as.generateSchemaEndpoints()
	}

	// Generate metrics endpoints if enabled
	if as.config.EnableMetricsAPI {
		as.generateMetricsEndpoints()
	}

	// Cross-origin preflight. Most routes declare concrete methods, so an
	// OPTIONS request fell through to 404 and a browser blocked every
	// cross-origin POST, PUT and DELETE against them. Registered last so it
	// only catches what no other route matched.
	as.router.PathPrefix("/").Methods(http.MethodOptions).HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	})

	as.logger.Info("Successfully generated all endpoints")
	return nil
}

// generateSystemEndpoints creates core system endpoints
func (as *AutoServer) generateSystemEndpoints() {
	// Health check
	as.router.HandleFunc("/health", as.handleHealth).Methods("GET", "OPTIONS")

	// Agent capabilities
	as.router.HandleFunc("/capabilities", as.handleCapabilities).Methods("GET", "OPTIONS")

	// List agents
	as.router.HandleFunc("/agents", as.handleListAgents).Methods("GET", "OPTIONS")

	// Agent info
	as.router.HandleFunc("/agents/{agentId}", as.handleAgentInfo).Methods("GET", "OPTIONS")

	as.logger.Info("Generated system endpoints")
}

// generateAgentEndpoints creates dynamic endpoints for each agent
func (as *AutoServer) generateAgentEndpoints() error {
	definitions := as.registry.ListDefinitions()

	for _, agentID := range definitions {
		definition, exists := as.registry.GetDefinition(agentID)
		if !exists {
			continue
		}

		// Create agent instance
		agentInstance, err := as.registry.CreateAgentFromDefinition(agentID, as.llmManager, as.toolRegistry)
		if err != nil {
			as.logger.WithError(err).WithField("agent_id", agentID).Error("Failed to create agent instance")
			continue
		}

		as.agentsMu.Lock()
		as.agentInstances[agentID] = agentInstance
		as.agentMetadata[agentID] = definition.GetMetadata()
		as.agentsMu.Unlock()

		// Generate endpoints for this agent
		basePath := fmt.Sprintf("%s/%s", as.config.BasePath, agentID)

		// Main agent execution endpoint
		as.router.HandleFunc(basePath, as.createAgentHandler(agentID)).Methods("POST", "OPTIONS")

		// Agent stream endpoint (if supported)
		as.router.HandleFunc(basePath+"/stream", as.createAgentStreamHandler(agentID)).Methods("POST", "OPTIONS")

		// Agent conversation endpoint
		as.router.HandleFunc(basePath+"/conversation", as.createConversationHandler(agentID)).Methods("GET", "POST", "DELETE")

		// Agent status endpoint
		as.router.HandleFunc(basePath+"/status", as.createStatusHandler(agentID)).Methods("GET")

		as.logger.WithField("agent_id", agentID).WithField("base_path", basePath).Info("Generated endpoints for agent")
	}

	return nil
}

// generateWebInterfaces creates web UI endpoints
func (as *AutoServer) generateWebInterfaces() {
	// Root handler redirects to chat
	as.router.HandleFunc("/", as.handleChatInterface).Methods("GET")

	// Main chat interface
	as.router.HandleFunc("/chat", as.handleChatInterface).Methods("GET")

	// Playground interface
	if as.config.EnablePlayground {
		as.router.HandleFunc("/playground", as.handlePlayground).Methods("GET")
	}

	// Debug interface
	as.router.HandleFunc("/debug", as.handleDebug).Methods("GET")

	as.logger.Info("Generated web interfaces")
}

// generateSchemaEndpoints creates schema API endpoints
func (as *AutoServer) generateSchemaEndpoints() {
	// All schemas
	as.router.HandleFunc("/schemas", as.handleSchemas).Methods("GET")

	// Specific agent schema
	as.router.HandleFunc("/schemas/{agentId}", as.handleAgentSchema).Methods("GET")

	// Schema validation endpoint
	as.router.HandleFunc("/validate/{agentId}", as.handleValidateSchema).Methods("POST")

	as.logger.Info("Generated schema endpoints")
}

// generateMetricsEndpoints creates metrics API endpoints
func (as *AutoServer) generateMetricsEndpoints() {
	// System metrics
	as.router.HandleFunc("/metrics", as.handleMetrics).Methods("GET")

	// Agent-specific metrics
	as.router.HandleFunc("/metrics/{agentId}", as.handleAgentMetrics).Methods("GET")

	as.logger.Info("Generated metrics endpoints")
}

// applyMiddleware applies the middleware chain.
//
// Recovery, the request size limit and the security headers are unconditional:
// they were previously opt-in through config.Middleware, so a deployment that
// omitted "recovery" from that list had a panicking handler tear down the
// connection, and MaxRequestSize was configured but never enforced at all.
// Recovery is registered first so it wraps everything that follows.
func (as *AutoServer) applyMiddleware() {
	if as.config.Security == nil {
		as.config.Security = DefaultSecurityConfig()
	}
	// A request limit set on the server config wins over the security default.
	if as.config.MaxRequestSize > 0 {
		as.config.Security.MaxRequestBytes = as.config.MaxRequestSize
	}

	as.router.Use(recoveryMiddleware(as.logger))
	as.router.Use(bodyLimitMiddleware(as.config.Security.maxBytes()))
	as.router.Use(securityHeadersMiddleware)
	as.router.Use(as.metricsMiddleware())

	for _, middleware := range as.config.Middleware {
		switch middleware {
		case "cors":
			if as.config.EnableCORS {
				as.router.Use(as.corsMiddleware())
			}
		case "logging":
			as.router.Use(loggingMiddleware(as.logger))
		case "recovery":
			// Applied unconditionally above.
		}
	}

	// Authentication runs innermost so rejected requests still carry the CORS
	// and security headers a browser needs to read the response.
	as.router.Use(as.authMiddleware())
}

// corsMiddleware answers cross-origin requests against the configured origin
// allowlist. It previously emitted a hardcoded "*" with no way to restrict it.
func (as *AutoServer) corsMiddleware() func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			origin := r.Header.Get("Origin")
			w.Header().Add("Vary", "Origin")

			if !as.config.Security.originAllowed(origin) {
				if r.Method == http.MethodOptions {
					w.WriteHeader(http.StatusForbidden)
					return
				}
				next.ServeHTTP(w, r)
				return
			}

			if allow := as.config.Security.corsOrigin(origin); allow != "" {
				w.Header().Set("Access-Control-Allow-Origin", allow)
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
}

// authMiddleware enforces API key authentication when configured. The
// auto-generated server previously had no authentication of any kind.
func (as *AutoServer) authMiddleware() func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			sec := as.config.Security
			if sec == nil || !sec.RequireAuth {
				next.ServeHTTP(w, r)
				return
			}
			if r.Method == http.MethodOptions || sec.isPublic(r.URL.Path) {
				next.ServeHTTP(w, r)
				return
			}

			if !sec.authorized(r.Header.Get("X-API-Key")) {
				as.logger.WithFields(logrus.Fields{
					"path":   r.URL.Path,
					"remote": r.RemoteAddr,
				}).Warn("Rejected unauthenticated request")
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusUnauthorized)
				_, _ = w.Write([]byte(`{"error":"missing or invalid API key"}`))
				return
			}

			next.ServeHTTP(w, r)
		})
	}
}

// allowedOrigin returns the value a streaming handler should echo, so
// server-sent event responses honor the same allowlist as everything else.
func (as *AutoServer) allowedOrigin(r *http.Request) string {
	if as.config == nil || as.config.Security == nil {
		return ""
	}
	origin := r.Header.Get("Origin")
	if !as.config.Security.originAllowed(origin) {
		return ""
	}
	return as.config.Security.corsOrigin(origin)
}

// Start starts the auto-server
func (as *AutoServer) Start(ctx context.Context) error {
	if err := as.GenerateEndpoints(); err != nil {
		return fmt.Errorf("failed to generate endpoints: %w", err)
	}
	// From here on the route table and agent maps are read by request
	// goroutines and must not be regenerated.
	as.started.Store(true)

	address := fmt.Sprintf("%s:%d", as.config.Host, as.config.Port)

	server := &http.Server{
		Addr:         address,
		Handler:      as.router,
		ReadTimeout:  as.config.ServerTimeout,
		WriteTimeout: as.config.ServerTimeout,
	}

	// Bind before reporting success. ListenAndServe used to run in a goroutine
	// that merely logged a failure, so Start blocked on ctx.Done() and the
	// caller believed the server was up when the port was already taken.
	listener, err := net.Listen("tcp", address)
	if err != nil {
		return fmt.Errorf("failed to listen on %s: %w", address, err)
	}

	as.mu.Lock()
	as.boundAddr = listener.Addr().String()
	as.mu.Unlock()

	as.logger.WithField("address", listener.Addr().String()).Info("Starting auto-generated multi-agent server")

	// Print available endpoints
	as.printEndpoints()

	serveErr := make(chan error, 1)
	go func() {
		if err := server.Serve(listener); err != nil && !errors.Is(err, http.ErrServerClosed) {
			as.logger.WithError(err).Error("Server stopped unexpectedly")
			serveErr <- err
			return
		}
		serveErr <- nil
	}()

	// Wait for shutdown, or for the server to stop on its own.
	select {
	case <-ctx.Done():
	case err := <-serveErr:
		if err != nil {
			return fmt.Errorf("server stopped: %w", err)
		}
	}

	as.logger.Info("Shutting down server...")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	return server.Shutdown(shutdownCtx)
}

// printEndpoints prints all available endpoints
func (as *AutoServer) printEndpoints() {
	as.logger.Info("🌐 Auto-Generated Endpoints:")
	as.logger.Info("📋 System Endpoints:")
	as.logger.Info("   GET  /health - Health check")
	as.logger.Info("   GET  /capabilities - System capabilities")
	as.logger.Info("   GET  /agents - List all agents")
	as.logger.Info("   GET  /agents/{agentId} - Agent information")

	if as.config.EnableWebUI {
		as.logger.Info("🎨 Web Interfaces:")
		as.logger.Info("   GET  /chat - Interactive chat interface")
		as.logger.Info("   GET  /debug - Debug interface")
		if as.config.EnablePlayground {
			as.logger.Info("   GET  /playground - API playground")
		}
	}

	if as.config.EnableSchemaAPI {
		as.logger.Info("📄 Schema API:")
		as.logger.Info("   GET  /schemas - All agent schemas")
		as.logger.Info("   GET  /schemas/{agentId} - Specific agent schema")
		as.logger.Info("   POST /validate/{agentId} - Validate agent input/output")
	}

	if as.config.EnableMetricsAPI {
		as.logger.Info("📊 Metrics API:")
		as.logger.Info("   GET  /metrics - System metrics")
		as.logger.Info("   GET  /metrics/{agentId} - Agent metrics")
	}

	as.logger.Info("🤖 Agent Endpoints:")
	for agentID := range as.agentInstances {
		basePath := fmt.Sprintf("%s/%s", as.config.BasePath, agentID)
		as.logger.WithField("agent", agentID).Info(fmt.Sprintf("   POST %s - Execute agent", basePath))
		as.logger.WithField("agent", agentID).Info(fmt.Sprintf("   POST %s/stream - Stream agent response", basePath))
		as.logger.WithField("agent", agentID).Info(fmt.Sprintf("   GET  %s/conversation - Get conversation history", basePath))
		as.logger.WithField("agent", agentID).Info(fmt.Sprintf("   POST %s/conversation - Add to conversation", basePath))
		as.logger.WithField("agent", agentID).Info(fmt.Sprintf("   DELETE %s/conversation - Clear conversation", basePath))
		as.logger.WithField("agent", agentID).Info(fmt.Sprintf("   GET  %s/status - Agent status", basePath))
	}
}

// setupLLMProviders initializes LLM providers based on configuration
func setupLLMProviders(manager *llm.ProviderManager, config *AutoServerConfig) {
	// Setup Ollama if endpoint is configured
	if config.OllamaEndpoint != "" {
		ollamaProvider, err := llm.NewOllamaProvider(&llm.ProviderConfig{
			Endpoint: config.OllamaEndpoint,
		})
		if err != nil {
			// Just skip this provider if it fails
			return
		}
		_ = manager.RegisterProvider("ollama", ollamaProvider)
	}

	// Setup other providers from config
	for providerName, providerConfig := range config.LLMProviders {
		// This would setup providers based on their configuration
		// Implementation depends on the specific provider types
		_ = providerName
		_ = providerConfig
	}
}

// metricsMiddleware tracks request counts
func (as *AutoServer) metricsMiddleware() func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			as.requestCount.Add(1)
			next.ServeHTTP(w, r)
		})
	}
}

// Middleware functions

func loggingMiddleware(logger *logrus.Logger) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			start := time.Now()
			next.ServeHTTP(w, r)
			logger.WithFields(logrus.Fields{
				"method":   r.Method,
				"path":     r.URL.Path,
				"duration": time.Since(start),
				"ip":       r.RemoteAddr,
			}).Info("Request processed")
		})
	}
}
