// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package builder

import (
	"sync"

	"github.com/sirupsen/logrus"

	"context"
	"fmt"
	"os"
	"time"

	"github.com/piotrlaczkowski/GoLangGraph/pkg/agent"
	"github.com/piotrlaczkowski/GoLangGraph/pkg/llm"
	"github.com/piotrlaczkowski/GoLangGraph/pkg/persistence"
	"github.com/piotrlaczkowski/GoLangGraph/pkg/server"
	"github.com/piotrlaczkowski/GoLangGraph/pkg/tools"
)

// QuickBuilder provides the ultimate minimal code experience for creating agents
type QuickBuilder struct {
	llmManager   *llm.ProviderManager
	toolRegistry *tools.ToolRegistry
	checkpointer persistence.Checkpointer
	config       *QuickConfig
}

// QuickConfig holds configuration for quick agent creation
type QuickConfig struct {
	// LLM Provider settings
	OpenAIKey    string
	OllamaURL    string
	GeminiKey    string
	DefaultModel string

	// Agent settings
	SystemPrompt  string
	Temperature   float64
	MaxTokens     int
	MaxIterations int

	// Persistence settings
	UseMemory   bool
	DatabaseURL string

	// Tools settings
	EnableAllTools bool
	CustomTools    []string
}

// DefaultQuickConfig returns a sensible default configuration
func DefaultQuickConfig() *QuickConfig {
	return &QuickConfig{
		OpenAIKey:      os.Getenv("OPENAI_API_KEY"),
		OllamaURL:      "http://localhost:11434",
		GeminiKey:      os.Getenv("GEMINI_API_KEY"),
		DefaultModel:   "gpt-3.5-turbo",
		SystemPrompt:   "You are a helpful AI assistant.",
		Temperature:    0.7,
		MaxTokens:      1000,
		MaxIterations:  10,
		UseMemory:      true,
		EnableAllTools: true,
	}
}

// NewQuickBuilder creates a new quick builder with auto-configuration
func NewQuickBuilder() *QuickBuilder {
	config := DefaultQuickConfig()

	// Auto-initialize LLM providers
	llmManager := llm.NewProviderManager()

	// Add OpenAI if key is available
	if config.OpenAIKey != "" {
		openaiProvider, err := llm.NewOpenAIProvider(&llm.ProviderConfig{
			APIKey:   config.OpenAIKey,
			Endpoint: "https://api.openai.com/v1",
		})
		if err == nil {
			if regErr := llmManager.RegisterProvider("openai", openaiProvider); regErr != nil {
				logrus.WithError(regErr).Warnf("failed to register %s provider", "openai")
			}
		}
	}

	// Add Ollama
	ollamaProvider, err := llm.NewOllamaProvider(&llm.ProviderConfig{
		Endpoint: config.OllamaURL,
	})
	if err == nil {
		if regErr := llmManager.RegisterProvider("ollama", ollamaProvider); regErr != nil {
			logrus.WithError(regErr).Warnf("failed to register %s provider", "ollama")
		}
	}

	// Add Gemini if key is available
	if config.GeminiKey != "" {
		geminiProvider, err := llm.NewGeminiProvider(&llm.ProviderConfig{
			APIKey: config.GeminiKey,
		})
		if err == nil {
			if regErr := llmManager.RegisterProvider("gemini", geminiProvider); regErr != nil {
				logrus.WithError(regErr).Warnf("failed to register %s provider", "gemini")
			}
		}
	}

	// Auto-initialize tools
	toolRegistry := tools.NewToolRegistry()
	if config.EnableAllTools {
		for _, tool := range []tools.Tool{
			tools.NewCalculatorTool(), tools.NewWebSearchTool(),
			tools.NewFileReadTool(), tools.NewFileWriteTool(),
			tools.NewShellTool(), tools.NewHTTPTool(), tools.NewTimeTool(),
		} {
			// A duplicate registration is not fatal, but silently swallowing it
			// leaves the builder missing a tool the caller asked for.
			if err := toolRegistry.RegisterTool(tool); err != nil {
				logrus.WithError(err).Warnf("failed to register tool %s", tool.GetName())
			}
		}
	}

	// Auto-initialize checkpointer
	var checkpointer persistence.Checkpointer
	if config.UseMemory {
		checkpointer = persistence.NewMemoryCheckpointer()
	}

	return &QuickBuilder{
		llmManager:   llmManager,
		toolRegistry: toolRegistry,
		checkpointer: checkpointer,
		config:       config,
	}
}

// WithConfig allows customizing the configuration
func (qb *QuickBuilder) WithConfig(config *QuickConfig) *QuickBuilder {
	qb.config = config
	return qb
}

// WithLLM adds or configures an LLM provider
func (qb *QuickBuilder) WithLLM(provider string, config interface{}) *QuickBuilder {
	switch provider {
	case "openai":
		if cfg, ok := config.(*llm.ProviderConfig); ok {
			openaiProvider, err := llm.NewOpenAIProvider(cfg)
			if err == nil {
				if regErr := qb.llmManager.RegisterProvider("openai", openaiProvider); regErr != nil {
					logrus.WithError(regErr).Warnf("failed to register %s provider", "openai")
				}
			}
		}
	case "ollama":
		if cfg, ok := config.(*llm.ProviderConfig); ok {
			ollamaProvider, err := llm.NewOllamaProvider(cfg)
			if err == nil {
				if regErr := qb.llmManager.RegisterProvider("ollama", ollamaProvider); regErr != nil {
					logrus.WithError(regErr).Warnf("failed to register %s provider", "ollama")
				}
			}
		}
	case "gemini":
		if cfg, ok := config.(*llm.ProviderConfig); ok {
			geminiProvider, err := llm.NewGeminiProvider(cfg)
			if err == nil {
				if regErr := qb.llmManager.RegisterProvider("gemini", geminiProvider); regErr != nil {
					logrus.WithError(regErr).Warnf("failed to register %s provider", "gemini")
				}
			}
		}
	}
	return qb
}

// WithTools adds custom tools
func (qb *QuickBuilder) WithTools(tools ...tools.Tool) *QuickBuilder {
	for _, tool := range tools {
		if err := qb.toolRegistry.RegisterTool(tool); err != nil {
			logrus.WithError(err).Warnf("failed to register tool %s", tool.GetName())
		}
	}
	return qb
}

// WithPersistence configures persistence
func (qb *QuickBuilder) WithPersistence(checkpointer persistence.Checkpointer) *QuickBuilder {
	qb.checkpointer = checkpointer
	return qb
}

// ========== ULTRA-MINIMAL AGENT CREATION ==========

// Chat creates a simple chat agent in 1 line
func (qb *QuickBuilder) Chat(name ...string) *agent.Agent {
	agentName := "ChatAgent"
	if len(name) > 0 {
		agentName = name[0]
	}

	// DefaultAgentConfig assigns a unique ID and the defaulted
	// fields; building a bare literal left ID empty, so every agent
	// created through this package collided on "" in AgentManager.
	config := agent.DefaultAgentConfig()
	config.Name = agentName
	config.Type = agent.AgentTypeChat
	config.SystemPrompt = qb.config.SystemPrompt
	config.Temperature = qb.config.Temperature
	config.MaxTokens = qb.config.MaxTokens
	config.Provider = qb.getBestProvider()
	config.Model = qb.config.DefaultModel

	return agent.NewAgent(config, qb.llmManager, qb.toolRegistry)
}

// ReAct creates a ReAct agent with reasoning capabilities
func (qb *QuickBuilder) ReAct(name ...string) *agent.Agent {
	agentName := "ReActAgent"
	if len(name) > 0 {
		agentName = name[0]
	}

	// DefaultAgentConfig assigns a unique ID and the defaulted
	// fields; building a bare literal left ID empty, so every agent
	// created through this package collided on "" in AgentManager.
	config := agent.DefaultAgentConfig()
	config.Name = agentName
	config.Type = agent.AgentTypeReAct
	config.SystemPrompt = "You are a helpful assistant that can reason step by step and use tools when needed."
	config.Temperature = qb.config.Temperature
	config.MaxTokens = qb.config.MaxTokens
	config.MaxIterations = qb.config.MaxIterations
	config.Provider = qb.getBestProvider()
	config.Model = qb.config.DefaultModel
	config.Tools = qb.toolRegistry.ListTools()

	return agent.NewAgent(config, qb.llmManager, qb.toolRegistry)
}

// Tool creates a tool-focused agent
func (qb *QuickBuilder) Tool(name ...string) *agent.Agent {
	agentName := "ToolAgent"
	if len(name) > 0 {
		agentName = name[0]
	}

	// DefaultAgentConfig assigns a unique ID and the defaulted
	// fields; building a bare literal left ID empty, so every agent
	// created through this package collided on "" in AgentManager.
	config := agent.DefaultAgentConfig()
	config.Name = agentName
	config.Type = agent.AgentTypeTool
	config.SystemPrompt = "You are a helpful assistant that specializes in using tools to accomplish tasks."
	config.Temperature = qb.config.Temperature
	config.MaxTokens = qb.config.MaxTokens
	config.Provider = qb.getBestProvider()
	config.Model = qb.config.DefaultModel
	config.Tools = qb.toolRegistry.ListTools()

	return agent.NewAgent(config, qb.llmManager, qb.toolRegistry)
}

// RAG creates a RAG (Retrieval-Augmented Generation) agent
func (qb *QuickBuilder) RAG(name ...string) *agent.Agent {
	agentName := "RAGAgent"
	if len(name) > 0 {
		agentName = name[0]
	}

	// DefaultAgentConfig assigns a unique ID and the defaulted
	// fields; building a bare literal left ID empty, so every agent
	// created through this package collided on "" in AgentManager.
	config := agent.DefaultAgentConfig()
	config.Name = agentName
	config.Type = agent.AgentTypeChat
	config.SystemPrompt = "You are a helpful assistant that can search and retrieve information from documents to answer questions accurately."
	config.Temperature = qb.config.Temperature
	config.MaxTokens = qb.config.MaxTokens
	config.Provider = qb.getBestProvider()
	config.Model = qb.config.DefaultModel
	config.Tools = []string{"web_search", "file_read"}

	return agent.NewAgent(config, qb.llmManager, qb.toolRegistry)
}

// Multi creates a multi-agent coordinator
func (qb *QuickBuilder) Multi() *agent.MultiAgentCoordinator {
	return agent.NewMultiAgentCoordinator()
}

// ========== SPECIALIZED AGENTS ==========

// Researcher creates a research-focused agent
func (qb *QuickBuilder) Researcher(name ...string) *agent.Agent {
	agentName := "Researcher"
	if len(name) > 0 {
		agentName = name[0]
	}

	// DefaultAgentConfig assigns a unique ID and the defaulted
	// fields; building a bare literal left ID empty, so every agent
	// created through this package collided on "" in AgentManager.
	config := agent.DefaultAgentConfig()
	config.Name = agentName
	config.Type = agent.AgentTypeReAct
	config.SystemPrompt = "You are a research specialist. You excel at finding, analyzing, and synthesizing information from multiple sources."
	config.Temperature = 0.3 // Lower temperature for more focused research
	config.MaxTokens = 2000
	config.Provider = qb.getBestProvider()
	config.Model = qb.config.DefaultModel
	config.Tools = []string{"web_search", "file_read", "http_request"}

	return agent.NewAgent(config, qb.llmManager, qb.toolRegistry)
}

// Writer creates a writing-focused agent
func (qb *QuickBuilder) Writer(name ...string) *agent.Agent {
	agentName := "Writer"
	if len(name) > 0 {
		agentName = name[0]
	}

	// DefaultAgentConfig assigns a unique ID and the defaulted
	// fields; building a bare literal left ID empty, so every agent
	// created through this package collided on "" in AgentManager.
	config := agent.DefaultAgentConfig()
	config.Name = agentName
	config.Type = agent.AgentTypeChat
	config.SystemPrompt = "You are a skilled technical writer. You excel at creating clear, well-structured, and engaging content."
	config.Temperature = 0.8 // Higher temperature for more creative writing
	config.MaxTokens = 2000
	config.Provider = qb.getBestProvider()
	config.Model = qb.config.DefaultModel
	config.Tools = []string{"file_write"}

	return agent.NewAgent(config, qb.llmManager, qb.toolRegistry)
}

// Analyst creates a data analysis agent
func (qb *QuickBuilder) Analyst(name ...string) *agent.Agent {
	agentName := "Analyst"
	if len(name) > 0 {
		agentName = name[0]
	}

	// DefaultAgentConfig assigns a unique ID and the defaulted
	// fields; building a bare literal left ID empty, so every agent
	// created through this package collided on "" in AgentManager.
	config := agent.DefaultAgentConfig()
	config.Name = agentName
	config.Type = agent.AgentTypeReAct
	config.SystemPrompt = "You are a data analyst. You excel at analyzing data, identifying patterns, and providing insights."
	config.Temperature = 0.2 // Low temperature for precise analysis
	config.MaxTokens = 1500
	config.Provider = qb.getBestProvider()
	config.Model = qb.config.DefaultModel
	config.Tools = []string{"calculator", "file_read", "shell"}

	return agent.NewAgent(config, qb.llmManager, qb.toolRegistry)
}

// Coder creates a coding assistant agent
func (qb *QuickBuilder) Coder(name ...string) *agent.Agent {
	agentName := "Coder"
	if len(name) > 0 {
		agentName = name[0]
	}

	// DefaultAgentConfig assigns a unique ID and the defaulted
	// fields; building a bare literal left ID empty, so every agent
	// created through this package collided on "" in AgentManager.
	config := agent.DefaultAgentConfig()
	config.Name = agentName
	config.Type = agent.AgentTypeReAct
	config.SystemPrompt = "You are a coding assistant. You excel at writing, debugging, and explaining code in multiple programming languages."
	config.Temperature = 0.3
	config.MaxTokens = 2000
	config.Provider = qb.getBestProvider()
	config.Model = qb.config.DefaultModel
	config.Tools = []string{"file_read", "file_write", "shell"}

	return agent.NewAgent(config, qb.llmManager, qb.toolRegistry)
}

// ========== WORKFLOW BUILDERS ==========

// Pipeline creates a sequential agent pipeline
func (qb *QuickBuilder) Pipeline(agents ...*agent.Agent) *AgentPipeline {
	return &AgentPipeline{
		agents:      agents,
		coordinator: agent.NewMultiAgentCoordinator(),
	}
}

// Swarm creates a parallel agent swarm
func (qb *QuickBuilder) Swarm(agents ...*agent.Agent) *AgentSwarm {
	return &AgentSwarm{
		agents:      agents,
		coordinator: agent.NewMultiAgentCoordinator(),
	}
}

// ========== SERVER BUILDER ==========

// Server creates a ready-to-use server
func (qb *QuickBuilder) Server(port ...int) *server.Server {
	serverPort := 8080
	if len(port) > 0 {
		serverPort = port[0]
	}

	config := &server.ServerConfig{
		Host:           "0.0.0.0",
		Port:           serverPort,
		ReadTimeout:    30 * time.Second,
		WriteTimeout:   30 * time.Second,
		MaxHeaderBytes: 1 << 20,
		EnableCORS:     true,
	}

	srv := server.NewServer(config)
	srv.SetLLMManager(qb.llmManager)
	srv.SetToolRegistry(qb.toolRegistry)

	// Create and set agent manager
	agentManager := server.NewAgentManager(qb.llmManager, qb.toolRegistry)
	srv.SetAgentManager(agentManager)

	// Create and set session manager
	sessionManager := persistence.NewSessionManager(nil)
	srv.SetSessionManager(sessionManager)

	return srv
}

// ========== HELPER METHODS ==========

// getBestProvider returns the best available provider
func (qb *QuickBuilder) getBestProvider() string {
	providers := qb.llmManager.ListProviders()

	// Priority order: OpenAI, Gemini, Ollama
	for _, preferred := range []string{"openai", "gemini", "ollama"} {
		for _, available := range providers {
			if available == preferred {
				return preferred
			}
		}
	}

	// Return first available provider
	if len(providers) > 0 {
		return providers[0]
	}

	// No provider is configured. Returning "mock" here named a provider that
	// does not exist, so the agent failed later with a confusing lookup error
	// instead of pointing at the real problem.
	logrus.Warn("no LLM provider is configured; set OPENAI_API_KEY, GEMINI_API_KEY, or run Ollama")
	return ""
}

// ========== WORKFLOW TYPES ==========

// AgentPipeline represents a sequential workflow
type AgentPipeline struct {
	agents      []*agent.Agent
	coordinator *agent.MultiAgentCoordinator

	mu       sync.Mutex
	agentIDs []string
}

// Execute runs the pipeline sequentially.
//
// Agents are registered once, on the first run: re-registering the same IDs on
// every call meant a second Execute either failed or silently replaced the
// coordinator's agents, and two concurrent calls raced on that registration.
func (ap *AgentPipeline) Execute(ctx context.Context, input string) ([]agent.AgentExecution, error) {
	return ap.coordinator.ExecuteSequential(ctx, ap.register(), input)
}

// register assigns each agent a stable ID exactly once.
func (ap *AgentPipeline) register() []string {
	ap.mu.Lock()
	defer ap.mu.Unlock()

	if ap.agentIDs != nil {
		return ap.agentIDs
	}

	ids := make([]string, len(ap.agents))
	for i, a := range ap.agents {
		id := fmt.Sprintf("agent_%d", i)
		ids[i] = id
		ap.coordinator.AddAgent(id, a)
	}
	ap.agentIDs = ids
	return ids
}

// AgentSwarm represents a parallel workflow
type AgentSwarm struct {
	agents      []*agent.Agent
	coordinator *agent.MultiAgentCoordinator

	mu       sync.Mutex
	agentIDs []string
}

// Execute runs the swarm in parallel.
//
// Results are returned in the order the agents were supplied. The previous
// implementation ranged over the results map, so Go's randomised map iteration
// meant a caller indexing the slice got a different agent's result each run.
func (as *AgentSwarm) Execute(ctx context.Context, input string) ([]agent.AgentExecution, error) {
	agentIDs := as.register()

	results, err := as.coordinator.ExecuteParallel(ctx, agentIDs, input)
	if err != nil {
		return nil, err
	}

	executions := make([]agent.AgentExecution, 0, len(results))
	for _, id := range agentIDs {
		if execution, ok := results[id]; ok {
			executions = append(executions, execution)
		}
	}

	return executions, nil
}

// register assigns each agent a stable ID exactly once.
func (as *AgentSwarm) register() []string {
	as.mu.Lock()
	defer as.mu.Unlock()

	if as.agentIDs != nil {
		return as.agentIDs
	}

	ids := make([]string, len(as.agents))
	for i, a := range as.agents {
		id := fmt.Sprintf("agent_%d", i)
		ids[i] = id
		as.coordinator.AddAgent(id, a)
	}
	as.agentIDs = ids
	return ids
}

// ========== GLOBAL QUICK FUNCTIONS ==========

// Quick returns a global quick builder instance
func Quick() *QuickBuilder {
	return NewQuickBuilder()
}

// OneLineChat creates a chat agent in one line
func OneLineChat(name ...string) *agent.Agent {
	return Quick().Chat(name...)
}

// OneLineReAct creates a ReAct agent in one line
func OneLineReAct(name ...string) *agent.Agent {
	return Quick().ReAct(name...)
}

// OneLineTool creates a tool agent in one line
func OneLineTool(name ...string) *agent.Agent {
	return Quick().Tool(name...)
}

// OneLineRAG creates a RAG agent in one line
func OneLineRAG(name ...string) *agent.Agent {
	return Quick().RAG(name...)
}

// OneLineServer creates a server in one line
func OneLineServer(port ...int) *server.Server {
	return Quick().Server(port...)
}

// OneLinePipeline creates a pipeline in one line
func OneLinePipeline(agents ...*agent.Agent) *AgentPipeline {
	return Quick().Pipeline(agents...)
}

// OneLineSwarm creates a swarm in one line
func OneLineSwarm(agents ...*agent.Agent) *AgentSwarm {
	return Quick().Swarm(agents...)
}
