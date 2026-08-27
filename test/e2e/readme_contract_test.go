// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package e2e

import (
	"context"
	"testing"

	"github.com/UnicoLab/GoLangGraph/pkg/agent"
	"github.com/UnicoLab/GoLangGraph/pkg/core"
	"github.com/UnicoLab/GoLangGraph/pkg/llm"
	"github.com/UnicoLab/GoLangGraph/pkg/tools"
	"github.com/UnicoLab/GoLangGraph/test/fakes"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// The README's Quick Start is the first code most users run. It was never
// exercised by a test, so nothing would catch it drifting from the API.
//
// These tests follow the documented shape exactly — including building
// AgentConfig as a bare literal, which is what the README shows — substituting
// only the model provider so they run without a live Ollama.

func TestReadme_SimpleChatAgent(t *testing.T) {
	llmManager := llm.NewProviderManager()
	provider := fakes.NewProvider("ollama", "Go is a statically typed language.")
	require.NoError(t, llmManager.RegisterProvider("ollama", provider))

	toolRegistry := tools.NewToolRegistry()

	// Exactly the config shape the README documents: no ID field.
	config := &agent.AgentConfig{
		Name:         "chat-agent",
		Type:         agent.AgentTypeChat,
		Model:        "gemma3:1b",
		Provider:     "ollama",
		SystemPrompt: "You are a helpful AI assistant.",
		Temperature:  0.7,
		MaxTokens:    500,
	}

	chatAgent := agent.NewAgent(config, llmManager, toolRegistry)
	require.NotNil(t, chatAgent)

	execution, err := chatAgent.Execute(context.Background(), "Hello! Tell me about Go programming.")
	require.NoError(t, err, "the documented quick start must actually run")

	assert.True(t, execution.Success)
	assert.Contains(t, execution.Output, "statically typed")
	assert.Equal(t, 1, provider.Calls(), "the agent must call the configured provider")

	// An agent built from a literal must still get an identity: AgentManager
	// keys by ID, so an empty one means every such agent collides.
	assert.NotEmpty(t, chatAgent.GetConfig().ID,
		"an agent configured the documented way must still have an ID")
}

// Two agents built the documented way must be independently addressable.
func TestReadme_LiteralConfiguredAgentsAreDistinct(t *testing.T) {
	llmManager := llm.NewProviderManager()
	require.NoError(t, llmManager.RegisterProvider("ollama", fakes.NewProvider("ollama", "ok")))
	registry := tools.NewToolRegistry()

	newDocumentedAgent := func(name string) agent.Agent {
		return agent.NewAgent(&agent.AgentConfig{
			Name:     name,
			Type:     agent.AgentTypeChat,
			Model:    "gemma3:1b",
			Provider: "ollama",
		}, llmManager, registry)
	}

	first := newDocumentedAgent("first")
	second := newDocumentedAgent("second")

	assert.NotEqual(t, first.GetConfig().ID, second.GetConfig().ID,
		"two agents must not share an ID, or one replaces the other when registered")
}

// The README's graph workflow section documents building a graph directly.
func TestReadme_GraphWorkflow(t *testing.T) {
	graph := core.NewGraph("workflow")

	graph.AddNode("start", "Start", func(ctx context.Context, state *core.BaseState) (*core.BaseState, error) {
		state.Set("step", "started")
		return state, nil
	})
	graph.AddNode("process", "Process", func(ctx context.Context, state *core.BaseState) (*core.BaseState, error) {
		state.Set("step", "processed")
		return state, nil
	})
	graph.AddEdge("start", "process", nil)
	require.NoError(t, graph.SetStartNode("start"))
	require.NoError(t, graph.AddEndNode("process"))

	initial := core.NewBaseState()
	initial.Set("input", "hello")

	result, err := graph.Execute(context.Background(), initial)
	require.NoError(t, err)

	step, ok := result.Get("step")
	require.True(t, ok)
	assert.Equal(t, "processed", step)

	input, ok := result.Get("input")
	require.True(t, ok, "the caller's initial state must survive execution")
	assert.Equal(t, "hello", input)
}

// A ReAct agent with tools, as the README's second example shows.
func TestReadme_ReActAgentWithTools(t *testing.T) {
	llmManager := llm.NewProviderManager()
	require.NoError(t, llmManager.RegisterProvider("ollama",
		fakes.NewProvider("ollama", "The answer is 4.")))

	registry := tools.NewToolRegistry()
	// The README lists these by name; they must exist in a default registry.
	for _, name := range []string{"calculator", "web_search", "file_read"} {
		_, exists := registry.GetTool(name)
		assert.True(t, exists, "the README references the %q tool", name)
	}

	config := &agent.AgentConfig{
		Name:          "react-agent",
		Type:          agent.AgentTypeReAct,
		Model:         "gemma3:1b",
		Provider:      "ollama",
		Tools:         []string{"calculator"},
		MaxIterations: 3,
	}

	reactAgent := agent.NewAgent(config, llmManager, registry)
	execution, err := reactAgent.Execute(context.Background(), "What is 2+2?")
	require.NoError(t, err)
	assert.NotEmpty(t, execution.ExecutionPath, "a run must record the nodes it visited")
}
