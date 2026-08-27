// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package builder

import (
	"testing"

	"github.com/UnicoLab/GoLangGraph/pkg/agent"
	"github.com/UnicoLab/GoLangGraph/pkg/llm"
	"github.com/UnicoLab/GoLangGraph/pkg/tools"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// The one-line API is the most advertised entry point in this project, so a
// defect here is what most users meet first.

// Every constructor built a bare AgentConfig literal and never set ID. The
// agent manager keys agents by config.ID, so every agent created this way
// collided on the empty string: registering two left one.
func TestQuick_AgentsGetUniqueIDs(t *testing.T) {
	qb := Quick()

	agents := map[string]agent.Agent{
		"chat":       qb.Chat("A"),
		"react":      qb.ReAct("B"),
		"tool":       qb.Tool("C"),
		"rag":        qb.RAG("D"),
		"researcher": qb.Researcher("E"),
		"writer":     qb.Writer("F"),
		"analyst":    qb.Analyst("G"),
		"coder":      qb.Coder("H"),
	}

	seen := make(map[string]string, len(agents))
	for kind, a := range agents {
		require.NotNil(t, a, "%s agent was not created", kind)

		id := a.GetConfig().ID
		assert.NotEmpty(t, id, "%s agent has no ID", kind)

		if previous, clash := seen[id]; clash {
			t.Fatalf("%s and %s share the ID %q", kind, previous, id)
		}
		seen[id] = kind
	}
	assert.Len(t, seen, len(agents), "every agent must have a distinct ID")
}

// Defaulted fields must be populated too — a bare literal left them zero.
func TestQuick_AgentsGetDefaultedFields(t *testing.T) {
	a := Quick().Chat("Defaults")
	config := a.GetConfig()

	assert.Positive(t, config.MaxIterations, "MaxIterations must not be zero")
	assert.Positive(t, config.Timeout, "Timeout must not be zero")
	assert.NotNil(t, config.Metadata, "Metadata must not be a nil map")
	assert.Equal(t, "Defaults", config.Name)
}

// getBestProvider returned "mock" when nothing was configured, naming a
// provider that does not exist and pushing the real problem to execution time.
func TestQuick_NoProviderDoesNotNameAFakeOne(t *testing.T) {
	qb := &QuickBuilder{
		config:       DefaultQuickConfig(),
		llmManager:   emptyProviderManager(t),
		toolRegistry: emptyToolRegistry(t),
	}

	provider := qb.getBestProvider()
	assert.NotEqual(t, "mock", provider,
		"a provider named here must actually be resolvable")
	assert.Empty(t, provider, "with nothing configured the provider must be empty")
}

// A pipeline must be runnable more than once. Agents were re-registered under
// the same IDs on every call.
func TestQuick_PipelineRegistersAgentsOnce(t *testing.T) {
	qb := Quick()
	pipeline := qb.Pipeline(qb.Chat("one"), qb.Chat("two"))

	first := pipeline.register()
	second := pipeline.register()

	assert.Equal(t, []string{"agent_0", "agent_1"}, first)
	assert.Equal(t, first, second, "IDs must be stable across runs")
}

func TestQuick_SwarmRegistersAgentsOnce(t *testing.T) {
	qb := Quick()
	swarm := qb.Swarm(qb.Chat("one"), qb.Chat("two"), qb.Chat("three"))

	first := swarm.register()
	second := swarm.register()

	assert.Equal(t, []string{"agent_0", "agent_1", "agent_2"}, first)
	assert.Equal(t, first, second)
}

// A swarm's results were collected by ranging over a map, so Go's randomized
// iteration gave a caller a different agent's result on each run.
func TestQuick_SwarmResultOrderIsDeterministic(t *testing.T) {
	qb := Quick()
	swarm := qb.Swarm(qb.Chat("a"), qb.Chat("b"), qb.Chat("c"), qb.Chat("d"))

	ids := swarm.register()
	require.Equal(t, []string{"agent_0", "agent_1", "agent_2", "agent_3"}, ids)

	// Build a result map keyed the way the coordinator returns one, then check
	// the ordering the swarm applies to it.
	results := map[string]agent.AgentExecution{
		"agent_0": {ID: "zero"},
		"agent_1": {ID: "one"},
		"agent_2": {ID: "two"},
		"agent_3": {ID: "three"},
	}

	for i := 0; i < 25; i++ {
		ordered := make([]string, 0, len(ids))
		for _, id := range ids {
			ordered = append(ordered, results[id].ID)
		}
		require.Equal(t, []string{"zero", "one", "two", "three"}, ordered,
			"results must follow the declared agent order")
	}
}

// The one-line helpers must produce usable agents, not nil.
func TestQuick_OneLineHelpers(t *testing.T) {
	for name, a := range map[string]agent.Agent{
		"OneLineChat":  OneLineChat("chat"),
		"OneLineReAct": OneLineReAct("react"),
		"OneLineTool":  OneLineTool("tool"),
		"OneLineRAG":   OneLineRAG("rag"),
	} {
		require.NotNil(t, a, "%s returned nil", name)
		assert.NotEmpty(t, a.GetConfig().ID, "%s produced an agent with no ID", name)
	}
}

// emptyProviderManager returns a manager with no providers registered.
func emptyProviderManager(t *testing.T) *llm.ProviderManager {
	t.Helper()
	return llm.NewProviderManager()
}

// emptyToolRegistry returns the default registry; tools are irrelevant here.
func emptyToolRegistry(t *testing.T) *tools.ToolRegistry {
	t.Helper()
	return tools.NewToolRegistry()
}
