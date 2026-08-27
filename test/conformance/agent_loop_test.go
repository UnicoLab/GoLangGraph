// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

package conformance

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/UnicoLab/GoLangGraph/pkg/agent"
	"github.com/UnicoLab/GoLangGraph/pkg/llm"
	"github.com/UnicoLab/GoLangGraph/pkg/tools"
	"github.com/UnicoLab/GoLangGraph/test/fakes"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// newAgent wires a real agent to a scripted provider and the real tool
// registry, so the agent loop, routing and tool execution are all genuine.
func newAgent(t *testing.T, kind agent.AgentType, provider *fakes.Provider, mutate func(*agent.AgentConfig)) agent.Agent {
	t.Helper()

	providers := llm.NewProviderManager()
	require.NoError(t, providers.RegisterProvider("fake", provider))

	registry := tools.NewToolRegistry()

	cfg := agent.DefaultAgentConfig()
	cfg.ID = "conformance-agent"
	cfg.Name = "Conformance Agent"
	cfg.Type = kind
	cfg.Provider = "fake"
	cfg.Model = "fake-model"
	cfg.Tools = []string{"calculator"}
	if mutate != nil {
		mutate(cfg)
	}

	return agent.NewAgent(cfg, providers, registry)
}

// LangGraph: an agent run produces a result and records what it did.
func TestConformance_ChatAgentRun(t *testing.T) {
	provider := fakes.NewProvider("fake", "the answer is 42")
	a := newAgent(t, agent.AgentTypeChat, provider, nil)

	execution, err := a.Execute(context.Background(), "what is the answer?")
	require.NoError(t, err)
	require.NotNil(t, execution)

	assert.True(t, execution.Success)
	assert.Equal(t, "what is the answer?", execution.Input)
	assert.Contains(t, execution.Output, "42")
	assert.NotEmpty(t, execution.ID)
	assert.NotZero(t, execution.Duration)
	assert.Equal(t, 1, provider.Calls(), "a chat turn should take exactly one model call")

	assert.NotEmpty(t, execution.ExecutionPath, "the nodes that ran must be recorded")
}

// The agent must pass the user's input to the model rather than inventing one.
func TestConformance_AgentSendsUserInput(t *testing.T) {
	provider := fakes.NewProvider("fake", "ack")
	a := newAgent(t, agent.AgentTypeChat, provider, nil)

	_, err := a.Execute(context.Background(), "a very specific question")
	require.NoError(t, err)

	prompts := provider.Prompts()
	require.NotEmpty(t, prompts)
	assert.Contains(t, strings.Join(prompts, "\n"), "a very specific question")
}

// LangGraph: an agent loop iterates until it decides to stop, bounded by a
// maximum iteration count that guarantees termination.
func TestConformance_ReActLoopTerminates(t *testing.T) {
	// Every reply asks for another action, so only the iteration bound can end
	// the loop. A framework without that bound would run forever.
	provider := fakes.NewProvider("fake", "Action: use a tool and then continue")
	a := newAgent(t, agent.AgentTypeReAct, provider, func(c *agent.AgentConfig) {
		c.MaxIterations = 3
	})

	done := make(chan struct{})
	var execution *agent.AgentExecution
	var err error
	go func() {
		defer close(done)
		execution, err = a.Execute(context.Background(), "solve this")
	}()

	select {
	case <-done:
	case <-time.After(60 * time.Second):
		t.Fatal("the agent loop did not terminate")
	}

	require.NotNil(t, execution, "an execution record must be produced even when the loop is cut short: %v", err)
	assert.NotEmpty(t, execution.ExecutionPath)
	assert.LessOrEqual(t, provider.Calls(), 50,
		"the loop must be bounded, but the model was called %d times", provider.Calls())
}

// A provider failure must surface as a failed execution carrying the reason,
// not a hang and not a success.
func TestConformance_AgentReportsProviderFailure(t *testing.T) {
	sentinel := errors.New("model backend unavailable")
	provider := fakes.NewProvider("fake", "unused").FailWith(sentinel, sentinel, sentinel, sentinel, sentinel)
	a := newAgent(t, agent.AgentTypeChat, provider, nil)

	execution, err := a.Execute(context.Background(), "hello")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "model backend unavailable")

	if execution != nil {
		assert.False(t, execution.Success)
		assert.NotEmpty(t, execution.ErrorMessage,
			"the failure reason must be serialisable for clients")
		assert.Contains(t, execution.ErrorMessage, "model backend unavailable")
	}
}

// Cancelling a run must stop it promptly rather than waiting on the provider.
func TestConformance_AgentCancellation(t *testing.T) {
	provider := fakes.NewProvider("fake", "slow").WithDelay(30 * time.Second)
	a := newAgent(t, agent.AgentTypeChat, provider, nil)

	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	start := time.Now()
	_, err := a.Execute(ctx, "hello")
	elapsed := time.Since(start)

	require.Error(t, err)
	assert.Less(t, elapsed, 10*time.Second, "cancellation took %s", elapsed)
}

// An agent must refuse to run twice at once, so its conversation and execution
// record cannot interleave.
func TestConformance_AgentRejectsConcurrentRuns(t *testing.T) {
	provider := fakes.NewProvider("fake", "ok").WithDelay(300 * time.Millisecond)
	a := newAgent(t, agent.AgentTypeChat, provider, nil)

	var wg sync.WaitGroup
	results := make([]error, 4)
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			_, results[i] = a.Execute(context.Background(), fmt.Sprintf("run-%d", i))
		}(i)
	}
	wg.Wait()

	var succeeded, rejected int
	for _, err := range results {
		if err == nil {
			succeeded++
			continue
		}
		if strings.Contains(err.Error(), "already running") {
			rejected++
		}
	}
	assert.Positive(t, succeeded, "at least one run must succeed")
	assert.Equal(t, 4, succeeded+rejected,
		"every run must either succeed or be rejected as concurrent, got %v", results)
}

// Execution history must accumulate across runs so a debugger can replay them.
func TestConformance_AgentHistoryAccumulates(t *testing.T) {
	provider := fakes.NewProvider("fake", "reply")
	a := newAgent(t, agent.AgentTypeChat, provider, nil)

	for i := 0; i < 3; i++ {
		_, err := a.Execute(context.Background(), fmt.Sprintf("question %d", i))
		require.NoError(t, err)
	}

	history := a.GetExecutionHistory()
	require.Len(t, history, 3)
	for i, execution := range history {
		assert.Equal(t, fmt.Sprintf("question %d", i), execution.Input)
		assert.True(t, execution.Success)
	}
}

// A conversation must carry across turns, which is what makes a thread a thread.
func TestConformance_AgentConversationIsRetained(t *testing.T) {
	provider := fakes.NewProvider("fake", "reply")
	a := newAgent(t, agent.AgentTypeChat, provider, nil)

	_, err := a.Execute(context.Background(), "first")
	require.NoError(t, err)
	_, err = a.Execute(context.Background(), "second")
	require.NoError(t, err)

	conversation := a.GetConversation()
	joined := ""
	for _, m := range conversation {
		joined += m.Role + ":" + m.Content + "\n"
	}
	assert.Contains(t, joined, "first")
	assert.Contains(t, joined, "second")
}

// LangGraph: tools are executed by the framework, and both their results and
// their failures are observable.
func TestConformance_ToolExecution(t *testing.T) {
	registry := tools.NewToolRegistry()

	calculator, exists := registry.GetTool("calculator")
	require.True(t, exists, "the built-in calculator must be registered")

	result, err := calculator.Execute(context.Background(), `{"expression":"2+3"}`)
	require.NoError(t, err)
	assert.Contains(t, result, "5")

	// A malformed call is an error, not a panic.
	_, err = calculator.Execute(context.Background(), `{"expression":"}`)
	assert.Error(t, err)

	// The definition a model is shown must name the tool and its parameters.
	definition := calculator.GetDefinition()
	assert.Equal(t, "calculator", definition.Function.Name)
	assert.NotEmpty(t, definition.Function.Description)
	assert.NotNil(t, definition.Function.Parameters)
}

// An unknown tool must be reported rather than silently skipped.
func TestConformance_UnknownToolIsReported(t *testing.T) {
	registry := tools.NewToolRegistry()
	_, exists := registry.GetTool("no-such-tool")
	assert.False(t, exists)

	// Definitions must not advertise a tool the registry cannot run.
	definitions := registry.GetDefinitions([]string{"no-such-tool"})
	assert.Empty(t, definitions)
}

// Tool definitions handed to a model must cover exactly the requested tools.
func TestConformance_ToolDefinitionsForAgent(t *testing.T) {
	registry := tools.NewToolRegistry()
	definitions := registry.GetDefinitions([]string{"calculator", "time", "no-such-tool"})

	names := make([]string, 0, len(definitions))
	for _, d := range definitions {
		names = append(names, d.Function.Name)
	}
	assert.Contains(t, names, "calculator")
	assert.Contains(t, names, "time")
	assert.NotContains(t, names, "no-such-tool",
		"an unknown tool must not appear in the definitions sent to a model")
}

// LangGraph/ReAct: a model that answers directly, without asking for a tool,
// must finish with that answer.
//
// The reason node's two outgoing edges only matched when the model requested an
// action or when the iteration limit was reached, so a plain answer on the
// first turn — the ordinary case — matched neither and the run failed with
// "no valid next node" instead of returning the answer.
func TestConformance_ReActFinishesOnDirectAnswer(t *testing.T) {
	provider := fakes.NewProvider("fake", "The answer is 4.")
	a := newAgent(t, agent.AgentTypeReAct, provider, func(c *agent.AgentConfig) {
		c.MaxIterations = 5
	})

	execution, err := a.Execute(context.Background(), "What is 2+2?")
	require.NoError(t, err, "a direct answer must not fail the run")
	require.NotNil(t, execution)

	assert.True(t, execution.Success)
	assert.NotEmpty(t, execution.ExecutionPath)
	assert.Contains(t, execution.ExecutionPath, "finalize",
		"a run that answers directly must reach the finalize node, got %v", execution.ExecutionPath)
	assert.Less(t, provider.Calls(), 5,
		"answering directly must not consume the whole iteration budget")
}

// Every reasoning outcome must have somewhere to go: routing from the reason
// node must be total, not leave a gap between "act" and "finalize".
func TestConformance_ReActRoutingIsTotal(t *testing.T) {
	replies := map[string]string{
		"direct answer":     "The answer is 4.",
		"explicit final":    "Final answer: 4",
		"requests a tool":   "Action: use the calculator",
		"empty reply":       "",
		"unrelated prose":   "I have been thinking about this problem for a while.",
		"mentions conclude": "Conclusion: it is four.",
	}

	for name, reply := range replies {
		t.Run(name, func(t *testing.T) {
			provider := fakes.NewProvider("fake", reply)
			a := newAgent(t, agent.AgentTypeReAct, provider, func(c *agent.AgentConfig) {
				c.MaxIterations = 3
			})

			_, err := a.Execute(context.Background(), "solve this")
			if err != nil {
				assert.NotContains(t, err.Error(), "no valid next node",
					"reasoning %q left the graph with nowhere to go", reply)
			}
		})
	}
}
