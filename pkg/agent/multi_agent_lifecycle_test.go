// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/piotrlaczkowski/GoLangGraph/pkg/llm"
	"github.com/piotrlaczkowski/GoLangGraph/pkg/tools"
	"github.com/piotrlaczkowski/GoLangGraph/test/fakes"
)

// healthCheckedConfig builds a config whose agents are health checked, which is
// what starts the manager's background goroutines.
func healthCheckedConfig(name string, agentIDs []string, period int) *MultiAgentConfig {
	config := &MultiAgentConfig{
		Name:    name,
		Version: "1.0",
		Agents:  make(map[string]*AgentConfig, len(agentIDs)),
		Routing: &RoutingConfig{Type: "path"},
		Deployment: &DeploymentConfig{
			Type:     "docker",
			Replicas: 1,
			HealthCheck: &HealthCheckConfig{
				Enabled:          true,
				PeriodSeconds:    period,
				TimeoutSeconds:   1,
				FailureThreshold: 2,
			},
		},
	}
	for _, id := range agentIDs {
		config.Agents[id] = newTestAgentConfig(id, "fake")
		config.Routing.Rules = append(config.Routing.Rules, RoutingRule{
			ID: id, Pattern: "/" + id, AgentID: id, Method: "POST",
		})
	}
	return config
}

// settleGoroutines waits for the goroutine count to come back down to baseline.
func settleGoroutines(t *testing.T, baseline int, timeout time.Duration) int {
	t.Helper()
	deadline := time.Now().Add(timeout)
	current := runtime.NumGoroutine()
	for time.Now().Before(deadline) {
		current = runtime.NumGoroutine()
		if current <= baseline {
			return current
		}
		time.Sleep(20 * time.Millisecond)
	}
	return current
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

// Defect: setupHealthChecking launched one bare `go mam.runHealthChecker(...)`
// per agent from the constructor, looping on `for range ticker.C` with no
// cancellation. Nothing could ever stop them, so every manager ever built
// leaked a goroutine per agent for the lifetime of the process, and Stop
// returned "stopped" while they kept ticking.
func TestMultiAgentManagerStopReclaimsHealthCheckerGoroutines(t *testing.T) {
	// Let anything left over from earlier tests wind down first.
	baseline := settleGoroutines(t, 0, 2*time.Second)

	const agentCount = 6
	agentIDs := make([]string, 0, agentCount)
	for i := 0; i < agentCount; i++ {
		agentIDs = append(agentIDs, fmt.Sprintf("agent-%d", i))
	}

	llmManager := llm.NewProviderManager()
	require.NoError(t, llmManager.RegisterProvider("fake", fakes.NewProvider("fake", "hi")))

	manager, err := NewMultiAgentManager(healthCheckedConfig("leak-check", agentIDs, 1), llmManager, tools.NewToolRegistry())
	require.NoError(t, err)

	// Construction alone must not start anything.
	assert.LessOrEqual(t, runtime.NumGoroutine(), baseline+2,
		"constructing a manager must not spawn background goroutines")

	require.NoError(t, manager.Start(context.Background()))

	// The checkers really are running now.
	require.Eventually(t, func() bool {
		return runtime.NumGoroutine() >= baseline+agentCount
	}, 3*time.Second, 20*time.Millisecond, "health checkers should be running after Start")

	stopCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	require.NoError(t, manager.Stop(stopCtx))

	after := settleGoroutines(t, baseline, 5*time.Second)
	assert.LessOrEqual(t, after, baseline,
		"Stop must reclaim every health checker goroutine (baseline %d, after %d)", baseline, after)
}

// Defect: runHealthChecker began with an unconditional
// time.Sleep(initial_delay_seconds). The shipped default config asks for 30s,
// so Stop had to wait out the whole delay before the goroutine so much as
// looked at cancellation - and there was no cancellation to look at.
func TestMultiAgentManagerStopIsPromptDespiteInitialDelay(t *testing.T) {
	config := healthCheckedConfig("slow-start", []string{"solo"}, 1)
	config.Deployment.HealthCheck.InitialDelaySeconds = 3600

	llmManager := llm.NewProviderManager()
	require.NoError(t, llmManager.RegisterProvider("fake", fakes.NewProvider("fake", "hi")))
	manager, err := NewMultiAgentManager(config, llmManager, tools.NewToolRegistry())
	require.NoError(t, err)

	require.NoError(t, manager.Start(context.Background()))

	stopCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	start := time.Now()
	require.NoError(t, manager.Stop(stopCtx))
	elapsed := time.Since(start)

	assert.Less(t, elapsed, 2*time.Second, "Stop waited out the initial delay (%v)", elapsed)
}

// Defect: an enabled health check with no period_seconds handed 0 to
// time.NewTicker, which panics - on a background goroutine, so the panic was
// unrecoverable and took the whole process down at startup.
func TestRegression_HealthCheckWithoutPeriodDoesNotPanic(t *testing.T) {
	config := &MultiAgentConfig{
		Name:    "no-period",
		Agents:  map[string]*AgentConfig{"solo": newTestAgentConfig("solo", "fake")},
		Routing: &RoutingConfig{Type: "path"},
		Deployment: &DeploymentConfig{
			Type: "docker", Replicas: 1,
			HealthCheck: &HealthCheckConfig{Enabled: true}, // period_seconds omitted
		},
	}

	manager := newManagerWithProviders(t, config, map[string]*fakes.Provider{"fake": fakes.NewProvider("fake", "hi")})

	// A default period is substituted rather than passed straight to NewTicker.
	assert.Equal(t, DefaultHealthCheckPeriod, config.Deployment.HealthCheck.Period())

	require.NoError(t, manager.Start(context.Background()))
	time.Sleep(200 * time.Millisecond) // long enough for a panicking goroutine to take us down

	stopCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	require.NoError(t, manager.Stop(stopCtx))
}

func TestMultiAgentManagerStopAndStartAreIdempotent(t *testing.T) {
	llmManager := llm.NewProviderManager()
	require.NoError(t, llmManager.RegisterProvider("fake", fakes.NewProvider("fake", "hi")))
	manager, err := NewMultiAgentManager(healthCheckedConfig("idempotent", []string{"a", "b"}, 1), llmManager, tools.NewToolRegistry())
	require.NoError(t, err)

	ctx := context.Background()

	// Stop before Start must be a no-op, not a panic or a hang.
	require.NoError(t, manager.Stop(ctx))

	require.NoError(t, manager.Start(ctx))
	require.NoError(t, manager.Start(ctx))
	assert.Equal(t, "running", manager.GetDeploymentState().Status)

	require.NoError(t, manager.Stop(ctx))
	require.NoError(t, manager.Stop(ctx))
	assert.Equal(t, "stopped", manager.GetDeploymentState().Status)

	// And it can come back up.
	require.NoError(t, manager.Start(ctx))
	assert.Equal(t, "running", manager.GetDeploymentState().Status)
	require.NoError(t, manager.Stop(ctx))
}

// Defect: Start and Stop took a context and ignored it entirely, so a caller
// whose context had already been cancelled still got "started successfully".
func TestMultiAgentManagerStartHonoursContext(t *testing.T) {
	manager, _ := newSingleAgentManager(t, nil)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	err := manager.Start(ctx)

	require.Error(t, err)
	assert.ErrorIs(t, err, context.Canceled)
	assert.NotEqual(t, "running", manager.GetDeploymentState().Status)
}

// ---------------------------------------------------------------------------
// Concurrency
// ---------------------------------------------------------------------------

// Defect: handleDeploymentStatus and handleAgentStatus copied the deployment
// state shallowly (or kept the live *AgentState), released the lock and then
// let the JSON encoder walk structures that request handlers were concurrently
// writing. -race reported the write in updateAgentError/updateAgentSuccess
// against the read in the encoder.
//
// This test hammers agent execution and every management endpoint at once; it
// fails under -race against the unfixed code.
func TestMultiAgentConcurrentExecutionAndIntrospectionIsRaceFree(t *testing.T) {
	const agentCount = 4

	agentIDs := make([]string, 0, agentCount)
	providers := map[string]*fakes.Provider{}
	config := &MultiAgentConfig{
		Name:    "concurrent",
		Version: "1.0",
		Agents:  map[string]*AgentConfig{},
		Routing: &RoutingConfig{Type: "path"},
	}
	for i := 0; i < agentCount; i++ {
		id := fmt.Sprintf("agent-%d", i)
		agentIDs = append(agentIDs, id)
		providers[id] = fakes.NewProvider(id, "reply from "+id)
		config.Agents[id] = newTestAgentConfig(id, id)
		config.Routing.Rules = append(config.Routing.Rules, RoutingRule{
			ID: id, Pattern: "/" + id, AgentID: id, Method: "POST",
		})
	}

	manager := newManagerWithProviders(t, config, providers)
	require.NoError(t, manager.Start(context.Background()))

	server := httptest.NewServer(manager.GetRouter())
	defer server.Close()

	var wg sync.WaitGroup
	var executed, introspected atomic.Int64

	// Executors: one goroutine per agent so that Agent.Execute's own
	// single-flight guard does not turn every call into an error.
	for _, id := range agentIDs {
		agentID := id
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 25; i++ {
				status, _ := postInput(t, server.URL+"/"+agentID, "hello")
				if status == http.StatusOK {
					executed.Add(1)
				}
			}
		}()
	}

	// Introspectors: every read-only endpoint, plus the in-process accessors.
	endpoints := []string{"/health", "/metrics", "/agents", "/config", "/routing", "/deployment/status", "/agents/agent-0", "/agents/agent-0/status", "/health/agent-1"}
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 30; j++ {
				endpoint := endpoints[j%len(endpoints)]
				resp, err := http.Get(server.URL + endpoint)
				if err != nil {
					continue
				}
				_ = resp.Body.Close()
				introspected.Add(1)

				_ = manager.GetDeploymentState()
				_ = manager.GetMetrics()
				_, _ = manager.OverallHealth()
			}
		}()
	}

	// Health checks run concurrently with everything else.
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 20; i++ {
			manager.CheckHealthNow(context.Background())
		}
	}()

	wg.Wait()

	assert.Equal(t, int64(agentCount*25), executed.Load(), "every request to a distinct agent should succeed")
	assert.Positive(t, introspected.Load())

	metrics := manager.GetMetrics()
	for _, id := range agentIDs {
		require.Contains(t, metrics.AgentMetrics, id)
		assert.Equal(t, int64(25), metrics.AgentMetrics[id].RequestCount)
		assert.Zero(t, metrics.AgentMetrics[id].ErrorCount)
	}
	assert.Zero(t, metrics.TotalErrors)
}

// A failing agent must not contaminate its neighbours: the others keep serving
// and only the broken one accumulates errors.
func TestMultiAgentPartialFailureIsIsolatedAndReported(t *testing.T) {
	good1 := fakes.NewProvider("good1", "fine")
	good2 := fakes.NewProvider("good2", "fine")
	// FailWith supplies an error for each of the first N calls; the fake keeps
	// its own call counter, so listing the error five times fails five calls.
	broken := fakes.NewProvider("broken", "").FailWith(
		errors.New("upstream is down"), errors.New("upstream is down"),
		errors.New("upstream is down"), errors.New("upstream is down"),
		errors.New("upstream is down"),
	)

	config := &MultiAgentConfig{
		Name: "partial-failure",
		Agents: map[string]*AgentConfig{
			"good1":  newTestAgentConfig("good1", "good1"),
			"good2":  newTestAgentConfig("good2", "good2"),
			"broken": newTestAgentConfig("broken", "broken"),
		},
		Routing: &RoutingConfig{
			Type: "path",
			Rules: []RoutingRule{
				{ID: "good1", Pattern: "/good1", AgentID: "good1", Method: "POST"},
				{ID: "good2", Pattern: "/good2", AgentID: "good2", Method: "POST"},
				{ID: "broken", Pattern: "/broken", AgentID: "broken", Method: "POST"},
			},
		},
	}

	manager := newManagerWithProviders(t, config, map[string]*fakes.Provider{
		"good1": good1, "good2": good2, "broken": broken,
	})
	server := httptest.NewServer(manager.GetRouter())
	defer server.Close()

	var wg sync.WaitGroup
	statuses := make(map[string][]int, 3)
	var mu sync.Mutex

	for _, agentID := range []string{"good1", "good2", "broken"} {
		id := agentID
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 5; i++ {
				status, _ := postInput(t, server.URL+"/"+id, "work")
				mu.Lock()
				statuses[id] = append(statuses[id], status)
				mu.Unlock()
			}
		}()
	}
	wg.Wait()

	for _, id := range []string{"good1", "good2"} {
		for _, status := range statuses[id] {
			assert.Equal(t, http.StatusOK, status, "healthy agent %s must keep serving", id)
		}
	}
	for _, status := range statuses["broken"] {
		assert.Equal(t, http.StatusInternalServerError, status)
	}

	metrics := manager.GetMetrics()
	assert.Zero(t, metrics.AgentMetrics["good1"].ErrorCount)
	assert.Zero(t, metrics.AgentMetrics["good2"].ErrorCount)
	assert.Equal(t, int64(5), metrics.AgentMetrics["broken"].ErrorCount)
	assert.Equal(t, int64(5), metrics.TotalErrors, "the partial failure must not be reported as success")

	state := manager.GetDeploymentState()
	assert.Equal(t, int64(0), state.AgentStates["good1"].ErrorCount)
	assert.Equal(t, int64(5), state.AgentStates["broken"].ErrorCount)
	assert.Contains(t, state.AgentStates["broken"].LastError, "upstream is down")
	assert.Equal(t, 5, state.ErrorCount)
}

// Cancelling the client's request must reach the agent and its provider rather
// than leaving the execution running to completion in the background.
func TestMultiAgentCancellationPropagatesToTheAgent(t *testing.T) {
	slow := fakes.NewProvider("slow", "too late").WithDelay(10 * time.Second)
	config := &MultiAgentConfig{
		Name:   "cancellation",
		Agents: map[string]*AgentConfig{"solo": newTestAgentConfig("solo", "slow")},
		Routing: &RoutingConfig{
			Type:  "path",
			Rules: []RoutingRule{{ID: "r", Pattern: "/solo", AgentID: "solo", Method: "POST"}},
		},
	}

	manager := newManagerWithProviders(t, config, map[string]*fakes.Provider{"slow": slow})
	server := httptest.NewServer(manager.GetRouter())
	defer server.Close()

	ctx, cancel := context.WithCancel(context.Background())
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, server.URL+"/solo", strings.NewReader(`{"input":"hi"}`))
	require.NoError(t, err)

	done := make(chan error, 1)
	go func() {
		resp, reqErr := http.DefaultClient.Do(req)
		if resp != nil {
			_ = resp.Body.Close()
		}
		done <- reqErr
	}()

	// Wait until the agent is actually executing, then cancel.
	agent := manager.agents["solo"]
	require.Eventually(t, agent.IsRunning, 3*time.Second, 10*time.Millisecond, "the agent should be executing")
	cancel()

	select {
	case reqErr := <-done:
		require.Error(t, reqErr, "a cancelled request must not return a normal response")
	case <-time.After(3 * time.Second):
		t.Fatal("the client request did not unblock after cancellation")
	}

	// The crucial assertion: the run is abandoned well before the provider's
	// 10s delay would have elapsed.
	require.Eventually(t, func() bool { return !agent.IsRunning() }, 3*time.Second, 10*time.Millisecond,
		"cancellation must reach the agent instead of letting it run to completion")

	assert.Equal(t, int64(1), manager.GetMetrics().AgentMetrics["solo"].ErrorCount)
}

// Restart while requests are in flight must neither deadlock nor race. The
// restart path takes the manager lock, the lifecycle lock and the metrics lock
// in sequence while handlers hold them too.
func TestMultiAgentRestartUnderConcurrentTraffic(t *testing.T) {
	providers := map[string]*fakes.Provider{
		"p0": fakes.NewProvider("p0", "a"),
		"p1": fakes.NewProvider("p1", "b"),
	}
	config := healthCheckedConfig("restart-under-load", []string{"a0", "a1"}, 1)
	config.Agents["a0"].Provider = "p0"
	config.Agents["a1"].Provider = "p1"

	manager := newManagerWithProviders(t, config, providers)
	require.NoError(t, manager.Start(context.Background()))

	server := httptest.NewServer(manager.GetRouter())
	defer server.Close()

	stop := make(chan struct{})
	var wg sync.WaitGroup

	for _, id := range []string{"a0", "a1"} {
		agentID := id
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-stop:
					return
				default:
				}
				resp, err := http.Post(server.URL+"/"+agentID, "application/json", strings.NewReader(`{"input":"hi"}`))
				if err == nil {
					_ = resp.Body.Close()
				}
			}
		}()
	}

	for i := 0; i < 3; i++ {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		err := manager.Restart(ctx)
		cancel()
		require.NoError(t, err, "restart %d failed", i)
	}

	close(stop)
	wg.Wait()

	assert.Equal(t, "running", manager.GetDeploymentState().Status)

	// Still serving after all of that.
	status, _ := postInput(t, server.URL+"/a0", "hi")
	assert.Equal(t, http.StatusOK, status)
}

// ---------------------------------------------------------------------------
// Malformed configuration
// ---------------------------------------------------------------------------

func TestMultiAgentMalformedConfigurationsAreRejected(t *testing.T) {
	validAgent := func() map[string]*AgentConfig {
		return map[string]*AgentConfig{"solo": newTestAgentConfig("solo", "fake")}
	}

	tests := []struct {
		name    string
		config  *MultiAgentConfig
		wantErr string
	}{
		{
			name:    "nil config",
			config:  nil,
			wantErr: "multi-agent config is required",
		},
		{
			name:    "no name",
			config:  &MultiAgentConfig{Agents: validAgent()},
			wantErr: "name is required",
		},
		{
			name:    "no agents",
			config:  &MultiAgentConfig{Name: "x", Agents: map[string]*AgentConfig{}},
			wantErr: "at least one agent must be defined",
		},
		{
			name:    "nil agent config",
			config:  &MultiAgentConfig{Name: "x", Agents: map[string]*AgentConfig{"ghost": nil}},
			wantErr: "agent ghost: configuration is empty",
		},
		{
			name: "agent without provider",
			config: &MultiAgentConfig{Name: "x", Agents: map[string]*AgentConfig{
				"solo": {ID: "solo", Name: "solo", Type: AgentTypeChat, Model: "m"},
			}},
			wantErr: "provider is required",
		},
		{
			name: "negative agent timeout",
			config: &MultiAgentConfig{Name: "x", Agents: func() map[string]*AgentConfig {
				agents := validAgent()
				agents["solo"].Timeout = -time.Second
				return agents
			}()},
			wantErr: "timeout must not be negative",
		},
		{
			name: "default agent missing",
			config: &MultiAgentConfig{Name: "x", Agents: validAgent(),
				Routing: &RoutingConfig{Type: "path", DefaultAgent: "nobody"}},
			wantErr: "default agent nobody does not exist",
		},
		{
			name: "rule for unknown agent",
			config: &MultiAgentConfig{Name: "x", Agents: validAgent(),
				Routing: &RoutingConfig{Type: "path", Rules: []RoutingRule{{ID: "r", Pattern: "/x", AgentID: "nobody"}}}},
			wantErr: "agent nobody does not exist",
		},
		{
			name: "duplicate rule ids",
			config: &MultiAgentConfig{Name: "x", Agents: validAgent(),
				Routing: &RoutingConfig{Type: "path", Rules: []RoutingRule{
					{ID: "dupe", Pattern: "/a", AgentID: "solo"},
					{ID: "dupe", Pattern: "/b", AgentID: "solo"},
				}}},
			wantErr: "duplicate rule id",
		},
		{
			name: "unknown match mode",
			config: &MultiAgentConfig{Name: "x", Agents: validAgent(),
				Routing: &RoutingConfig{Type: "path", Rules: []RoutingRule{
					{ID: "r", Pattern: "/a", Match: "fuzzy", AgentID: "solo"},
				}}},
			wantErr: "unsupported match mode",
		},
		{
			name: "invalid regex pattern",
			config: &MultiAgentConfig{Name: "x", Agents: validAgent(),
				Routing: &RoutingConfig{Type: "path", Rules: []RoutingRule{
					{ID: "r", Pattern: "([", Match: MatchRegex, AgentID: "solo"},
				}}},
			wantErr: "invalid regex pattern",
		},
		{
			name: "invalid condition operator",
			config: &MultiAgentConfig{Name: "x", Agents: validAgent(),
				Routing: &RoutingConfig{Type: "path", Rules: []RoutingRule{
					{ID: "r", Pattern: "/a", AgentID: "solo", Conditions: []RoutingCondition{
						{Type: ConditionHeader, Key: "X", Value: "y", Operator: "approximately"},
					}},
				}}},
			wantErr: "unsupported operator",
		},
		{
			name: "header condition without key",
			config: &MultiAgentConfig{Name: "x", Agents: validAgent(),
				Routing: &RoutingConfig{Type: "path", Rules: []RoutingRule{
					{ID: "r", Pattern: "/a", AgentID: "solo", Conditions: []RoutingCondition{
						{Type: ConditionHeader, Value: "y"},
					}},
				}}},
			wantErr: "requires a key",
		},
		{
			name: "zero replicas",
			config: &MultiAgentConfig{Name: "x", Agents: validAgent(),
				Deployment: &DeploymentConfig{Type: "docker", Replicas: 0}},
			wantErr: "replicas must be at least 1",
		},
		{
			name: "conflicting ports",
			config: &MultiAgentConfig{Name: "x", Agents: validAgent(),
				Deployment: &DeploymentConfig{Type: "docker", Replicas: 1, Networking: &NetworkingConfig{
					Ports: []PortConfig{{Name: "http", Port: 8080}, {Name: "metrics", Port: 8080}},
				}}},
			wantErr: "assigned to both",
		},
		{
			name: "negative health check period",
			config: &MultiAgentConfig{Name: "x", Agents: validAgent(),
				Deployment: &DeploymentConfig{Type: "docker", Replicas: 1,
					HealthCheck: &HealthCheckConfig{Enabled: true, PeriodSeconds: -5}}},
			wantErr: "period_seconds must not be negative",
		},
		{
			name: "health check for unknown agent",
			config: &MultiAgentConfig{Name: "x", Agents: validAgent(),
				Deployment: &DeploymentConfig{Type: "docker", Replicas: 1,
					HealthCheck: &HealthCheckConfig{Enabled: true, PeriodSeconds: 5,
						AgentSpecific: map[string]*HealthCheckConfig{"nobody": {Enabled: true}}}}},
			wantErr: "agent does not exist",
		},
		{
			name: "scaling bounds inverted",
			config: &MultiAgentConfig{Name: "x", Agents: validAgent(),
				Deployment: &DeploymentConfig{Type: "docker", Replicas: 1,
					Scaling: &ScalingConfig{Enabled: true, MinReplicas: 5, MaxReplicas: 2}}},
			wantErr: "must not be below min_replicas",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Validate must report, never panic.
			err := tt.config.Validate()
			require.Error(t, err)
			assert.Contains(t, err.Error(), tt.wantErr)

			// And the manager must refuse to build from it.
			llmManager := llm.NewProviderManager()
			require.NoError(t, llmManager.RegisterProvider("fake", fakes.NewProvider("fake", "hi")))
			manager, buildErr := NewMultiAgentManager(tt.config, llmManager, tools.NewToolRegistry())
			assert.Error(t, buildErr)
			assert.Nil(t, manager)
		})
	}
}

func TestLoadMultiAgentConfigFromFile(t *testing.T) {
	dir := t.TempDir()

	valid := `
name: from-file
version: "1.0"
agents:
  solo:
    id: solo
    name: Solo
    type: chat
    model: fake-model
    provider: fake
routing:
  type: path
  default_agent: solo
  rules:
    - id: r
      pattern: /solo
      agent_id: solo
      method: POST
`

	write := func(name, content string) string {
		path := filepath.Join(dir, name)
		require.NoError(t, os.WriteFile(path, []byte(content), 0o600))
		return path
	}

	t.Run("yaml", func(t *testing.T) {
		config, err := LoadMultiAgentConfigFromFile(write("good.yaml", valid))
		require.NoError(t, err)
		assert.Equal(t, "from-file", config.Name)
		assert.Contains(t, config.Agents, "solo")
		assert.Equal(t, "solo", config.Routing.DefaultAgent)
	})

	t.Run("json", func(t *testing.T) {
		config, err := LoadMultiAgentConfigFromFile(write("good.yaml", valid))
		require.NoError(t, err)
		encoded, err := json.Marshal(config)
		require.NoError(t, err)

		reloaded, err := LoadMultiAgentConfigFromFile(write("good.json", string(encoded)))
		require.NoError(t, err)
		assert.Equal(t, config.Name, reloaded.Name)
	})

	t.Run("empty agent entry is reported not panicked", func(t *testing.T) {
		_, err := LoadMultiAgentConfigFromFile(write("ghost.yaml", "name: ghosts\nagents:\n  ghost:\n"))
		require.Error(t, err)
		assert.Contains(t, err.Error(), "configuration is empty")
	})

	t.Run("missing file", func(t *testing.T) {
		_, err := LoadMultiAgentConfigFromFile(filepath.Join(dir, "nope.yaml"))
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed to read config file")
	})

	t.Run("unsupported extension", func(t *testing.T) {
		_, err := LoadMultiAgentConfigFromFile(write("config.toml", "name = \"x\""))
		require.Error(t, err)
		assert.Contains(t, err.Error(), "unsupported config file format")
	})

	t.Run("malformed yaml", func(t *testing.T) {
		_, err := LoadMultiAgentConfigFromFile(write("broken.yaml", "name: [unterminated"))
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed to parse YAML config")
	})

	t.Run("invalid configuration", func(t *testing.T) {
		_, err := LoadMultiAgentConfigFromFile(write("invalid.yaml", "version: \"1.0\"\n"))
		require.Error(t, err)
		assert.Contains(t, err.Error(), "invalid configuration")
	})
}

// ---------------------------------------------------------------------------
// Rate limiter internals
// ---------------------------------------------------------------------------

func TestTokenBucketRefillsOverTime(t *testing.T) {
	now := time.Now()
	bucket := newTokenBucket(2, time.Second, 2, now)

	allowed, _ := bucket.allow(now)
	assert.True(t, allowed)
	allowed, _ = bucket.allow(now)
	assert.True(t, allowed)

	allowed, retry := bucket.allow(now)
	assert.False(t, allowed, "the third request in the same instant must be rejected")
	assert.Positive(t, retry)

	// After a full period the budget is back.
	allowed, _ = bucket.allow(now.Add(time.Second))
	assert.True(t, allowed)
}

// Unbounded growth: a per-IP limiter keyed by a caller-controlled value keeps
// one bucket per source address forever unless it is swept.
func TestRateLimiterKeyedBucketsStayBounded(t *testing.T) {
	clock := time.Now()
	limiter := &rateLimiter{
		perIP:     &rateLimitRule{requests: 10, period: time.Minute, burst: 10},
		skipPaths: map[string]bool{},
		buckets:   map[string]*tokenBucket{},
		lastSeen:  map[string]time.Time{},
		now:       func() time.Time { return clock },
	}

	for i := 0; i < maxRateLimitKeys+500; i++ {
		// Advance the clock so early keys age out of the sweep window.
		clock = clock.Add(time.Millisecond)
		allowed, _ := limiter.allowKeyed(fmt.Sprintf("ip:10.0.%d.%d", i/256, i%256), limiter.perIP)
		require.True(t, allowed)
	}

	limiter.mu.Lock()
	size := len(limiter.buckets)
	seen := len(limiter.lastSeen)
	limiter.mu.Unlock()

	assert.LessOrEqual(t, size, maxRateLimitKeys, "per-key buckets must not grow without bound")
	assert.Equal(t, size, seen, "the bucket and last-seen maps must stay in step")
}

func TestRateLimiterRequiresConfiguredLimits(t *testing.T) {
	config := &MultiAgentConfig{
		Name:   "rate-limit-no-config",
		Agents: map[string]*AgentConfig{"solo": newTestAgentConfig("solo", "fake")},
		Routing: &RoutingConfig{
			Type:       "path",
			Rules:      []RoutingRule{{ID: "r", Pattern: "/solo", AgentID: "solo", Method: "POST"}},
			Middleware: []MiddlewareConfig{{Type: "rate_limit", Enabled: true}},
		},
	}

	llmManager := llm.NewProviderManager()
	require.NoError(t, llmManager.RegisterProvider("fake", fakes.NewProvider("fake", "hi")))

	_, err := NewMultiAgentManager(config, llmManager, tools.NewToolRegistry())

	require.Error(t, err)
	assert.Contains(t, err.Error(), "no limits are configured")
}

// The inline middleware spelling used by the shipped example config must work.
func TestRateLimiterReadsInlineMiddlewareConfig(t *testing.T) {
	config := &MultiAgentConfig{
		Name:   "inline-rate-limit",
		Agents: map[string]*AgentConfig{"solo": newTestAgentConfig("solo", "fake")},
		Routing: &RoutingConfig{
			Type:  "path",
			Rules: []RoutingRule{{ID: "r", Pattern: "/solo", AgentID: "solo", Method: "POST"}},
			Middleware: []MiddlewareConfig{{
				Type:    "rate_limit",
				Enabled: true,
				Config:  map[string]interface{}{"requests_per_minute": 2, "burst_limit": 2},
			}},
		},
	}

	manager := newManagerWithProviders(t, config, map[string]*fakes.Provider{"fake": fakes.NewProvider("fake", "hi")})
	server := httptest.NewServer(manager.GetRouter())
	defer server.Close()

	limited := 0
	for i := 0; i < 5; i++ {
		if status, _ := postInput(t, server.URL+"/solo", "hi"); status == http.StatusTooManyRequests {
			limited++
		}
	}
	assert.Equal(t, 3, limited)
}

// ---------------------------------------------------------------------------
// Metrics snapshots
// ---------------------------------------------------------------------------

func TestGetMetricsReturnsAnIndependentSnapshot(t *testing.T) {
	manager, _ := newSingleAgentManager(t, nil)

	snapshot := manager.GetMetrics()
	require.Contains(t, snapshot.AgentMetrics, "solo")

	manager.recordMetrics("solo", time.Millisecond, true)
	manager.updateRoutingMetrics("solo", false)
	manager.recordFailedRoute()

	assert.Equal(t, int64(0), snapshot.AgentMetrics["solo"].RequestCount, "the snapshot must not change")
	assert.Equal(t, int64(0), snapshot.TotalErrors)
	assert.Equal(t, int64(0), snapshot.RoutingMetrics.FailedRoutes)

	fresh := manager.GetMetrics()
	assert.Equal(t, int64(1), fresh.AgentMetrics["solo"].RequestCount)
	assert.Equal(t, int64(1), fresh.TotalErrors)
	assert.Equal(t, int64(1), fresh.RoutingMetrics.RoutingDecisions["solo"])
	assert.Equal(t, int64(1), fresh.RoutingMetrics.FailedRoutes)
}
