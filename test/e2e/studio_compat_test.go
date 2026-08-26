// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

// Package e2e drives a real GoLangGraph server over HTTP and WebSocket.
//
// studio_compat_test.go pins the contract that GoLangGraph Studio depends on.
// Studio is a first-class client: every request it makes is exercised here
// against a live server, and each assertion names the Studio code that reads
// the field, so a server change that would break the console fails here first.
package e2e

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/piotrlaczkowski/GoLangGraph/pkg/agent"
	"github.com/piotrlaczkowski/GoLangGraph/pkg/core"
	"github.com/piotrlaczkowski/GoLangGraph/pkg/llm"
	"github.com/piotrlaczkowski/GoLangGraph/pkg/server"
	"github.com/piotrlaczkowski/GoLangGraph/pkg/tools"
	"github.com/piotrlaczkowski/GoLangGraph/test/fakes"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// liveServer is a running GoLangGraph server plus the pieces under test.
type liveServer struct {
	baseURL  string
	apiKey   string
	provider *fakes.Provider
	agent    *agent.Agent
	srv      *server.Server
}

// startServer boots a real server on a free port, exactly as an operator would.
func startServer(t *testing.T, mutate func(*server.ServerConfig)) *liveServer {
	t.Helper()

	listener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	port := listener.Addr().(*net.TCPAddr).Port
	require.NoError(t, listener.Close())

	cfg := server.DefaultServerConfig()
	cfg.Host = "127.0.0.1"
	cfg.Port = port
	cfg.StaticDir = ""
	if mutate != nil {
		mutate(cfg)
	}

	s := server.NewServer(cfg)

	provider := fakes.NewProvider("fake", "fake response")
	providers := llm.NewProviderManager()
	require.NoError(t, providers.RegisterProvider("fake", provider))
	s.SetLLMManager(providers)

	// NewToolRegistry already registers the built-in tool set.
	registry := tools.NewToolRegistry()
	s.SetToolRegistry(registry)

	agents := server.NewAgentManager(providers, registry)
	agentCfg := agent.DefaultAgentConfig()
	agentCfg.ID = "studio-agent"
	agentCfg.Name = "Studio Agent"
	agentCfg.Type = agent.AgentTypeChat
	agentCfg.Provider = "fake"
	agentCfg.Model = "fake-model"
	agentCfg.Tools = []string{"calculator"}
	instance, err := agents.CreateAgent(agentCfg)
	require.NoError(t, err)
	s.SetAgentManager(agents)

	// A workflow graph, registered so the graph endpoints have real content.
	g := core.NewGraph("studio-workflow")
	g.AddNode("ingest", "Ingest", func(ctx context.Context, st *core.BaseState) (*core.BaseState, error) {
		in, _ := st.Get("input")
		st.Set("ingested", fmt.Sprintf("%v", in))
		return st, nil
	})
	g.AddNode("decide", "Decide", func(ctx context.Context, st *core.BaseState) (*core.BaseState, error) {
		in, _ := st.Get("ingested")
		st.Set("route", map[bool]string{true: "long", false: "short"}[len(fmt.Sprint(in)) > 5])
		return st, nil
	})
	g.AddNode("long", "Long path", func(ctx context.Context, st *core.BaseState) (*core.BaseState, error) {
		st.Set("result", "long")
		return st, nil
	})
	g.AddNode("short", "Short path", func(ctx context.Context, st *core.BaseState) (*core.BaseState, error) {
		st.Set("result", "short")
		return st, nil
	})
	g.AddEdge("ingest", "decide", nil)
	require.NoError(t, g.AddConditionalEdges("decide",
		func(ctx context.Context, st *core.BaseState) (string, error) {
			v, _ := st.Get("route")
			return fmt.Sprint(v), nil
		},
		map[string]string{"long": "long", "short": "short"}))
	require.NoError(t, g.SetStartNode("ingest"))
	require.NoError(t, g.AddEndNode("long"))
	require.NoError(t, g.AddEndNode("short"))
	s.GraphManager().Register("studio-workflow", g)

	go func() {
		if err := s.Start(); err != nil {
			t.Logf("server stopped: %v", err)
		}
	}()

	live := &liveServer{
		baseURL:  fmt.Sprintf("http://127.0.0.1:%d", port),
		provider: provider,
		agent:    instance,
		srv:      s,
	}
	if cfg.Security != nil && cfg.Security.RequireAuth && len(cfg.Security.APIKeys) > 0 {
		live.apiKey = cfg.Security.APIKeys[0]
	}

	live.waitReady(t)
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = s.Stop(ctx)
	})
	return live
}

func (l *liveServer) waitReady(t *testing.T) {
	t.Helper()
	deadline := time.Now().Add(15 * time.Second)
	for time.Now().Before(deadline) {
		resp, err := http.Get(l.baseURL + "/api/v1/health")
		if err == nil {
			_ = resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return
			}
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatal("server did not become ready")
}

// do issues a request the way the Studio API client does.
func (l *liveServer) do(t *testing.T, method, path string, body interface{}) (int, []byte) {
	t.Helper()

	var reader io.Reader
	if body != nil {
		raw, err := json.Marshal(body)
		require.NoError(t, err)
		reader = bytes.NewReader(raw)
	}

	req, err := http.NewRequest(method, l.baseURL+path, reader)
	require.NoError(t, err)
	req.Header.Set("Content-Type", "application/json")
	if l.apiKey != "" {
		req.Header.Set("X-API-Key", l.apiKey)
	}

	resp, err := http.DefaultClient.Do(req)
	require.NoError(t, err)
	defer func() { _ = resp.Body.Close() }()

	data, err := io.ReadAll(resp.Body)
	require.NoError(t, err)
	return resp.StatusCode, data
}

func decode(t *testing.T, data []byte, target interface{}) {
	t.Helper()
	require.NoError(t, json.Unmarshal(data, target), "response was not valid JSON: %s", string(data))
}

// ---------------------------------------------------------------------------
// The endpoints Studio's api/client.ts calls
// ---------------------------------------------------------------------------

// Studio: client.health()
func TestStudio_Health(t *testing.T) {
	live := startServer(t, nil)

	status, body := live.do(t, http.MethodGet, "/api/v1/health", nil)
	require.Equal(t, http.StatusOK, status)

	var health struct {
		Status    string                 `json:"status"`
		Timestamp string                 `json:"timestamp"`
		Version   string                 `json:"version"`
		Providers map[string]interface{} `json:"providers"`
	}
	decode(t, body, &health)

	assert.Equal(t, "healthy", health.Status)
	assert.NotEmpty(t, health.Timestamp, "Studio renders the timestamp")
	assert.NotEmpty(t, health.Version)
}

// Studio: client.listAgents() reads res.agents and renders name/type/model, so
// the list must carry configuration objects rather than bare IDs.
func TestStudio_ListAgentsReturnsConfigurations(t *testing.T) {
	live := startServer(t, nil)

	status, body := live.do(t, http.MethodGet, "/api/v1/agents", nil)
	require.Equal(t, http.StatusOK, status)

	var listed struct {
		Agents []map[string]interface{} `json:"agents"`
	}
	decode(t, body, &listed)
	require.Len(t, listed.Agents, 1, "body was: %s", string(body))

	got := listed.Agents[0]
	for _, field := range []string{"id", "name", "type", "model", "provider", "temperature", "max_tokens", "max_iterations", "tools", "enable_streaming", "timeout"} {
		assert.Contains(t, got, field, "Studio's AgentConfig requires %q", field)
	}
	assert.Equal(t, "studio-agent", got["id"])
	assert.Equal(t, "Studio Agent", got["name"])
	assert.Equal(t, "chat", got["type"])
}

// Studio: client.getAgent(id) reads res.agent.
func TestStudio_GetAgent(t *testing.T) {
	live := startServer(t, nil)

	status, body := live.do(t, http.MethodGet, "/api/v1/agents/studio-agent", nil)
	require.Equal(t, http.StatusOK, status)

	var wrapper struct {
		Agent map[string]interface{} `json:"agent"`
	}
	decode(t, body, &wrapper)
	assert.Equal(t, "studio-agent", wrapper.Agent["id"])

	status, body = live.do(t, http.MethodGet, "/api/v1/agents/missing", nil)
	assert.Equal(t, http.StatusNotFound, status)
	var apiErr struct {
		Error string `json:"error"`
	}
	decode(t, body, &apiErr)
	assert.NotEmpty(t, apiErr.Error, "Studio's ApiError reads the error field")
}

// Studio: client.listTools() reads res.tools as string[].
func TestStudio_ListTools(t *testing.T) {
	live := startServer(t, nil)

	status, body := live.do(t, http.MethodGet, "/api/v1/tools", nil)
	require.Equal(t, http.StatusOK, status)

	var listed struct {
		Tools []string `json:"tools"`
	}
	decode(t, body, &listed)
	assert.Contains(t, listed.Tools, "calculator")
}

// Studio: client.listProviders() reads res.providers as ProviderInfo[], with a
// name field. Credentials must never appear.
func TestStudio_ListProvidersReturnsDescriptions(t *testing.T) {
	live := startServer(t, nil)

	status, body := live.do(t, http.MethodGet, "/api/v1/providers", nil)
	require.Equal(t, http.StatusOK, status)

	var listed struct {
		Providers []map[string]interface{} `json:"providers"`
	}
	decode(t, body, &listed)
	require.Len(t, listed.Providers, 1, "body was: %s", string(body))

	got := listed.Providers[0]
	assert.Equal(t, "fake", got["name"], "Studio's ProviderInfo requires name")
	assert.Equal(t, "fake-model", got["model"])

	assert.NotContains(t, got, "api_key", "provider credentials must never be served")
	assert.NotContains(t, string(body), "super-secret-key")
}

// Studio: client.executeAgent(id, input) reads res.execution.
func TestStudio_ExecuteAgent(t *testing.T) {
	live := startServer(t, nil)
	live.provider.Script("hello from the model")

	status, body := live.do(t, http.MethodPost, "/api/v1/agents/studio-agent/execute",
		map[string]string{"input": "hi"})
	require.Equal(t, http.StatusOK, status, "body was: %s", string(body))

	var wrapper struct {
		Execution map[string]interface{} `json:"execution"`
	}
	decode(t, body, &wrapper)
	require.NotEmpty(t, wrapper.Execution, "body was: %s", string(body))

	// The Go struct tags every field, so the wire format is snake_case like the
	// rest of the API. Studio's AgentExecution type mirrors exactly these names.
	for _, field := range []string{"id", "timestamp", "input", "output", "duration", "success", "execution_path"} {
		assert.Contains(t, wrapper.Execution, field, "Studio's AgentExecution requires %q", field)
	}
	assert.Equal(t, true, wrapper.Execution["success"])
	assert.Equal(t, "hi", wrapper.Execution["input"])
	assert.Equal(t, "hello from the model", wrapper.Execution["output"])

	// Studio highlights the nodes that ran from execution_path; an empty list
	// leaves its debug view blank for a run that did execute.
	path, ok := wrapper.Execution["execution_path"].([]interface{})
	require.True(t, ok, "execution_path must be a list: %s", string(body))
	assert.NotEmpty(t, path, "the nodes that ran must be reported to the debugger")
}

// Studio: client.getAgentHistory(id) reads res.history.
func TestStudio_AgentHistory(t *testing.T) {
	live := startServer(t, nil)

	status, _ := live.do(t, http.MethodPost, "/api/v1/agents/studio-agent/execute",
		map[string]string{"input": "remember me"})
	require.Equal(t, http.StatusOK, status)

	status, body := live.do(t, http.MethodGet, "/api/v1/agents/studio-agent/history", nil)
	require.Equal(t, http.StatusOK, status)

	var wrapper struct {
		History []map[string]interface{} `json:"history"`
	}
	decode(t, body, &wrapper)
	require.NotEmpty(t, wrapper.History, "an executed agent must have history: %s", string(body))
	assert.Equal(t, "remember me", wrapper.History[0]["input"])
}

// Studio: client.getGraphTopology(id) reads res.topology.nodes / .edges, and
// maps n.id, n.name, n.type, e.from and e.to.
func TestStudio_GraphTopologyShape(t *testing.T) {
	live := startServer(t, nil)

	status, body := live.do(t, http.MethodGet, "/api/v1/graphs/studio-workflow/topology", nil)
	require.Equal(t, http.StatusOK, status)

	var resp struct {
		GraphID  string `json:"graph_id"`
		Topology struct {
			Nodes []map[string]interface{} `json:"nodes"`
			Edges []map[string]interface{} `json:"edges"`
		} `json:"topology"`
	}
	decode(t, body, &resp)

	assert.Equal(t, "studio-workflow", resp.GraphID)
	require.NotEmpty(t, resp.Topology.Nodes, "Studio falls back to a synthetic graph when nodes are empty")
	require.NotEmpty(t, resp.Topology.Edges)

	for _, n := range resp.Topology.Nodes {
		assert.Contains(t, n, "id")
		assert.Contains(t, n, "name")
		assert.Contains(t, n, "type")
	}
	for _, e := range resp.Topology.Edges {
		assert.Contains(t, e, "from")
		assert.Contains(t, e, "to")
	}

	// Conditional destinations must be present, or the rendered graph is wrong.
	var sawLong, sawShort bool
	for _, e := range resp.Topology.Edges {
		if e["from"] == "decide" && e["to"] == "long" {
			sawLong = true
		}
		if e["from"] == "decide" && e["to"] == "short" {
			sawShort = true
		}
	}
	assert.True(t, sawLong && sawShort, "both conditional routes must appear: %v", resp.Topology.Edges)
}

// Studio requests a topology using the *agent* ID, so an agent's execution
// graph must resolve through the same endpoint.
func TestStudio_GraphTopologyResolvesAgentID(t *testing.T) {
	live := startServer(t, nil)

	status, body := live.do(t, http.MethodGet, "/api/v1/graphs/studio-agent/topology", nil)
	require.Equal(t, http.StatusOK, status, "body was: %s", string(body))

	var resp struct {
		Topology struct {
			Nodes []map[string]interface{} `json:"nodes"`
		} `json:"topology"`
	}
	decode(t, body, &resp)
	assert.NotEmpty(t, resp.Topology.Nodes,
		"an agent's own graph must be reachable by its ID, or Studio's graph view is always empty")
}

// ---------------------------------------------------------------------------
// Cross-origin and authentication, as a browser-hosted Studio experiences them
// ---------------------------------------------------------------------------

func TestStudio_CORSPreflightAndRequest(t *testing.T) {
	origin := "http://localhost:3000"
	live := startServer(t, func(c *server.ServerConfig) {
		c.Security.AllowedOrigins = []string{origin}
	})

	// Preflight for the JSON POST Studio makes.
	req, err := http.NewRequest(http.MethodOptions, live.baseURL+"/api/v1/agents/studio-agent/execute", nil)
	require.NoError(t, err)
	req.Header.Set("Origin", origin)
	req.Header.Set("Access-Control-Request-Method", "POST")
	req.Header.Set("Access-Control-Request-Headers", "content-type,x-api-key")

	resp, err := http.DefaultClient.Do(req)
	require.NoError(t, err)
	defer func() { _ = resp.Body.Close() }()

	require.Less(t, resp.StatusCode, 300, "preflight must succeed or the browser blocks every call")
	assert.Equal(t, origin, resp.Header.Get("Access-Control-Allow-Origin"))
	assert.Contains(t, resp.Header.Get("Access-Control-Allow-Headers"), "X-API-Key")
	assert.Contains(t, resp.Header.Get("Access-Control-Allow-Methods"), "POST")
}

func TestStudio_AuthenticatedSession(t *testing.T) {
	live := startServer(t, func(c *server.ServerConfig) {
		c.Security.RequireAuth = true
		c.Security.APIKeys = []string{"studio-key"}
	})

	// With the key, Studio works.
	status, _ := live.do(t, http.MethodGet, "/api/v1/agents", nil)
	assert.Equal(t, http.StatusOK, status)

	// Without it, the server refuses.
	req, err := http.NewRequest(http.MethodGet, live.baseURL+"/api/v1/agents", nil)
	require.NoError(t, err)
	resp, err := http.DefaultClient.Do(req)
	require.NoError(t, err)
	defer func() { _ = resp.Body.Close() }()
	assert.Equal(t, http.StatusUnauthorized, resp.StatusCode)

	// Health stays reachable so the connection screen can probe the server.
	probe, err := http.Get(live.baseURL + "/api/v1/health")
	require.NoError(t, err)
	defer func() { _ = probe.Body.Close() }()
	assert.Equal(t, http.StatusOK, probe.StatusCode)
}

// ---------------------------------------------------------------------------
// Debugging a workflow, which is what Studio exists to do
// ---------------------------------------------------------------------------

func TestStudio_ExecuteGraphReturnsPerStepDebugInfo(t *testing.T) {
	live := startServer(t, nil)

	status, body := live.do(t, http.MethodPost, "/api/v1/graphs/studio-workflow/execute",
		map[string]interface{}{"input": "a longer message"})
	require.Equal(t, http.StatusOK, status, "body was: %s", string(body))

	var resp struct {
		Status string                 `json:"status"`
		State  map[string]interface{} `json:"state"`
		Steps  []struct {
			NodeID  string                 `json:"node_id"`
			Step    int                    `json:"step"`
			Success bool                   `json:"success"`
			State   map[string]interface{} `json:"state"`
		} `json:"steps"`
	}
	decode(t, body, &resp)

	assert.Equal(t, "completed", resp.Status)
	assert.Equal(t, "long", resp.State["result"], "conditional routing must pick the long path")

	require.Len(t, resp.Steps, 3, "a debugger needs every node visit: %v", resp.Steps)
	assert.Equal(t, []string{"ingest", "decide", "long"},
		[]string{resp.Steps[0].NodeID, resp.Steps[1].NodeID, resp.Steps[2].NodeID})

	for i, step := range resp.Steps {
		assert.True(t, step.Success)
		assert.Equal(t, i, step.Step)
		assert.NotEmpty(t, step.State, "each step must carry the state after it, for step-through debugging")
	}
}

// A live graph run streamed over WebSocket, the way an interactive debugger
// would drive it.
func TestStudio_WebSocketGraphRun(t *testing.T) {
	live := startServer(t, nil)

	wsURL := "ws" + strings.TrimPrefix(live.baseURL, "http") + "/api/v1/ws/graphs/studio-workflow/stream"
	conn, resp, err := websocket.DefaultDialer.Dial(wsURL, nil)
	if err != nil && resp != nil {
		t.Fatalf("dial failed: %v (status %s)", err, resp.Status)
	}
	require.NoError(t, err)
	defer func() { _ = conn.Close() }()

	require.NoError(t, conn.WriteJSON(map[string]interface{}{
		"type": "execute", "input": "tiny",
	}))
	require.NoError(t, conn.SetReadDeadline(time.Now().Add(20*time.Second)))

	var kinds, nodes []string
	var finalState map[string]interface{}
	for {
		var msg map[string]interface{}
		require.NoError(t, conn.ReadJSON(&msg))
		kind, _ := msg["type"].(string)
		kinds = append(kinds, kind)

		if kind == "step" {
			step := msg["step"].(map[string]interface{})
			nodes = append(nodes, step["node_id"].(string))
		}
		if kind == "complete" {
			finalState, _ = msg["state"].(map[string]interface{})
			break
		}
		if kind == "error" {
			t.Fatalf("unexpected error frame: %v", msg)
		}
	}

	assert.Equal(t, "start", kinds[0])
	assert.Equal(t, []string{"ingest", "decide", "short"}, nodes,
		"a short input must take the short branch")
	require.NotNil(t, finalState)
	assert.Equal(t, "short", finalState["result"])
}

// A failing provider must surface as a readable error, not a hang or a 500 with
// no explanation.
func TestStudio_ProviderFailureIsReported(t *testing.T) {
	live := startServer(t, nil)
	live.provider.FailWith(fmt.Errorf("model backend is offline"))

	status, body := live.do(t, http.MethodPost, "/api/v1/agents/studio-agent/execute",
		map[string]string{"input": "hi"})

	assert.GreaterOrEqual(t, status, 400, "a provider failure must not report success")
	assert.Contains(t, strings.ToLower(string(body)), "offline",
		"Studio shows the server's error field to the user: %s", string(body))
	assert.NotContains(t, string(body), "goroutine ", "stack traces must not reach the console")
}

// A failed execution must carry a readable reason. A Go error field marshals to
// an empty object, which would leave Studio showing a failure with no cause.
func TestStudio_FailedExecutionCarriesReason(t *testing.T) {
	live := startServer(t, nil)
	live.provider.FailWith(fmt.Errorf("context length exceeded"))

	_, body := live.do(t, http.MethodPost, "/api/v1/agents/studio-agent/execute",
		map[string]string{"input": "hi"})

	// Whether the failure is reported on the execution object or as a top-level
	// error, the reason itself must survive serialisation.
	assert.Contains(t, string(body), "context length exceeded",
		"the failure reason must reach the client: %s", string(body))
	assert.NotContains(t, string(body), `"error":{}`,
		"a Go error must not serialise as an empty object")
}
