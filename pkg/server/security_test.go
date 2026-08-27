// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

package server

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/piotrlaczkowski/GoLangGraph/pkg/core"
	"github.com/piotrlaczkowski/GoLangGraph/pkg/persistence"
	"github.com/sirupsen/logrus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// newTestServer builds a server with a registered graph for API tests.
func newTestServer(t *testing.T, mutate func(*ServerConfig)) *Server {
	t.Helper()
	cfg := DefaultServerConfig()
	cfg.Port = 0
	// No static catch-all: it would shadow routes registered by individual tests.
	cfg.StaticDir = ""
	if mutate != nil {
		mutate(cfg)
	}
	// The logger level now comes from cfg.LogLevel; overriding it here would
	// mask whether that configuration is actually applied.
	s := NewServer(cfg)

	g := core.NewGraph("demo")
	g.Config.EnableStreaming = false
	g.AddNode("greet", "Greet", func(ctx context.Context, st *core.BaseState) (*core.BaseState, error) {
		in, _ := st.Get("input")
		st.Set("greeting", fmt.Sprintf("hello %v", in))
		return st, nil
	})
	g.AddNode("finish", "Finish", func(ctx context.Context, st *core.BaseState) (*core.BaseState, error) {
		st.Set("done", true)
		return st, nil
	})
	g.AddEdge("greet", "finish", nil)
	require.NoError(t, g.SetStartNode("greet"))
	require.NoError(t, g.AddEndNode("finish"))
	s.GraphManager().Register("demo", g)

	return s
}

func doRequest(t *testing.T, s *Server, method, path string, body interface{}, headers map[string]string) *httptest.ResponseRecorder {
	t.Helper()
	var reader *bytes.Reader
	if body != nil {
		raw, err := json.Marshal(body)
		require.NoError(t, err)
		reader = bytes.NewReader(raw)
	} else {
		reader = bytes.NewReader(nil)
	}
	req := httptest.NewRequest(method, path, reader)
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	rec := httptest.NewRecorder()
	s.router.ServeHTTP(rec, req)
	return rec
}

// ---------------------------------------------------------------------------
// Authentication
// ---------------------------------------------------------------------------

func TestServer_AuthRequiredRejectsMissingKey(t *testing.T) {
	s := newTestServer(t, func(c *ServerConfig) {
		c.Security.RequireAuth = true
		c.Security.APIKeys = []string{"secret-key"}
	})

	rec := doRequest(t, s, http.MethodGet, "/api/v1/graphs", nil, nil)
	assert.Equal(t, http.StatusUnauthorized, rec.Code)
	assert.Contains(t, rec.Body.String(), "error")

	rec = doRequest(t, s, http.MethodGet, "/api/v1/graphs", nil, map[string]string{"X-API-Key": "wrong"})
	assert.Equal(t, http.StatusUnauthorized, rec.Code)

	rec = doRequest(t, s, http.MethodGet, "/api/v1/graphs", nil, map[string]string{"X-API-Key": "secret-key"})
	assert.Equal(t, http.StatusOK, rec.Code)
}

func TestServer_HealthIsPublicUnderAuth(t *testing.T) {
	s := newTestServer(t, func(c *ServerConfig) {
		c.Security.RequireAuth = true
		c.Security.APIKeys = []string{"k"}
	})
	rec := doRequest(t, s, http.MethodGet, "/api/v1/health", nil, nil)
	assert.Equal(t, http.StatusOK, rec.Code, "probes must not need credentials")
}

func TestServer_AuthDisabledByDefault(t *testing.T) {
	s := newTestServer(t, nil)
	rec := doRequest(t, s, http.MethodGet, "/api/v1/graphs", nil, nil)
	assert.Equal(t, http.StatusOK, rec.Code)
}

func TestServer_AuthAcceptsAnyConfiguredKey(t *testing.T) {
	s := newTestServer(t, func(c *ServerConfig) {
		c.Security.RequireAuth = true
		c.Security.APIKeys = []string{"first", "second"}
	})
	for _, key := range []string{"first", "second"} {
		rec := doRequest(t, s, http.MethodGet, "/api/v1/graphs", nil, map[string]string{"X-API-Key": key})
		assert.Equal(t, http.StatusOK, rec.Code, "key %q must be accepted", key)
	}
}

// An empty key list with auth on must fail closed rather than allowing all.
func TestServer_AuthWithNoKeysFailsClosed(t *testing.T) {
	s := newTestServer(t, func(c *ServerConfig) {
		c.Security.RequireAuth = true
		c.Security.APIKeys = nil
	})
	rec := doRequest(t, s, http.MethodGet, "/api/v1/graphs", nil, map[string]string{"X-API-Key": "anything"})
	assert.Equal(t, http.StatusUnauthorized, rec.Code)
}

// ---------------------------------------------------------------------------
// CORS and origin handling
// ---------------------------------------------------------------------------

func TestServer_CORSRestrictsOrigins(t *testing.T) {
	s := newTestServer(t, func(c *ServerConfig) {
		c.Security.AllowedOrigins = []string{"https://studio.example.com"}
	})

	rec := doRequest(t, s, http.MethodGet, "/api/v1/health", nil,
		map[string]string{"Origin": "https://studio.example.com"})
	assert.Equal(t, "https://studio.example.com", rec.Header().Get("Access-Control-Allow-Origin"))
	assert.Contains(t, rec.Header().Get("Vary"), "Origin")

	rec = doRequest(t, s, http.MethodGet, "/api/v1/health", nil,
		map[string]string{"Origin": "https://evil.example.com"})
	assert.Empty(t, rec.Header().Get("Access-Control-Allow-Origin"),
		"a disallowed origin must not receive CORS approval")
}

func TestServer_CORSPreflightRejectsDisallowedOrigin(t *testing.T) {
	s := newTestServer(t, func(c *ServerConfig) {
		c.Security.AllowedOrigins = []string{"https://ok.example.com"}
	})
	rec := doRequest(t, s, http.MethodOptions, "/api/v1/graphs", nil,
		map[string]string{"Origin": "https://evil.example.com"})
	assert.Equal(t, http.StatusForbidden, rec.Code)
}

func TestServer_CORSAllowsAPIKeyHeader(t *testing.T) {
	s := newTestServer(t, nil)
	rec := doRequest(t, s, http.MethodOptions, "/api/v1/graphs", nil,
		map[string]string{"Origin": "http://localhost:5173"})
	assert.Equal(t, http.StatusOK, rec.Code)
	assert.Contains(t, rec.Header().Get("Access-Control-Allow-Headers"), "X-API-Key",
		"Studio authenticates with X-API-Key, so preflight must permit it")
}

func TestServer_WebSocketOriginCheck(t *testing.T) {
	s := newTestServer(t, func(c *ServerConfig) {
		c.Security.AllowedOrigins = []string{"https://studio.example.com"}
	})

	allowed := httptest.NewRequest(http.MethodGet, "/api/v1/ws/graphs/demo/stream", nil)
	allowed.Header.Set("Origin", "https://studio.example.com")
	assert.True(t, s.upgrader.CheckOrigin(allowed))

	denied := httptest.NewRequest(http.MethodGet, "/api/v1/ws/graphs/demo/stream", nil)
	denied.Header.Set("Origin", "https://evil.example.com")
	assert.False(t, s.upgrader.CheckOrigin(denied),
		"accepting any origin allows cross-site WebSocket hijacking")
}

// ---------------------------------------------------------------------------
// Request hygiene
// ---------------------------------------------------------------------------

func TestServer_RejectsOversizedBody(t *testing.T) {
	s := newTestServer(t, func(c *ServerConfig) {
		c.Security.MaxRequestBytes = 1024
	})

	huge := strings.Repeat("a", 8192)
	req := httptest.NewRequest(http.MethodPost, "/api/v1/graphs/demo/execute",
		strings.NewReader(`{"input":"`+huge+`"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	s.router.ServeHTTP(rec, req)

	assert.NotEqual(t, http.StatusOK, rec.Code, "an oversized body must be rejected")
}

func TestServer_MalformedJSONIsRejected(t *testing.T) {
	s := newTestServer(t, nil)
	for _, body := range []string{"{", "", "null", "[]", `{"input":`, `{"input": 12345}`} {
		req := httptest.NewRequest(http.MethodPost, "/api/v1/graphs/demo/execute", strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()
		s.router.ServeHTTP(rec, req)
		assert.NotEqual(t, http.StatusInternalServerError, rec.Code,
			"malformed body %q must not produce a server error", body)
	}
}

func TestServer_SecurityHeadersArePresent(t *testing.T) {
	s := newTestServer(t, nil)
	rec := doRequest(t, s, http.MethodGet, "/api/v1/health", nil, nil)
	assert.Equal(t, "nosniff", rec.Header().Get("X-Content-Type-Options"))
	assert.Equal(t, "DENY", rec.Header().Get("X-Frame-Options"))
}

func TestServer_PanicInHandlerReturns500(t *testing.T) {
	s := newTestServer(t, nil)
	s.router.HandleFunc("/boom", func(w http.ResponseWriter, r *http.Request) {
		panic("handler exploded")
	})

	rec := doRequest(t, s, http.MethodGet, "/boom", nil, nil)
	assert.Equal(t, http.StatusInternalServerError, rec.Code)
	assert.NotContains(t, rec.Body.String(), "goroutine", "stack traces must not reach clients")
	assert.Contains(t, rec.Body.String(), "error")
}

// ---------------------------------------------------------------------------
// Graph API
// ---------------------------------------------------------------------------

func TestServer_GraphListAndTopology(t *testing.T) {
	s := newTestServer(t, nil)

	rec := doRequest(t, s, http.MethodGet, "/api/v1/graphs", nil, nil)
	require.Equal(t, http.StatusOK, rec.Code)
	var listed struct {
		Graphs []GraphSummaryView `json:"graphs"`
	}
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &listed))
	require.Len(t, listed.Graphs, 1)
	assert.Equal(t, "demo", listed.Graphs[0].ID)
	assert.Equal(t, 2, listed.Graphs[0].NodeCount)

	rec = doRequest(t, s, http.MethodGet, "/api/v1/graphs/demo/topology", nil, nil)
	require.Equal(t, http.StatusOK, rec.Code)
	var topo struct {
		GraphID  string            `json:"graph_id"`
		Topology GraphTopologyView `json:"topology"`
	}
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &topo))
	assert.Equal(t, "demo", topo.GraphID)
	require.Len(t, topo.Topology.Nodes, 2, "topology must report real nodes, not a placeholder")
	require.Len(t, topo.Topology.Edges, 1)
	assert.Equal(t, "greet", topo.Topology.Edges[0].From)
	assert.Equal(t, "finish", topo.Topology.Edges[0].To)

	// Nodes carry the flags a visualiser needs.
	byID := map[string]GraphNodeView{}
	for _, n := range topo.Topology.Nodes {
		byID[n.ID] = n
	}
	assert.True(t, byID["greet"].IsStart)
	assert.True(t, byID["finish"].IsEnd)
}

func TestServer_GraphNotFound(t *testing.T) {
	s := newTestServer(t, nil)
	for _, path := range []string{
		"/api/v1/graphs/missing",
		"/api/v1/graphs/missing/topology",
	} {
		rec := doRequest(t, s, http.MethodGet, path, nil, nil)
		assert.Equal(t, http.StatusNotFound, rec.Code, "path %s", path)
	}
	rec := doRequest(t, s, http.MethodPost, "/api/v1/graphs/missing/execute",
		map[string]string{"input": "x"}, nil)
	assert.Equal(t, http.StatusNotFound, rec.Code)
}

func TestServer_GraphExecuteReturnsRealResult(t *testing.T) {
	s := newTestServer(t, nil)

	rec := doRequest(t, s, http.MethodPost, "/api/v1/graphs/demo/execute",
		map[string]interface{}{"input": "world"}, nil)
	require.Equal(t, http.StatusOK, rec.Code)

	var resp struct {
		GraphID string                 `json:"graph_id"`
		Status  string                 `json:"status"`
		State   map[string]interface{} `json:"state"`
		Steps   []ExecutionStepView    `json:"steps"`
	}
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))

	assert.Equal(t, "completed", resp.Status)
	assert.Equal(t, "hello world", resp.State["greeting"],
		"execution must actually run the graph, not return a placeholder")
	assert.Equal(t, true, resp.State["done"])
	require.Len(t, resp.Steps, 2, "every executed node must be reported")
	assert.Equal(t, "greet", resp.Steps[0].NodeID)
	assert.Equal(t, "finish", resp.Steps[1].NodeID)
}

func TestServer_GraphExecuteReportsFailure(t *testing.T) {
	s := newTestServer(t, nil)
	failing := core.NewGraph("failing")
	failing.Config.EnableStreaming = false
	failing.AddNode("bad", "Bad", func(ctx context.Context, st *core.BaseState) (*core.BaseState, error) {
		return nil, fmt.Errorf("provider unavailable")
	})
	require.NoError(t, failing.SetStartNode("bad"))
	s.GraphManager().Register("failing", failing)

	rec := doRequest(t, s, http.MethodPost, "/api/v1/graphs/failing/execute",
		map[string]interface{}{"input": "x"}, nil)
	assert.Equal(t, http.StatusInternalServerError, rec.Code)

	var resp map[string]interface{}
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))
	assert.Equal(t, "failed", resp["status"])
	assert.Contains(t, resp["error"], "provider unavailable")
	assert.NotContains(t, fmt.Sprint(resp["error"]), "goroutine")
}

func TestServer_GraphExecuteReportsInterruptAsResumable(t *testing.T) {
	s := newTestServer(t, nil)
	g := core.NewGraph("pausing")
	g.Config.EnableStreaming = false
	g.Config.InterruptBefore = []string{"second"}
	g.AddNode("first", "First", func(ctx context.Context, st *core.BaseState) (*core.BaseState, error) {
		st.Set("first", true)
		return st, nil
	})
	g.AddNode("second", "Second", func(ctx context.Context, st *core.BaseState) (*core.BaseState, error) {
		st.Set("second", true)
		return st, nil
	})
	g.AddEdge("first", "second", nil)
	require.NoError(t, g.SetStartNode("first"))
	require.NoError(t, g.AddEndNode("second"))
	s.GraphManager().Register("pausing", g)

	rec := doRequest(t, s, http.MethodPost, "/api/v1/graphs/pausing/execute",
		map[string]interface{}{"input": "x"}, nil)
	require.Equal(t, http.StatusOK, rec.Code, "a pause is a normal outcome, not a server error")

	var resp map[string]interface{}
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))
	assert.Equal(t, "interrupted", resp["status"])
	interrupt := resp["interrupt"].(map[string]interface{})
	assert.Equal(t, "second", interrupt["node_id"])
	assert.Equal(t, true, interrupt["before"])
	state := resp["state"].(map[string]interface{})
	assert.Equal(t, true, state["first"])
	assert.NotContains(t, state, "second")
}

func TestServer_GraphExecuteAcceptsStateSeed(t *testing.T) {
	s := newTestServer(t, nil)
	rec := doRequest(t, s, http.MethodPost, "/api/v1/graphs/demo/execute",
		map[string]interface{}{"state": map[string]interface{}{"input": "seeded", "extra": 1}}, nil)
	require.Equal(t, http.StatusOK, rec.Code)

	var resp map[string]interface{}
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))
	state := resp["state"].(map[string]interface{})
	assert.Equal(t, "hello seeded", state["greeting"])
	assert.EqualValues(t, 1, state["extra"], "caller-supplied keys must reach the graph")
}

// Concurrent API executions must not interfere with each other.
func TestServer_ConcurrentGraphExecutions(t *testing.T) {
	s := newTestServer(t, nil)

	const n = 24
	var wg sync.WaitGroup
	results := make([]string, n)
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			rec := doRequest(t, s, http.MethodPost, "/api/v1/graphs/demo/execute",
				map[string]interface{}{"input": fmt.Sprintf("client-%d", i)}, nil)
			if rec.Code != http.StatusOK {
				results[i] = "status " + rec.Result().Status
				return
			}
			var resp map[string]interface{}
			if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
				results[i] = err.Error()
				return
			}
			state := resp["state"].(map[string]interface{})
			want := fmt.Sprintf("hello client-%d", i)
			if state["greeting"] != want {
				results[i] = fmt.Sprintf("got %v want %v", state["greeting"], want)
			}
		}(i)
	}
	wg.Wait()

	for i, r := range results {
		assert.Empty(t, r, "request %d", i)
	}
}

// A cancelled request must stop the run rather than finishing it.
func TestServer_RequestCancellationStopsExecution(t *testing.T) {
	s := newTestServer(t, nil)
	started := make(chan struct{})
	var once sync.Once
	slow := core.NewGraph("slow")
	slow.Config.EnableStreaming = false
	slow.AddNode("wait", "Wait", func(ctx context.Context, st *core.BaseState) (*core.BaseState, error) {
		once.Do(func() { close(started) })
		<-ctx.Done()
		return nil, ctx.Err()
	})
	require.NoError(t, slow.SetStartNode("wait"))
	s.GraphManager().Register("slow", slow)

	ctx, cancel := context.WithCancel(context.Background())
	req := httptest.NewRequest(http.MethodPost, "/api/v1/graphs/slow/execute",
		strings.NewReader(`{"input":"x"}`)).WithContext(ctx)
	rec := httptest.NewRecorder()

	done := make(chan struct{})
	go func() {
		defer close(done)
		s.router.ServeHTTP(rec, req)
	}()

	<-started
	cancel()

	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("cancelled request did not stop the graph run")
	}
	assert.Equal(t, http.StatusRequestTimeout, rec.Code)
}

func TestServer_StopIsSafeBeforeStart(t *testing.T) {
	s := newTestServer(t, nil)
	assert.NoError(t, s.Stop(context.Background()), "Stop before Start must not panic")
}

// ---------------------------------------------------------------------------
// Thread history and diagnostics
// ---------------------------------------------------------------------------

// The checkpoints endpoint previously returned an empty list unconditionally,
// so a thread's history was invisible even when checkpoints existed.
func TestServer_ListCheckpointsReturnsRealHistory(t *testing.T) {
	s := newTestServer(t, nil)

	cp := persistence.NewMemoryCheckpointer()
	s.SetCheckpointer(cp)

	for step, node := range []string{"a", "b", "c"} {
		st := core.NewBaseState()
		st.Set("step", step)
		require.NoError(t, cp.Save(context.Background(), &persistence.Checkpoint{
			ID:        fmt.Sprintf("cp-%d", step),
			ThreadID:  "thread-1",
			NodeID:    node,
			StepID:    step,
			State:     st,
			CreatedAt: time.Now().Add(time.Duration(step) * time.Second),
		}))
	}

	rec := doRequest(t, s, http.MethodGet, "/api/v1/threads/thread-1/checkpoints", nil, nil)
	require.Equal(t, http.StatusOK, rec.Code)

	var resp struct {
		ThreadID    string `json:"thread_id"`
		Checkpoints []struct {
			ID     string `json:"id"`
			NodeID string `json:"node_id"`
			StepID int    `json:"step_id"`
		} `json:"checkpoints"`
	}
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))

	assert.Equal(t, "thread-1", resp.ThreadID)
	require.Len(t, resp.Checkpoints, 3, "the thread's checkpoints must be listed")
	assert.Equal(t, []string{"a", "b", "c"},
		[]string{resp.Checkpoints[0].NodeID, resp.Checkpoints[1].NodeID, resp.Checkpoints[2].NodeID},
		"checkpoints must be ordered oldest first so a client can replay them")
}

func TestServer_ListCheckpointsWithoutCheckpointer(t *testing.T) {
	s := newTestServer(t, nil)
	rec := doRequest(t, s, http.MethodGet, "/api/v1/threads/t/checkpoints", nil, nil)
	assert.Equal(t, http.StatusServiceUnavailable, rec.Code,
		"a missing checkpointer must be reported, not reported as an empty history")
}

func TestServer_ListCheckpointsRejectsUnsafeThreadID(t *testing.T) {
	s := newTestServer(t, nil)
	s.SetCheckpointer(persistence.NewFileCheckpointer(t.TempDir()))

	rec := doRequest(t, s, http.MethodGet, "/api/v1/threads/..%2F..%2Fetc/checkpoints", nil, nil)
	assert.NotEqual(t, http.StatusOK, rec.Code)
}

// Metrics must be measured, not hardcoded.
func TestServer_DebugMetricsAreMeasured(t *testing.T) {
	s := newTestServer(t, func(c *ServerConfig) { c.DevMode = true })

	// Generate some traffic, including a failure.
	for i := 0; i < 3; i++ {
		doRequest(t, s, http.MethodGet, "/api/v1/health", nil, nil)
	}
	doRequest(t, s, http.MethodGet, "/api/v1/graphs/missing", nil, nil)

	rec := doRequest(t, s, http.MethodGet, "/debug/metrics", nil, nil)
	require.Equal(t, http.StatusOK, rec.Code)

	var resp struct {
		Metrics map[string]interface{} `json:"metrics"`
	}
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))

	total, ok := resp.Metrics["requests_total"].(float64)
	require.True(t, ok)
	assert.Greater(t, total, float64(3), "requests must actually be counted")

	failed, ok := resp.Metrics["requests_failed"].(float64)
	require.True(t, ok)
	assert.GreaterOrEqual(t, failed, float64(1), "failures must be counted")

	alloc, ok := resp.Metrics["memory_alloc_bytes"].(float64)
	require.True(t, ok, "memory must be measured, not reported as N/A")
	assert.Greater(t, alloc, float64(0))

	assert.Contains(t, resp.Metrics, "goroutines")
	assert.Contains(t, resp.Metrics, "uptime_seconds")
	assert.EqualValues(t, 1, resp.Metrics["graphs_registered"])
}

// Metrics must not panic when no agent manager is configured.
func TestServer_DebugMetricsWithoutAgentManager(t *testing.T) {
	s := newTestServer(t, func(c *ServerConfig) { c.DevMode = true })
	s.SetAgentManager(nil)

	rec := doRequest(t, s, http.MethodGet, "/debug/metrics", nil, nil)
	assert.Equal(t, http.StatusOK, rec.Code)
}

// An endpoint that cannot do what it claims must say so rather than report
// success.
func TestServer_UnimplementedDebugEndpointsAreHonest(t *testing.T) {
	s := newTestServer(t, func(c *ServerConfig) { c.DevMode = true })

	reload := doRequest(t, s, http.MethodPost, "/debug/reload", nil, nil)
	assert.Equal(t, http.StatusNotImplemented, reload.Code)
	assert.NotContains(t, reload.Body.String(), "successfully",
		"an operator must not be told a reload happened when none did")

	logs := doRequest(t, s, http.MethodGet, "/debug/logs", nil, nil)
	assert.Equal(t, http.StatusNotImplemented, logs.Code)
}

// Concurrent traffic must not race the metrics counters or the connection map.
func TestServer_MetricsAreRaceFree(t *testing.T) {
	s := newTestServer(t, func(c *ServerConfig) { c.DevMode = true })

	var wg sync.WaitGroup
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 10; j++ {
				doRequest(t, s, http.MethodGet, "/api/v1/health", nil, nil)
				doRequest(t, s, http.MethodGet, "/debug/metrics", nil, nil)
			}
		}()
	}
	wg.Wait()
}

// LogLevel was declared in ServerConfig and never read, so an operator setting
// it saw no change in output at all.
func TestServer_LogLevelIsApplied(t *testing.T) {
	for _, tc := range []struct {
		configured string
		want       logrus.Level
	}{
		{"debug", logrus.DebugLevel},
		{"warn", logrus.WarnLevel},
		{"error", logrus.ErrorLevel},
	} {
		t.Run(tc.configured, func(t *testing.T) {
			s := newTestServer(t, func(c *ServerConfig) { c.LogLevel = tc.configured })
			assert.Equal(t, tc.want, s.logger.GetLevel())
		})
	}
}

// An unrecognised level must not change the logger or crash the server.
func TestServer_InvalidLogLevelKeepsDefault(t *testing.T) {
	s := newTestServer(t, func(c *ServerConfig) { c.LogLevel = "not-a-level" })
	assert.Equal(t, logrus.InfoLevel, s.logger.GetLevel())
}
