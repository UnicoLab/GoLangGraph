// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package server

import (
	"bytes"
	"context"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/UnicoLab/GoLangGraph/pkg/agent"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// The auto-generated server is a second, independent serving path: it is what
// the `auto-serve` CLI command runs. It previously had no authentication of any
// kind, a hardcoded wildcard CORS header, and a MaxRequestSize that was
// configured and never enforced — so hardening Server left all of this open.

func newAutoServer(t *testing.T, mutate func(*AutoServerConfig)) *AutoServer {
	t.Helper()

	cfg := DefaultAutoServerConfig()
	cfg.Port = 0
	cfg.EnableWebUI = false
	cfg.EnablePlayground = false
	if mutate != nil {
		mutate(cfg)
	}

	// Each test gets its own registry: the default is process-wide, so tests
	// sharing it would see each other's agents.
	as := NewAutoServerWithRegistry(cfg, agent.NewAgentRegistry())

	agentCfg := agent.DefaultAgentConfig()
	agentCfg.ID = "auto-agent"
	agentCfg.Name = "Auto Agent"
	agentCfg.Type = agent.AgentTypeChat
	agentCfg.Model = "fake-model"
	agentCfg.Provider = "fake"
	require.NoError(t, as.RegisterAgent("auto-agent", agent.NewBaseAgentDefinition(agentCfg)))
	require.NoError(t, as.GenerateEndpoints())

	return as
}

func autoRequest(t *testing.T, as *AutoServer, method, path string, body string, headers map[string]string) *httptest.ResponseRecorder {
	t.Helper()

	var reader *bytes.Reader
	if body != "" {
		reader = bytes.NewReader([]byte(body))
	} else {
		reader = bytes.NewReader(nil)
	}

	req := httptest.NewRequest(method, path, reader)
	req.Header.Set("Content-Type", "application/json")
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	rec := httptest.NewRecorder()
	as.router.ServeHTTP(rec, req)
	return rec
}

// ---------------------------------------------------------------------------
// Authentication
// ---------------------------------------------------------------------------

func TestAutoServer_AuthRequiredRejectsMissingKey(t *testing.T) {
	as := newAutoServer(t, func(c *AutoServerConfig) {
		c.Security.RequireAuth = true
		c.Security.APIKeys = []string{"auto-key"}
	})

	rec := autoRequest(t, as, http.MethodGet, "/agents", "", nil)
	assert.Equal(t, http.StatusUnauthorized, rec.Code,
		"the auto server previously served every endpoint unauthenticated")

	rec = autoRequest(t, as, http.MethodGet, "/agents", "", map[string]string{"X-API-Key": "wrong"})
	assert.Equal(t, http.StatusUnauthorized, rec.Code)

	rec = autoRequest(t, as, http.MethodGet, "/agents", "", map[string]string{"X-API-Key": "auto-key"})
	assert.Equal(t, http.StatusOK, rec.Code)
}

func TestAutoServer_HealthIsPublicUnderAuth(t *testing.T) {
	as := newAutoServer(t, func(c *AutoServerConfig) {
		c.Security.RequireAuth = true
		c.Security.APIKeys = []string{"k"}
		c.Security.PublicPaths = []string{"/health"}
	})

	rec := autoRequest(t, as, http.MethodGet, "/health", "", nil)
	assert.Equal(t, http.StatusOK, rec.Code, "probes must not need credentials")
}

func TestAutoServer_AuthDisabledByDefault(t *testing.T) {
	as := newAutoServer(t, nil)
	rec := autoRequest(t, as, http.MethodGet, "/agents", "", nil)
	assert.Equal(t, http.StatusOK, rec.Code)
}

func TestAutoServer_AuthWithNoKeysFailsClosed(t *testing.T) {
	as := newAutoServer(t, func(c *AutoServerConfig) {
		c.Security.RequireAuth = true
		c.Security.APIKeys = nil
	})
	rec := autoRequest(t, as, http.MethodGet, "/agents", "", map[string]string{"X-API-Key": "anything"})
	assert.Equal(t, http.StatusUnauthorized, rec.Code)
}

// ---------------------------------------------------------------------------
// Cross-origin access
// ---------------------------------------------------------------------------

func TestAutoServer_CORSRestrictsOrigins(t *testing.T) {
	as := newAutoServer(t, func(c *AutoServerConfig) {
		c.Security.AllowedOrigins = []string{"https://studio.example.com"}
	})

	rec := autoRequest(t, as, http.MethodGet, "/health", "",
		map[string]string{"Origin": "https://studio.example.com"})
	assert.Equal(t, "https://studio.example.com", rec.Header().Get("Access-Control-Allow-Origin"))
	assert.Contains(t, rec.Header().Get("Vary"), "Origin")

	rec = autoRequest(t, as, http.MethodGet, "/health", "",
		map[string]string{"Origin": "https://evil.example.com"})
	assert.Empty(t, rec.Header().Get("Access-Control-Allow-Origin"),
		"a hardcoded wildcard offered every origin access to every endpoint")
}

// Most routes declare concrete methods, so preflight used to 404 and a browser
// blocked the request that followed.
func TestAutoServer_PreflightIsAnswered(t *testing.T) {
	as := newAutoServer(t, nil)

	for _, path := range []string{"/agents", "/schemas", "/metrics", "/api/agents/auto-agent"} {
		rec := autoRequest(t, as, http.MethodOptions, path, "",
			map[string]string{"Origin": "http://localhost:3000"})
		assert.Less(t, rec.Code, 300, "preflight for %s must succeed, got %d", path, rec.Code)
		assert.Contains(t, rec.Header().Get("Access-Control-Allow-Headers"), "X-API-Key",
			"preflight for %s must permit the auth header", path)
	}
}

func TestAutoServer_PreflightRejectsDisallowedOrigin(t *testing.T) {
	as := newAutoServer(t, func(c *AutoServerConfig) {
		c.Security.AllowedOrigins = []string{"https://ok.example.com"}
	})
	rec := autoRequest(t, as, http.MethodOptions, "/agents", "",
		map[string]string{"Origin": "https://evil.example.com"})
	assert.Equal(t, http.StatusForbidden, rec.Code)
}

// ---------------------------------------------------------------------------
// Request hygiene
// ---------------------------------------------------------------------------

// MaxRequestSize was declared in the configuration and never enforced.
func TestAutoServer_EnforcesMaxRequestSize(t *testing.T) {
	as := newAutoServer(t, func(c *AutoServerConfig) {
		c.MaxRequestSize = 1024
	})

	huge := `{"message":"` + strings.Repeat("a", 8192) + `"}`
	rec := autoRequest(t, as, http.MethodPost, "/validate/auto-agent", huge, nil)
	assert.NotEqual(t, http.StatusOK, rec.Code, "an oversized body must be rejected")
}

func TestAutoServer_SecurityHeadersArePresent(t *testing.T) {
	as := newAutoServer(t, nil)
	rec := autoRequest(t, as, http.MethodGet, "/health", "", nil)
	assert.Equal(t, "nosniff", rec.Header().Get("X-Content-Type-Options"))
	assert.Equal(t, "DENY", rec.Header().Get("X-Frame-Options"))
}

// Recovery was opt-in through a middleware name list, so a deployment that
// omitted it had a panicking handler tear down the connection.
func TestAutoServer_PanicIsRecoveredEvenWithoutOptIn(t *testing.T) {
	as := newAutoServer(t, func(c *AutoServerConfig) {
		c.Middleware = []string{"cors", "logging"} // deliberately no "recovery"
	})
	as.router.HandleFunc("/boom", func(w http.ResponseWriter, r *http.Request) {
		panic("handler exploded")
	})

	rec := autoRequest(t, as, http.MethodGet, "/boom", "", nil)
	assert.Equal(t, http.StatusInternalServerError, rec.Code)
	assert.NotContains(t, rec.Body.String(), "goroutine", "stack traces must not reach clients")
}

func TestAutoServer_MalformedJSONIsRejected(t *testing.T) {
	as := newAutoServer(t, nil)

	for _, body := range []string{"{", "", "null", "[]", `{"message":`} {
		rec := autoRequest(t, as, http.MethodPost, "/validate/auto-agent", body, nil)
		assert.NotEqual(t, http.StatusInternalServerError, rec.Code,
			"malformed body %q must not produce a server error", body)
	}
}

// ---------------------------------------------------------------------------
// Schema validation on this serving path
// ---------------------------------------------------------------------------

func TestAutoServer_ValidationActuallyValidates(t *testing.T) {
	as := newAutoServer(t, nil)

	// A payload missing the required field must be rejected. This endpoint
	// previously answered "valid": true for anything at all.
	rec := autoRequest(t, as, http.MethodPost, "/validate/auto-agent", `{}`, nil)
	require.Equal(t, http.StatusOK, rec.Code)

	var resp struct {
		Valid  bool     `json:"valid"`
		Errors []string `json:"errors"`
	}
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))
	assert.False(t, resp.Valid, "a payload missing a required field must not validate")
	assert.NotEmpty(t, resp.Errors)

	// A conforming payload validates.
	rec = autoRequest(t, as, http.MethodPost, "/validate/auto-agent", `{"message":"hello"}`, nil)
	require.Equal(t, http.StatusOK, rec.Code)
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))
	assert.True(t, resp.Valid, "a conforming payload must validate: %v", resp.Errors)
}

func TestAutoServer_ValidationRejectsUnknownType(t *testing.T) {
	as := newAutoServer(t, nil)
	rec := autoRequest(t, as, http.MethodPost, "/validate/auto-agent?type=sideways", `{"message":"x"}`, nil)
	assert.Equal(t, http.StatusBadRequest, rec.Code)
}

func TestAutoServer_ValidationUnknownAgent(t *testing.T) {
	as := newAutoServer(t, nil)
	rec := autoRequest(t, as, http.MethodPost, "/validate/missing", `{"message":"x"}`, nil)
	assert.Equal(t, http.StatusNotFound, rec.Code)
}

// ---------------------------------------------------------------------------
// Concurrency
// ---------------------------------------------------------------------------

// The metrics middleware increments a shared counter on every request.
func TestAutoServer_ConcurrentRequestsAreRaceFree(t *testing.T) {
	as := newAutoServer(t, nil)

	var wg sync.WaitGroup
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 10; j++ {
				autoRequest(t, as, http.MethodGet, "/health", "", nil)
				autoRequest(t, as, http.MethodGet, "/metrics", "", nil)
			}
		}()
	}
	wg.Wait()
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

// Regenerating endpoints on a live server would rewrite the route table and the
// agent maps while request goroutines read them.
func TestAutoServer_RegenerateWhileRunningIsRefused(t *testing.T) {
	as := newAutoServer(t, nil)

	as.started.Store(true)
	err := as.GenerateEndpoints()
	require.Error(t, err)
	assert.Contains(t, err.Error(), "running")
}

// Two servers must be able to hold different agents. The default registry is
// process-wide, which silently gave one server another's agents.
func TestAutoServer_IsolatedRegistriesDoNotShareAgents(t *testing.T) {
	first := newAutoServer(t, nil)
	second := newAutoServer(t, nil)

	extra := agent.DefaultAgentConfig()
	extra.ID = "second-only"
	extra.Name = "Second Only"
	extra.Type = agent.AgentTypeChat
	extra.Model = "fake-model"
	extra.Provider = "fake"
	require.NoError(t, second.RegisterAgent("second-only", agent.NewBaseAgentDefinition(extra)))

	_, existsInFirst := first.Registry().GetDefinition("second-only")
	assert.False(t, existsInFirst,
		"an agent registered on one server must not appear on another")

	_, existsInSecond := second.Registry().GetDefinition("second-only")
	assert.True(t, existsInSecond)
}

// Reading the agent maps concurrently with the metrics endpoint must be safe.
func TestAutoServer_AgentLookupsAreRaceFree(t *testing.T) {
	as := newAutoServer(t, nil)

	var wg sync.WaitGroup
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 20; j++ {
				_, _ = as.agentInstance("auto-agent")
				_, _ = as.agentMeta("auto-agent")
				_ = as.agentCount()
				_ = as.agentIDs()
				autoRequest(t, as, http.MethodGet, "/agents/auto-agent", "", nil)
			}
		}()
	}
	wg.Wait()
}

// ---------------------------------------------------------------------------
// Start and directory loading
// ---------------------------------------------------------------------------

// Start reported success when the port was already taken: ListenAndServe ran in
// a goroutine that only logged the failure, so Start blocked on ctx.Done() and
// the caller believed the server was up.
func TestAutoServer_StartReportsBindFailure(t *testing.T) {
	occupied, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	defer func() { _ = occupied.Close() }()

	port := occupied.Addr().(*net.TCPAddr).Port

	as := newAutoServer(t, func(c *AutoServerConfig) {
		c.Host = "127.0.0.1"
		c.Port = port
	})

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	startErr := make(chan error, 1)
	go func() { startErr <- as.Start(ctx) }()

	select {
	case err := <-startErr:
		require.Error(t, err, "a taken port must be reported, not swallowed")
		assert.Contains(t, err.Error(), "listen")
	case <-time.After(5 * time.Second):
		t.Fatal("Start blocked instead of reporting that the port was unavailable")
	}
}

// A server started on port 0 must report the port it actually got.
func TestAutoServer_AddressReportsBoundPort(t *testing.T) {
	as := newAutoServer(t, func(c *AutoServerConfig) {
		c.Host = "127.0.0.1"
		c.Port = 0
	})

	assert.Empty(t, as.Address(), "no address before Start")

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- as.Start(ctx) }()

	require.Eventually(t, func() bool { return as.Address() != "" },
		5*time.Second, 20*time.Millisecond, "Start must publish the bound address")

	assert.NotContains(t, as.Address(), ":0", "the reported port must be the real one")

	cancel()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("Start did not return after its context was canceled")
	}
}

// LoadAgentsFromDirectory scanned nothing: it listed whatever was already
// registered and logged that count as though it had loaded them.
func TestAutoServer_LoadAgentsFromDirectory(t *testing.T) {
	dir := t.TempDir()

	config := `name: from-directory
version: "1.0"
agents:
  loaded-agent:
    id: loaded-agent
    name: Loaded Agent
    type: chat
    model: fake-model
    provider: fake
`
	require.NoError(t, os.WriteFile(filepath.Join(dir, "agents.yaml"), []byte(config), 0o600))

	as := newAutoServer(t, nil)
	require.NoError(t, as.LoadAgentsFromDirectory(dir))

	_, exists := as.Registry().GetDefinition("loaded-agent")
	assert.True(t, exists, "the agent defined in the directory must be registered")
}

func TestAutoServer_LoadAgentsFromDirectoryReportsProblems(t *testing.T) {
	as := newAutoServer(t, nil)

	// A directory that does not exist.
	require.Error(t, as.LoadAgentsFromDirectory(filepath.Join(t.TempDir(), "missing")))

	// A directory with nothing loadable must say so rather than report success.
	empty := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(empty, "notes.txt"), []byte("hi"), 0o600))
	err := as.LoadAgentsFromDirectory(empty)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "no agent configuration files")

	// A malformed config must fail loudly.
	bad := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(bad, "broken.yaml"), []byte("{{{"), 0o600))
	assert.Error(t, as.LoadAgentsFromDirectory(bad))
}
