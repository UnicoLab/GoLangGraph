// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

package agent

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	yaml "gopkg.in/yaml.v3"

	"github.com/UnicoLab/GoLangGraph/pkg/llm"
	"github.com/UnicoLab/GoLangGraph/pkg/tools"
	"github.com/UnicoLab/GoLangGraph/test/fakes"
)

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

// newTestAgentConfig builds a minimal valid agent config bound to provider.
func newTestAgentConfig(id, provider string) *AgentConfig {
	return &AgentConfig{
		ID:            id,
		Name:          id,
		Type:          AgentTypeChat,
		Model:         "fake-model",
		Provider:      provider,
		SystemPrompt:  "you are a test agent",
		MaxIterations: 1,
		Tools:         []string{},
	}
}

// newManagerWithProviders builds a manager whose LLM manager already knows the
// supplied providers. It stops the manager when the test finishes so no test
// can leave health checkers running behind it.
func newManagerWithProviders(t *testing.T, config *MultiAgentConfig, providers map[string]*fakes.Provider) *MultiAgentManager {
	t.Helper()

	llmManager := llm.NewProviderManager()
	for name, provider := range providers {
		require.NoError(t, llmManager.RegisterProvider(name, provider))
	}

	manager, err := NewMultiAgentManager(config, llmManager, tools.NewToolRegistry())
	require.NoError(t, err)
	t.Cleanup(func() {
		stopCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_ = manager.Stop(stopCtx)
	})
	return manager
}

// newSingleAgentManager is the common one-agent, one-route setup.
func newSingleAgentManager(t *testing.T, mutate func(*MultiAgentConfig)) (*MultiAgentManager, *fakes.Provider) {
	t.Helper()

	provider := fakes.NewProvider("fake", "hello from the fake provider")
	config := &MultiAgentConfig{
		Name:    "single-agent",
		Version: "1.0",
		Agents:  map[string]*AgentConfig{"solo": newTestAgentConfig("solo", "fake")},
		Routing: &RoutingConfig{
			Type:  "path",
			Rules: []RoutingRule{{ID: "solo-rule", Pattern: "/solo", AgentID: "solo", Method: "POST"}},
		},
	}
	if mutate != nil {
		mutate(config)
	}

	return newManagerWithProviders(t, config, map[string]*fakes.Provider{"fake": provider}), provider
}

func postInput(t *testing.T, url, input string) (int, string) {
	t.Helper()
	body := `{"input":` + jsonString(input) + `}`
	resp, err := http.Post(url, "application/json", strings.NewReader(body))
	require.NoError(t, err)
	defer func() { _ = resp.Body.Close() }()
	payload, err := io.ReadAll(resp.Body)
	require.NoError(t, err)
	return resp.StatusCode, string(payload)
}

// jsonString quotes a value for embedding in a JSON request body.
func jsonString(s string) string {
	encoded, err := json.Marshal(s)
	if err != nil {
		return `""`
	}
	return string(encoded)
}

func getJSON(t *testing.T, url string) (int, map[string]interface{}) {
	t.Helper()
	resp, err := http.Get(url)
	require.NoError(t, err)
	defer func() { _ = resp.Body.Close() }()

	var decoded map[string]interface{}
	payload, err := io.ReadAll(resp.Body)
	require.NoError(t, err)
	if len(payload) > 0 {
		_ = json.Unmarshal(payload, &decoded)
	}
	return resp.StatusCode, decoded
}

// ---------------------------------------------------------------------------
// Config validation defects
// ---------------------------------------------------------------------------

// Defect: MultiAgentConfig.Validate dereferenced every *AgentConfig without a
// nil check, so a YAML agent key with no body ("agents:\n  ghost:") took the
// process down with a nil pointer panic instead of returning a config error.
func TestRegression_ValidateRejectsEmptyAgentInsteadOfPanicking(t *testing.T) {
	var config MultiAgentConfig
	require.NoError(t, yaml.Unmarshal([]byte("name: ghosts\nagents:\n  ghost:\n"), &config))
	require.Contains(t, config.Agents, "ghost")
	require.Nil(t, config.Agents["ghost"], "the YAML must actually decode to a nil agent config")

	err := config.Validate()

	require.Error(t, err)
	assert.Contains(t, err.Error(), "agent ghost: configuration is empty")
}

// Defect: the manager panicked in setupRouting on a config with no routing
// block, even though Validate explicitly treats routing as optional.
func TestRegression_ManagerBuildsWithoutRoutingSection(t *testing.T) {
	config := &MultiAgentConfig{
		Name:   "no-routing",
		Agents: map[string]*AgentConfig{"solo": newTestAgentConfig("solo", "fake")},
		// Routing deliberately nil.
	}

	manager := newManagerWithProviders(t, config, map[string]*fakes.Provider{
		"fake": fakes.NewProvider("fake", "hi"),
	})

	// Management endpoints must still be reachable.
	server := httptest.NewServer(manager.GetRouter())
	defer server.Close()

	status, body := getJSON(t, server.URL+"/routing")
	assert.Equal(t, http.StatusOK, status)
	assert.NotNil(t, body)
}

// Defect: validateRouting accepted rules with no pattern, which then installed
// a route that could never match, leaving the named agent unreachable.
func TestRegression_ValidateRejectsPatternlessRule(t *testing.T) {
	config := &MultiAgentConfig{
		Name:   "patternless",
		Agents: map[string]*AgentConfig{"solo": newTestAgentConfig("solo", "fake")},
		Routing: &RoutingConfig{
			Type:  "path",
			Rules: []RoutingRule{{ID: "broken", AgentID: "solo"}},
		},
	}

	err := config.Validate()

	require.Error(t, err)
	assert.Contains(t, err.Error(), "pattern is required")
}

// Defect: Validate accepted any routing type and setupRoutingRule silently fell
// through to path routing for the unknown ones.
func TestRegression_ValidateRejectsUnknownRoutingType(t *testing.T) {
	config := &MultiAgentConfig{
		Name:   "weird-routing",
		Agents: map[string]*AgentConfig{"solo": newTestAgentConfig("solo", "fake")},
		Routing: &RoutingConfig{
			Type:  "carrier-pigeon",
			Rules: []RoutingRule{{ID: "r", Pattern: "/solo", AgentID: "solo"}},
		},
	}

	err := config.Validate()

	require.Error(t, err)
	assert.Contains(t, err.Error(), "unsupported routing type")
}

// ---------------------------------------------------------------------------
// Routing defects
// ---------------------------------------------------------------------------

// Defect: matchesRule switched on rule.Pattern against the literal strings
// "prefix"/"suffix"/"exact"/"contains", so the match mode could never be
// chosen and every real pattern fell through to prefix matching - while the
// HTTP router matched the same rule exactly. GetAgentByPath also ignored
// Priority, so it disagreed with the router about which rule wins.
func TestRegression_GetAgentByPathHonoursPriorityAndMatchMode(t *testing.T) {
	config := &MultiAgentConfig{
		Name: "priority",
		Agents: map[string]*AgentConfig{
			"broad":     newTestAgentConfig("broad", "fake"),
			"specific":  newTestAgentConfig("specific", "fake"),
			"suffixing": newTestAgentConfig("suffixing", "fake"),
		},
		Routing: &RoutingConfig{
			Type: "path",
			Rules: []RoutingRule{
				{ID: "broad", Pattern: "/api", Match: MatchPrefix, AgentID: "broad", Priority: 1},
				{ID: "specific", Pattern: "/api/special", Match: MatchPrefix, AgentID: "specific", Priority: 100},
				{ID: "suffixing", Pattern: ".json", Match: MatchSuffix, AgentID: "suffixing", Priority: 50},
			},
		},
	}
	require.NoError(t, config.Validate())

	// Highest priority wins even though it is declared second.
	agentID, found := config.GetAgentByPath("/api/special/thing")
	assert.True(t, found)
	assert.Equal(t, "specific", agentID, "priority must decide, not declaration order")

	// The lower-priority prefix rule still serves everything else under /api.
	agentID, found = config.GetAgentByPath("/api/other")
	assert.True(t, found)
	assert.Equal(t, "broad", agentID)

	// Suffix mode is now reachable at all.
	agentID, found = config.GetAgentByPath("/reports/latest.json")
	assert.True(t, found)
	assert.Equal(t, "suffixing", agentID)

	// A pattern that happens to be spelled like a match mode is treated as a
	// pattern, which is what the old switch got wrong.
	odd := &MultiAgentConfig{
		Name:   "odd",
		Agents: map[string]*AgentConfig{"solo": newTestAgentConfig("solo", "fake")},
		Routing: &RoutingConfig{
			Type:  "path",
			Rules: []RoutingRule{{ID: "r", Pattern: "exact", Match: MatchExact, AgentID: "solo"}},
		},
	}
	_, found = odd.GetAgentByPath("nothing-like-it")
	assert.False(t, found, "an unrelated path must not match the literal pattern \"exact\"")
	agentID, found = odd.GetAgentByPath("exact")
	assert.True(t, found)
	assert.Equal(t, "solo", agentID)
}

// Defect: header patterns were split on every colon and dropped unless exactly
// two parts came back, so "Authorization: Bearer xyz" - a value containing a
// colon-separated scheme - installed no route at all, silently, and the agent
// was unreachable while startup reported success.
func TestRegression_HeaderRoutingRuleKeepsColonsInValue(t *testing.T) {
	provider := fakes.NewProvider("fake", "routed")
	config := &MultiAgentConfig{
		Name:   "header-routing",
		Agents: map[string]*AgentConfig{"solo": newTestAgentConfig("solo", "fake")},
		Routing: &RoutingConfig{
			Type:  "header",
			Rules: []RoutingRule{{ID: "bearer", Pattern: "Authorization: Bearer xyz", AgentID: "solo", Method: "POST"}},
		},
	}

	manager := newManagerWithProviders(t, config, map[string]*fakes.Provider{"fake": provider})
	server := httptest.NewServer(manager.GetRouter())
	defer server.Close()

	req, err := http.NewRequest(http.MethodPost, server.URL+"/anything", strings.NewReader(`{"input":"hi"}`))
	require.NoError(t, err)
	req.Header.Set("Authorization", "Bearer xyz")

	resp, err := http.DefaultClient.Do(req)
	require.NoError(t, err)
	defer func() { _ = resp.Body.Close() }()

	assert.Equal(t, http.StatusOK, resp.StatusCode, "the rule must actually be installed")
	assert.Equal(t, 1, provider.Calls())

	// A request without the header must not reach the agent.
	resp2, err := http.Post(server.URL+"/anything", "application/json", strings.NewReader(`{"input":"hi"}`))
	require.NoError(t, err)
	defer func() { _ = resp2.Body.Close() }()
	assert.Equal(t, http.StatusNotFound, resp2.StatusCode)
}

// Defect: a header/query pattern the router could not express left `route` nil
// and setupRoutingRule returned without a word, so a broken config started
// cleanly with a silently missing route. Failures are now reported.
func TestRegression_UninstallableRoutingRuleIsReported(t *testing.T) {
	config := &MultiAgentConfig{
		Name:   "bad-header-pattern",
		Agents: map[string]*AgentConfig{"solo": newTestAgentConfig("solo", "fake")},
		Routing: &RoutingConfig{
			Type:  "header",
			Rules: []RoutingRule{{ID: "no-colon", Pattern: "AuthorizationBearerXyz", AgentID: "solo"}},
		},
	}

	llmManager := llm.NewProviderManager()
	require.NoError(t, llmManager.RegisterProvider("fake", fakes.NewProvider("fake", "hi")))

	_, err := NewMultiAgentManager(config, llmManager, tools.NewToolRegistry())

	require.Error(t, err)
	assert.Contains(t, err.Error(), "no-colon")
}

// Defect: RoutingRule.Conditions were parsed, stored and serialised but never
// compared against a request, so a rule guarded by a condition matched
// everything.
func TestRegression_RoutingConditionsAreEvaluated(t *testing.T) {
	provider := fakes.NewProvider("fake", "conditional")
	config := &MultiAgentConfig{
		Name:   "conditions",
		Agents: map[string]*AgentConfig{"solo": newTestAgentConfig("solo", "fake")},
		Routing: &RoutingConfig{
			Type: "path",
			Rules: []RoutingRule{{
				ID:      "guarded",
				Pattern: "/guarded",
				AgentID: "solo",
				Method:  "POST",
				Conditions: []RoutingCondition{
					{Type: ConditionHeader, Key: "X-Tenant", Value: "acme", Operator: OperatorEquals},
				},
			}},
		},
	}

	manager := newManagerWithProviders(t, config, map[string]*fakes.Provider{"fake": provider})
	server := httptest.NewServer(manager.GetRouter())
	defer server.Close()

	// Condition satisfied.
	req, err := http.NewRequest(http.MethodPost, server.URL+"/guarded", strings.NewReader(`{"input":"hi"}`))
	require.NoError(t, err)
	req.Header.Set("X-Tenant", "acme")
	resp, err := http.DefaultClient.Do(req)
	require.NoError(t, err)
	defer func() { _ = resp.Body.Close() }()
	assert.Equal(t, http.StatusOK, resp.StatusCode)

	// Condition violated: the rule must not fire.
	req2, err := http.NewRequest(http.MethodPost, server.URL+"/guarded", strings.NewReader(`{"input":"hi"}`))
	require.NoError(t, err)
	req2.Header.Set("X-Tenant", "someone-else")
	resp2, err := http.DefaultClient.Do(req2)
	require.NoError(t, err)
	defer func() { _ = resp2.Body.Close() }()
	assert.Equal(t, http.StatusNotFound, resp2.StatusCode)

	assert.Equal(t, 1, provider.Calls(), "only the satisfying request may reach the agent")
}

// Defect: a "body" condition can never be evaluated by a router, and accepting
// it meant silently ignoring a guard the operator wrote down.
func TestRegression_UnsupportedConditionTypeIsRejected(t *testing.T) {
	config := &MultiAgentConfig{
		Name:   "body-condition",
		Agents: map[string]*AgentConfig{"solo": newTestAgentConfig("solo", "fake")},
		Routing: &RoutingConfig{
			Type: "path",
			Rules: []RoutingRule{{
				ID:         "r",
				Pattern:    "/solo",
				AgentID:    "solo",
				Conditions: []RoutingCondition{{Type: "body", Key: "kind", Value: "x"}},
			}},
		},
	}

	err := config.Validate()

	require.Error(t, err)
	assert.Contains(t, err.Error(), "unsupported condition type")
}

// ---------------------------------------------------------------------------
// Security defects
// ---------------------------------------------------------------------------

// Defect: GET /config encoded the raw configuration, so an unauthenticated
// caller got provider API keys, database and cache passwords and both secret
// maps back verbatim.
func TestRegression_ConfigEndpointRedactsSecrets(t *testing.T) {
	manager, _ := newSingleAgentManager(t, func(config *MultiAgentConfig) {
		config.Deployment = &DeploymentConfig{
			Type:     "docker",
			Replicas: 1,
			Secrets:  map[string]string{"tls": "DEPLOYMENT-SECRET"},
		}
		config.Shared = &SharedConfig{
			Secrets:      map[string]string{"token": "SHARED-SECRET"},
			LLMProviders: map[string]*LLMProviderConfig{"openai": {Type: "openai", APIKey: "sk-LEAKED-KEY"}},
			Database:     &DatabaseConfig{Host: "db", Password: "DB-PASSWORD"},
			Cache:        &CacheConfig{Host: "redis", Password: "CACHE-PASSWORD"},
			Security: &SecurityConfig{
				Authentication: &AuthConfig{Type: "apikey", Config: map[string]interface{}{"api_keys": []interface{}{"AUTH-KEY"}}},
			},
			Monitoring: &MonitoringConfig{Alerting: &AlertingConfig{
				Slack: &SlackConfig{WebhookURL: "https://hooks.example/SLACK-SECRET"},
				Email: &EmailConfig{SMTP: &SMTPConfig{Host: "smtp", Password: "SMTP-PASSWORD"}},
			}},
		}
	})

	server := httptest.NewServer(manager.GetRouter())
	defer server.Close()

	resp, err := http.Get(server.URL + "/config")
	require.NoError(t, err)
	defer func() { _ = resp.Body.Close() }()
	payload, err := io.ReadAll(resp.Body)
	require.NoError(t, err)
	body := string(payload)

	for _, secret := range []string{
		"DEPLOYMENT-SECRET", "SHARED-SECRET", "sk-LEAKED-KEY", "DB-PASSWORD", // pragma: allowlist secret
		"CACHE-PASSWORD", "AUTH-KEY", "SLACK-SECRET", "SMTP-PASSWORD",
	} {
		assert.NotContains(t, body, secret, "/config must not echo %s", secret)
	}
	assert.Contains(t, body, RedactedPlaceholder)

	// Redaction must not mutate the manager's live configuration.
	assert.Equal(t, "sk-LEAKED-KEY", manager.GetConfig().Shared.LLMProviders["openai"].APIKey)
	assert.Equal(t, "DB-PASSWORD", manager.GetConfig().Shared.Database.Password)
}

// Defect: the auth middleware read an API key and then accepted any non-empty
// value ("in a real implementation, validate the API key"), so a deployment
// that believed it required authentication was open to anyone who sent a
// header at all.
func TestRegression_AuthMiddlewareValidatesTheKey(t *testing.T) {
	provider := fakes.NewProvider("fake", "authorised")
	config := &MultiAgentConfig{
		Name:   "auth",
		Agents: map[string]*AgentConfig{"solo": newTestAgentConfig("solo", "fake")},
		Routing: &RoutingConfig{
			Type:  "path",
			Rules: []RoutingRule{{ID: "r", Pattern: "/solo", AgentID: "solo", Method: "POST"}},
			Middleware: []MiddlewareConfig{{
				Type:    "auth",
				Enabled: true,
				Config:  map[string]interface{}{"api_keys": []interface{}{"correct-horse"}},
			}},
		},
	}

	manager := newManagerWithProviders(t, config, map[string]*fakes.Provider{"fake": provider})
	server := httptest.NewServer(manager.GetRouter())
	defer server.Close()

	cases := []struct {
		name     string
		key      string
		expected int
	}{
		{"no key", "", http.StatusUnauthorized},
		{"wrong key", "battery-staple", http.StatusUnauthorized},
		{"correct key", "correct-horse", http.StatusOK},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req, err := http.NewRequest(http.MethodPost, server.URL+"/solo", strings.NewReader(`{"input":"hi"}`))
			require.NoError(t, err)
			if tc.key != "" {
				req.Header.Set("X-API-Key", tc.key)
			}
			resp, err := http.DefaultClient.Do(req)
			require.NoError(t, err)
			defer func() { _ = resp.Body.Close() }()
			assert.Equal(t, tc.expected, resp.StatusCode)
		})
	}

	assert.Equal(t, 1, provider.Calls(), "only the authorised request may reach the agent")
}

// Defect: enabling the auth middleware with no keys configured used to accept
// every request. Failing construction is the honest outcome - a config that
// asks for authentication it cannot perform must not start.
func TestRegression_AuthMiddlewareWithoutKeysFailsToStart(t *testing.T) {
	config := &MultiAgentConfig{
		Name:   "auth-no-keys",
		Agents: map[string]*AgentConfig{"solo": newTestAgentConfig("solo", "fake")},
		Routing: &RoutingConfig{
			Type:       "path",
			Rules:      []RoutingRule{{ID: "r", Pattern: "/solo", AgentID: "solo", Method: "POST"}},
			Middleware: []MiddlewareConfig{{Type: "auth", Enabled: true}},
		},
	}

	llmManager := llm.NewProviderManager()
	require.NoError(t, llmManager.RegisterProvider("fake", fakes.NewProvider("fake", "hi")))

	_, err := NewMultiAgentManager(config, llmManager, tools.NewToolRegistry())

	require.Error(t, err)
	assert.Contains(t, err.Error(), "no API keys are configured")
}

// Defect: rateLimitMiddleware called next.ServeHTTP and nothing else while the
// config carried a complete RateLimitConfig, so every configured limit was
// silently ignored.
func TestRegression_RateLimitMiddlewareEnforcesGlobalLimit(t *testing.T) {
	provider := fakes.NewProvider("fake", "limited")
	config := &MultiAgentConfig{
		Name:   "rate-limited",
		Agents: map[string]*AgentConfig{"solo": newTestAgentConfig("solo", "fake")},
		Routing: &RoutingConfig{
			Type:       "path",
			Rules:      []RoutingRule{{ID: "r", Pattern: "/solo", AgentID: "solo", Method: "POST"}},
			Middleware: []MiddlewareConfig{{Type: "rate_limit", Enabled: true}},
		},
		Shared: &SharedConfig{Security: &SecurityConfig{
			RateLimit: &RateLimitConfig{
				Enabled: true,
				Global:  &RateLimit{Requests: 3, Period: time.Hour, Burst: 3},
			},
		}},
	}

	manager := newManagerWithProviders(t, config, map[string]*fakes.Provider{"fake": provider})
	server := httptest.NewServer(manager.GetRouter())
	defer server.Close()

	allowed, limited := 0, 0
	for i := 0; i < 8; i++ {
		status, _ := postInput(t, server.URL+"/solo", "hi")
		switch status {
		case http.StatusOK:
			allowed++
		case http.StatusTooManyRequests:
			limited++
		default:
			t.Fatalf("unexpected status %d", status)
		}
	}

	assert.Equal(t, 3, allowed, "the configured budget of 3 requests must be honoured")
	assert.Equal(t, 5, limited)
	assert.Equal(t, 3, provider.Calls(), "rejected requests must not reach the agent")
}

// The per-agent budget in RateLimitConfig.PerAgent needs the agent's identity,
// which only the agent handler has; it was previously never consulted at all.
func TestRegression_RateLimitAppliesPerAgentBudget(t *testing.T) {
	busy := fakes.NewProvider("busy", "busy reply")
	calm := fakes.NewProvider("calm", "calm reply")

	config := &MultiAgentConfig{
		Name: "per-agent-rate-limit",
		Agents: map[string]*AgentConfig{
			"busy": newTestAgentConfig("busy", "busy"),
			"calm": newTestAgentConfig("calm", "calm"),
		},
		Routing: &RoutingConfig{
			Type: "path",
			Rules: []RoutingRule{
				{ID: "busy", Pattern: "/busy", AgentID: "busy", Method: "POST"},
				{ID: "calm", Pattern: "/calm", AgentID: "calm", Method: "POST"},
			},
			Middleware: []MiddlewareConfig{{Type: "rate_limit", Enabled: true}},
		},
		Shared: &SharedConfig{Security: &SecurityConfig{
			RateLimit: &RateLimitConfig{
				Enabled:  true,
				PerAgent: map[string]*RateLimit{"busy": {Requests: 2, Period: time.Hour, Burst: 2}},
			},
		}},
	}

	manager := newManagerWithProviders(t, config, map[string]*fakes.Provider{"busy": busy, "calm": calm})
	server := httptest.NewServer(manager.GetRouter())
	defer server.Close()

	busyLimited := 0
	for i := 0; i < 5; i++ {
		if status, _ := postInput(t, server.URL+"/busy", "hi"); status == http.StatusTooManyRequests {
			busyLimited++
		}
	}
	assert.Equal(t, 3, busyLimited, "the busy agent's budget of 2 must be enforced")

	// The agent without a budget is untouched.
	for i := 0; i < 5; i++ {
		status, _ := postInput(t, server.URL+"/calm", "hi")
		assert.Equal(t, http.StatusOK, status)
	}
	assert.Equal(t, 5, calm.Calls())
}

// Defect: an unknown middleware type was logged as a warning and dropped, so a
// typo in a config silently disabled the protection it named.
func TestRegression_UnknownMiddlewareTypeFailsToStart(t *testing.T) {
	config := &MultiAgentConfig{
		Name:   "typo-middleware",
		Agents: map[string]*AgentConfig{"solo": newTestAgentConfig("solo", "fake")},
		Routing: &RoutingConfig{
			Type:       "path",
			Rules:      []RoutingRule{{ID: "r", Pattern: "/solo", AgentID: "solo", Method: "POST"}},
			Middleware: []MiddlewareConfig{{Type: "authh", Enabled: true}},
		},
	}

	llmManager := llm.NewProviderManager()
	require.NoError(t, llmManager.RegisterProvider("fake", fakes.NewProvider("fake", "hi")))

	_, err := NewMultiAgentManager(config, llmManager, tools.NewToolRegistry())

	require.Error(t, err)
	assert.Contains(t, err.Error(), "unknown middleware type")
}

// Defect: corsMiddleware read mam.config.Shared.Security.CORS.Enabled with no
// nil check on any level of that chain, so a config that enabled the CORS
// middleware without a shared security section panicked inside the HTTP
// handler - after the server was already accepting traffic.
func TestRegression_CORSMiddlewareSurvivesMissingSharedSection(t *testing.T) {
	provider := fakes.NewProvider("fake", "cors ok")
	config := &MultiAgentConfig{
		Name:   "cors-no-shared",
		Agents: map[string]*AgentConfig{"solo": newTestAgentConfig("solo", "fake")},
		Routing: &RoutingConfig{
			Type:       "path",
			Rules:      []RoutingRule{{ID: "r", Pattern: "/solo", AgentID: "solo", Method: "POST"}},
			Middleware: []MiddlewareConfig{{Type: "cors", Enabled: true}},
		},
		// Shared deliberately nil.
	}

	manager := newManagerWithProviders(t, config, map[string]*fakes.Provider{"fake": provider})
	server := httptest.NewServer(manager.GetRouter())
	defer server.Close()

	status, _ := postInput(t, server.URL+"/solo", "hi")
	assert.Equal(t, http.StatusOK, status)
}

// With CORS actually configured the headers must be emitted.
func TestCORSMiddlewareEmitsConfiguredHeaders(t *testing.T) {
	manager, _ := newSingleAgentManager(t, func(config *MultiAgentConfig) {
		config.Routing.Middleware = []MiddlewareConfig{{Type: "cors", Enabled: true}}
		config.Shared = &SharedConfig{Security: &SecurityConfig{CORS: &CORSConfig{
			Enabled:        true,
			AllowedOrigins: []string{"https://allowed.example"},
			AllowedMethods: []string{"GET", "POST"},
			MaxAge:         600,
		}}}
	})

	server := httptest.NewServer(manager.GetRouter())
	defer server.Close()

	req, err := http.NewRequest(http.MethodPost, server.URL+"/solo", strings.NewReader(`{"input":"hi"}`))
	require.NoError(t, err)
	req.Header.Set("Origin", "https://allowed.example")
	resp, err := http.DefaultClient.Do(req)
	require.NoError(t, err)
	defer func() { _ = resp.Body.Close() }()

	assert.Equal(t, "https://allowed.example", resp.Header.Get("Access-Control-Allow-Origin"))
	assert.Equal(t, "600", resp.Header.Get("Access-Control-Max-Age"))

	// A disallowed origin is not echoed back.
	req2, err := http.NewRequest(http.MethodPost, server.URL+"/solo", strings.NewReader(`{"input":"hi"}`))
	require.NoError(t, err)
	req2.Header.Set("Origin", "https://evil.example")
	resp2, err := http.DefaultClient.Do(req2)
	require.NoError(t, err)
	defer func() { _ = resp2.Body.Close() }()
	assert.Empty(t, resp2.Header.Get("Access-Control-Allow-Origin"))
}

// Defect: the POST handler decoded whatever the client sent, with no ceiling on
// the body size.
func TestRegression_RequestBodyIsBounded(t *testing.T) {
	manager, provider := newSingleAgentManager(t, nil)
	server := httptest.NewServer(manager.GetRouter())
	defer server.Close()

	huge := `{"input":"` + strings.Repeat("A", MaxRequestBodyBytes+1024) + `"}`
	resp, err := http.Post(server.URL+"/solo", "application/json", strings.NewReader(huge))
	require.NoError(t, err)
	defer func() { _ = resp.Body.Close() }()

	assert.Equal(t, http.StatusBadRequest, resp.StatusCode)
	assert.Equal(t, 0, provider.Calls(), "an over-sized body must never reach the agent")
}

// ---------------------------------------------------------------------------
// Observability defects
// ---------------------------------------------------------------------------

// Defect: /health hard-coded "status": "healthy" and HTTP 200 while listing
// agents whose health said otherwise, so a liveness probe pointed at it always
// passed.
func TestRegression_HealthEndpointReportsRealStatus(t *testing.T) {
	manager, _ := newSingleAgentManager(t, nil)
	server := httptest.NewServer(manager.GetRouter())
	defer server.Close()

	manager.updateAgentHealthStatus("solo", "healthy")
	status, body := getJSON(t, server.URL+"/health")
	assert.Equal(t, http.StatusOK, status)
	assert.Equal(t, "healthy", body["status"])

	manager.updateAgentHealthStatus("solo", "unhealthy")
	status, body = getJSON(t, server.URL+"/health")
	assert.Equal(t, http.StatusServiceUnavailable, status, "an unhealthy agent must not answer 200")
	assert.Equal(t, "unhealthy", body["status"])

	status, body = getJSON(t, server.URL+"/health/solo")
	assert.Equal(t, http.StatusServiceUnavailable, status)
	assert.Equal(t, "unhealthy", body["status"])
}

// Defect: the health check asked agent.IsRunning(), which reports whether the
// agent is *mid-execution*. An idle, perfectly healthy agent was therefore
// marked unhealthy on every single tick.
func TestRegression_IdleAgentIsHealthy(t *testing.T) {
	provider := fakes.NewProvider("fake", "hi")
	config := &MultiAgentConfig{
		Name:   "health-checks",
		Agents: map[string]*AgentConfig{"solo": newTestAgentConfig("solo", "fake")},
		Routing: &RoutingConfig{
			Type:  "path",
			Rules: []RoutingRule{{ID: "r", Pattern: "/solo", AgentID: "solo", Method: "POST"}},
		},
		Deployment: &DeploymentConfig{
			Type: "docker", Replicas: 1,
			HealthCheck: &HealthCheckConfig{Enabled: true, PeriodSeconds: 1, FailureThreshold: 2},
		},
	}

	manager := newManagerWithProviders(t, config, map[string]*fakes.Provider{"fake": provider})

	// The agent is idle - which used to be read as "not running" and therefore
	// "unhealthy".
	require.False(t, manager.agents["solo"].IsRunning())

	results := manager.CheckHealthNow(context.Background())
	require.Contains(t, results, "solo")
	assert.Equal(t, "healthy", results["solo"].Status)
	assert.Equal(t, 0, results["solo"].ConsecutiveFails)

	state := manager.GetDeploymentState()
	assert.Equal(t, "healthy", state.AgentStates["solo"].HealthStatus)

	// And an actually broken provider is detected.
	provider.SetHealthy(false)
	results = manager.CheckHealthNow(context.Background())
	assert.Equal(t, "unhealthy", results["solo"].Status)
	assert.Equal(t, 1, results["solo"].ConsecutiveFails)
	assert.Contains(t, results["solo"].LastError, "unhealthy")

	// Below the failure threshold the agent is degraded, not yet unhealthy.
	assert.Equal(t, "degraded", manager.GetDeploymentState().AgentStates["solo"].HealthStatus)

	results = manager.CheckHealthNow(context.Background())
	assert.Equal(t, 2, results["solo"].ConsecutiveFails)
	assert.Equal(t, "unhealthy", manager.GetDeploymentState().AgentStates["solo"].HealthStatus)
}

// Defect: recordMetrics did all of its work inside "if the agent has a metrics
// entry", so a request routed to a missing agent - the 404 path, which is
// exactly a failure worth counting - incremented nothing. FailedRoutes was
// declared, serialised and never written at all.
func TestRegression_FailuresForUnknownAgentsAreCounted(t *testing.T) {
	manager, _ := newSingleAgentManager(t, nil)

	before := manager.GetMetrics()
	assert.Equal(t, int64(0), before.RoutingMetrics.FailedRoutes)

	handler := manager.createAgentHandler("does-not-exist", false)
	recorder := httptest.NewRecorder()
	handler(recorder, httptest.NewRequest(http.MethodPost, "/does-not-exist", strings.NewReader(`{"input":"hi"}`)))

	assert.Equal(t, http.StatusNotFound, recorder.Code)

	after := manager.GetMetrics()
	assert.Equal(t, before.TotalErrors+1, after.TotalErrors, "a 404 for a missing agent is an error")
	assert.Equal(t, int64(1), after.RoutingMetrics.FailedRoutes)
	require.Contains(t, after.AgentMetrics, "does-not-exist")
	assert.Equal(t, int64(1), after.AgentMetrics["does-not-exist"].ErrorCount)
}

// Defect: DeploymentState.ErrorCount and LastError were declared, serialised to
// /deployment/status and never written, so the deployment always looked clean
// no matter how many executions failed.
func TestRegression_DeploymentLevelErrorsAreRecorded(t *testing.T) {
	failing := fakes.NewProvider("failing", "").FailWith(errors.New("provider exploded"))
	config := &MultiAgentConfig{
		Name:   "deployment-errors",
		Agents: map[string]*AgentConfig{"solo": newTestAgentConfig("solo", "failing")},
		Routing: &RoutingConfig{
			Type:  "path",
			Rules: []RoutingRule{{ID: "r", Pattern: "/solo", AgentID: "solo", Method: "POST"}},
		},
	}

	manager := newManagerWithProviders(t, config, map[string]*fakes.Provider{"failing": failing})
	server := httptest.NewServer(manager.GetRouter())
	defer server.Close()

	status, _ := postInput(t, server.URL+"/solo", "hi")
	assert.Equal(t, http.StatusInternalServerError, status)

	state := manager.GetDeploymentState()
	assert.Equal(t, 1, state.ErrorCount, "the deployment must count the failure")
	assert.Contains(t, state.LastError, "provider exploded")
	assert.Equal(t, int64(1), state.AgentStates["solo"].ErrorCount)
	assert.Contains(t, state.AgentStates["solo"].LastError, "provider exploded")
}

// Defect: GetDeploymentState dereferenced the struct and returned the copy,
// which still shared the AgentStates map and every *AgentState in it - so the
// "snapshot" kept changing under the caller and reading it raced with request
// handlers.
func TestRegression_GetDeploymentStateIsADeepCopy(t *testing.T) {
	manager, _ := newSingleAgentManager(t, nil)

	snapshot := manager.GetDeploymentState()
	require.Contains(t, snapshot.AgentStates, "solo")
	assert.Equal(t, int64(0), snapshot.AgentStates["solo"].RequestCount)

	manager.updateAgentSuccess("solo")
	manager.updateAgentError("solo", errors.New("boom"))

	assert.Equal(t, int64(0), snapshot.AgentStates["solo"].RequestCount, "the snapshot must not change")
	assert.Equal(t, int64(0), snapshot.AgentStates["solo"].ErrorCount)
	assert.Equal(t, 0, snapshot.ErrorCount)

	// Mutating the snapshot must not reach the manager either.
	snapshot.AgentStates["solo"].Status = "tampered"
	delete(snapshot.AgentStates, "solo")
	fresh := manager.GetDeploymentState()
	require.Contains(t, fresh.AgentStates, "solo")
	assert.NotEqual(t, "tampered", fresh.AgentStates["solo"].Status)
	assert.Equal(t, int64(1), fresh.AgentStates["solo"].RequestCount)
}

// Defect: POST /deployment/restart logged "Restart requested", answered
// "restart_initiated" and did nothing at all.
func TestRegression_RestartActuallyRestartsAgents(t *testing.T) {
	manager, _ := newSingleAgentManager(t, nil)
	require.NoError(t, manager.Start(context.Background()))

	original := manager.agents["solo"]
	manager.updateAgentError("solo", errors.New("earlier failure"))
	require.Equal(t, 1, manager.GetDeploymentState().ErrorCount)

	server := httptest.NewServer(manager.GetRouter())
	defer server.Close()

	resp, err := http.Post(server.URL+"/deployment/restart", "application/json", nil)
	require.NoError(t, err)
	defer func() { _ = resp.Body.Close() }()
	assert.Equal(t, http.StatusOK, resp.StatusCode)

	var body map[string]interface{}
	require.NoError(t, json.NewDecoder(resp.Body).Decode(&body))
	assert.Equal(t, "restarted", body["status"])

	state := manager.GetDeploymentState()
	assert.Equal(t, "running", state.Status)
	assert.Equal(t, 0, state.ErrorCount, "restart must clear the deployment error state")
	assert.Empty(t, state.LastError)
	assert.Equal(t, int64(0), state.AgentStates["solo"].ErrorCount)
	assert.NotSame(t, original, manager.agents["solo"], "the agent must actually be rebuilt")

	// The manager is still usable afterwards.
	status, _ := postInput(t, server.URL+"/solo", "hi")
	assert.Equal(t, http.StatusOK, status)
}

// ---------------------------------------------------------------------------
// Execution defects
// ---------------------------------------------------------------------------

// Defect: the handler hard-coded a five minute deadline and never looked at
// AgentConfig.Timeout, so a config promising a short budget still let a slow
// provider hold the request open for minutes.
func TestRegression_AgentTimeoutComesFromConfig(t *testing.T) {
	slow := fakes.NewProvider("slow", "eventually").WithDelay(5 * time.Second)
	config := &MultiAgentConfig{
		Name: "timeout",
		Agents: map[string]*AgentConfig{
			"solo": func() *AgentConfig {
				c := newTestAgentConfig("solo", "slow")
				c.Timeout = 150 * time.Millisecond
				return c
			}(),
		},
		Routing: &RoutingConfig{
			Type:  "path",
			Rules: []RoutingRule{{ID: "r", Pattern: "/solo", AgentID: "solo", Method: "POST"}},
		},
	}

	manager := newManagerWithProviders(t, config, map[string]*fakes.Provider{"slow": slow})
	server := httptest.NewServer(manager.GetRouter())
	defer server.Close()

	start := time.Now()
	status, _ := postInput(t, server.URL+"/solo", "hi")
	elapsed := time.Since(start)

	assert.Equal(t, http.StatusGatewayTimeout, status, "a timeout must not be reported as a generic 500")
	assert.Less(t, elapsed, 2*time.Second, "the configured 150ms budget must be applied, took %v", elapsed)
}

// Defect: GetEnabledAgents returned every agent with a comment admitting it did
// no filtering, so a config that switched an agent off still got it created,
// health checked and routed to.
func TestRegression_DisabledAgentsAreNotCreated(t *testing.T) {
	config := &MultiAgentConfig{
		Name: "disabled-agents",
		Agents: map[string]*AgentConfig{
			"live": newTestAgentConfig("live", "fake"),
			"off": func() *AgentConfig {
				c := newTestAgentConfig("off", "fake")
				c.Metadata = map[string]interface{}{"disabled": true}
				return c
			}(),
			"off-string": func() *AgentConfig {
				c := newTestAgentConfig("off-string", "fake")
				c.Metadata = map[string]interface{}{"enabled": "false"}
				return c
			}(),
		},
		Routing: &RoutingConfig{
			Type:  "path",
			Rules: []RoutingRule{{ID: "r", Pattern: "/live", AgentID: "live", Method: "POST"}},
		},
		Deployment: &DeploymentConfig{
			Type: "docker", Replicas: 1,
			HealthCheck: &HealthCheckConfig{Enabled: true, PeriodSeconds: 1},
		},
	}

	assert.True(t, config.IsAgentEnabled("live"))
	assert.False(t, config.IsAgentEnabled("off"))
	assert.False(t, config.IsAgentEnabled("off-string"))
	assert.Len(t, config.GetEnabledAgents(), 1)

	manager := newManagerWithProviders(t, config, map[string]*fakes.Provider{"fake": fakes.NewProvider("fake", "hi")})

	state := manager.GetDeploymentState()
	assert.Contains(t, state.AgentStates, "live")
	assert.NotContains(t, state.AgentStates, "off")
	assert.NotContains(t, state.AgentStates, "off-string")

	_, hasChecker := manager.HealthCheckerStatus("off")
	assert.False(t, hasChecker, "a disabled agent must not be health checked")
	_, hasChecker = manager.HealthCheckerStatus("live")
	assert.True(t, hasChecker)
}

// A routing rule pointing at a disabled agent is a config error rather than a
// route that silently 404s at runtime.
func TestRegression_RoutingToDisabledAgentIsRejected(t *testing.T) {
	config := &MultiAgentConfig{
		Name: "route-to-disabled",
		Agents: map[string]*AgentConfig{
			"off": func() *AgentConfig {
				c := newTestAgentConfig("off", "fake")
				c.Metadata = map[string]interface{}{"disabled": true}
				return c
			}(),
		},
		Routing: &RoutingConfig{
			Type:  "path",
			Rules: []RoutingRule{{ID: "r", Pattern: "/off", AgentID: "off"}},
		},
	}

	err := config.Validate()

	require.Error(t, err)
	assert.Contains(t, err.Error(), "is disabled")
}
