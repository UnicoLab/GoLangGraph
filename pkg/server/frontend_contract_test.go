// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/UnicoLab/GoLangGraph/pkg/agent"
	"github.com/UnicoLab/GoLangGraph/pkg/llm"
	"github.com/UnicoLab/GoLangGraph/pkg/tools"
)

// TestFrontendAPIContract verifies the JSON contract consumed by the
// GoLangGraph Studio frontend (`/api/v1`): snake_case agent configs and
// PascalCase execution payloads (Go's default field names).
func TestFrontendAPIContract(t *testing.T) {
	llmManager := llm.NewProviderManager()
	llmManager.RegisterProvider("mock", &MockProvider{})

	toolRegistry := tools.NewToolRegistry()
	manager := NewAgentManager(llmManager, toolRegistry)

	cfg := &agent.AgentConfig{
		ID:              "test-agent-1",
		Name:            "test-chat-agent",
		Type:            agent.AgentTypeChat,
		Model:           "mock-model",
		Provider:        "mock",
		SystemPrompt:    "You are a test agent",
		Temperature:     0.7,
		MaxTokens:       1000,
		MaxIterations:   5,
		Tools:           []string{},
		EnableStreaming: false,
		Timeout:         30 * time.Second,
	}

	if _, err := manager.CreateAgent(cfg); err != nil {
		t.Fatalf("failed to create agent: %v", err)
	}

	ids := manager.ListAgents()
	if len(ids) == 0 {
		t.Fatal("expected at least one registered agent")
	}
	agentID := ids[0]

	server := NewServer(&ServerConfig{Host: "localhost", Port: 8080, EnableCORS: true})
	server.SetLLMManager(llmManager)
	server.SetToolRegistry(toolRegistry)
	server.SetAgentManager(manager)

	// 1. List agents -> {"agents": [{...snake_case...}]}
	listRaw := doRequest(t, server, "GET", "/api/v1/agents", "")
	var list struct {
		Agents []map[string]interface{} `json:"agents"`
	}
	if err := json.Unmarshal(listRaw, &list); err != nil {
		t.Fatalf("list agents response not valid JSON: %v", err)
	}
	if len(list.Agents) == 0 {
		t.Fatal("expected at least one agent in /api/v1/agents")
	}
	first := list.Agents[0]
	for _, key := range []string{"id", "name", "type", "model", "provider", "system_prompt", "temperature", "max_tokens", "max_iterations", "tools", "enable_streaming"} {
		if _, ok := first[key]; !ok {
			t.Errorf("agent config missing snake_case field %q", key)
		}
	}

	// 2. Get agent -> {"agent": {...}}
	getRaw := doRequest(t, server, "GET", "/api/v1/agents/"+agentID, "")
	var get struct {
		Agent map[string]interface{} `json:"agent"`
	}
	if err := json.Unmarshal(getRaw, &get); err != nil {
		t.Fatalf("get agent response not valid JSON: %v", err)
	}
	if get.Agent["name"] != "test-chat-agent" {
		t.Errorf("expected agent name, got %v", get.Agent["name"])
	}

	// 3. Execute agent -> {"execution": {PascalCase...}}
	execRaw := doRequest(t, server, "POST", "/api/v1/agents/"+agentID+"/execute", `{"input":"hello"}`)
	var exec struct {
		Execution map[string]interface{} `json:"execution"`
	}
	if err := json.Unmarshal(execRaw, &exec); err != nil {
		t.Fatalf("execute response not valid JSON: %v", err)
	}
	if exec.Execution == nil {
		t.Fatal("expected an execution object in execute response")
	}
	for _, key := range []string{"ID", "Input", "Output", "Success", "Status", "Duration", "Steps", "ToolCalls"} {
		if _, ok := exec.Execution[key]; !ok {
			t.Errorf("execution missing PascalCase field %q", key)
		}
	}
	if success, _ := exec.Execution["Success"].(bool); !success {
		t.Errorf("expected successful execution, got %v", exec.Execution["Success"])
	}
}

func doRequest(t *testing.T, server *Server, method, path, body string) []byte {
	t.Helper()
	req, err := http.NewRequest(method, path, strings.NewReader(body))
	if err != nil {
		t.Fatalf("failed to build request: %v", err)
	}
	if body != "" {
		req.Header.Set("Content-Type", "application/json")
	}
	rr := httptest.NewRecorder()
	server.router.ServeHTTP(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("%s %s returned status %d: %s", method, path, rr.Code, rr.Body.String())
	}
	return rr.Body.Bytes()
}
