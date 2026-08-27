// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/UnicoLab/GoLangGraph/pkg/agent"
	"github.com/UnicoLab/GoLangGraph/pkg/core"
	"github.com/UnicoLab/GoLangGraph/pkg/llm"
	"github.com/UnicoLab/GoLangGraph/pkg/tools"
)

func newPipelineTestServer(t *testing.T) *Server {
	t.Helper()
	providers := llm.NewProviderManager()
	if err := providers.RegisterProvider("mock", &MockProvider{}); err != nil {
		t.Fatalf("register provider: %v", err)
	}
	manager := NewAgentManager(providers, tools.NewToolRegistry())
	for _, id := range []string{"writer", "reviewer"} {
		cfg := agent.DefaultAgentConfig()
		cfg.ID = id
		cfg.Name = strings.Title(id)
		cfg.Provider = "mock"
		cfg.Model = "mock-model"
		if _, err := manager.CreateAgent(cfg); err != nil {
			t.Fatalf("create %s: %v", id, err)
		}
	}
	s := NewServer(&ServerConfig{Host: "localhost", Port: 8080})
	s.SetAgentManager(manager)
	s.SetGraphManager(NewGraphManager())
	return s
}

func TestStudioPipelineIsCreatedInspectableAndExecutable(t *testing.T) {
	s := newPipelineTestServer(t)
	body := `{"id":"draft-review","name":"Draft review","nodes":[{"id":"draft","agent_id":"writer"},{"id":"review","agent_id":"reviewer"}],"input_schema":{"query":{"type":"string","required":true}}}`
	request := httptest.NewRequest(http.MethodPost, "/api/v1/pipelines", strings.NewReader(body))
	request.Header.Set("Content-Type", "application/json")
	recorder := httptest.NewRecorder()
	s.router.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusCreated {
		t.Fatalf("create pipeline status %d: %s", recorder.Code, recorder.Body.String())
	}

	graph, found := s.GraphManager().Get("draft-review")
	if !found {
		t.Fatal("pipeline was not registered")
	}
	if topology := describeTopology(graph); len(topology.Nodes) != 2 || len(topology.Edges) != 1 {
		t.Fatalf("unexpected topology: %#v", topology)
	}
	if node := graph.Nodes["draft"]; node.Metadata["agent_id"] != "writer" {
		t.Fatalf("pipeline node metadata = %#v", node.Metadata)
	}
	missingContract := httptest.NewRecorder()
	s.router.ServeHTTP(missingContract, httptest.NewRequest(http.MethodPost, "/api/v1/graphs/draft-review/execute", strings.NewReader(`{"input":"hello"}`)))
	if missingContract.Code != http.StatusBadRequest {
		t.Fatalf("missing required runtime input status %d: %s", missingContract.Code, missingContract.Body.String())
	}
	validContract := httptest.NewRecorder()
	validContractRequest := httptest.NewRequest(http.MethodPost, "/api/v1/graphs/draft-review/execute", strings.NewReader(`{"input":"hello","state":{"query":"hello"}}`))
	validContractRequest.Header.Set("Content-Type", "application/json")
	s.router.ServeHTTP(validContract, validContractRequest)
	if validContract.Code != http.StatusOK {
		t.Fatalf("valid runtime input status %d: %s", validContract.Code, validContract.Body.String())
	}

	state, err := graph.Execute(context.Background(), buildInitialState("hello", nil))
	if err != nil {
		t.Fatalf("execute pipeline: %v", err)
	}
	if output, ok := state.Get("agent.review.output"); !ok || output == nil {
		t.Fatalf("review output was not preserved in state: %#v", state.GetAll())
	}
}

func TestStudioPipelineRejectsUnknownAgentsWithoutRegistering(t *testing.T) {
	s := newPipelineTestServer(t)
	request := httptest.NewRequest(http.MethodPost, "/api/v1/pipelines", strings.NewReader(`{"id":"bad","name":"Bad","nodes":[{"id":"missing","agent_id":"does-not-exist"}]}`))
	recorder := httptest.NewRecorder()
	s.router.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("status %d: %s", recorder.Code, recorder.Body.String())
	}
	if _, found := s.GraphManager().Get("bad"); found {
		t.Fatal("invalid pipeline was registered")
	}

	var response map[string]string
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode error response: %v", err)
	}
	if !strings.Contains(response["error"], "does-not-exist") {
		t.Fatalf("unexpected error response: %#v", response)
	}
}

func TestPipelineSchemaRejectsInvalidDefinitionsAndValues(t *testing.T) {
	schema := PipelineSchema{
		"query": {Type: "string", Required: true},
		"limit": {Type: "number"},
	}
	if err := schema.ValidateDefinition("input"); err != nil {
		t.Fatalf("valid definition: %v", err)
	}
	if err := schema.ValidateValues("input", map[string]core.StateValue{"query": "hello", "limit": float64(5)}); err != nil {
		t.Fatalf("valid values: %v", err)
	}
	if err := schema.ValidateValues("input", map[string]core.StateValue{"limit": 5}); err == nil {
		t.Fatal("required field should be rejected")
	}
	if err := schema.ValidateValues("input", map[string]core.StateValue{"query": 5}); err == nil {
		t.Fatal("wrong field type should be rejected")
	}
	if err := (PipelineSchema{"bad": {Type: "function"}}).ValidateDefinition("input"); err == nil {
		t.Fatal("unsupported schema type should be rejected")
	}
}
