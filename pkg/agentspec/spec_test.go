// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package agentspec_test

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/UnicoLab/GoLangGraph/pkg/agentspec"
)

type request struct {
	Text  string `json:"text"`
	Limit int    `json:"limit,omitempty"`
}

func (r request) Validate() error {
	if strings.TrimSpace(r.Text) == "" {
		return fmt.Errorf("text cannot be blank")
	}
	return nil
}

type response struct {
	Items []string `json:"items"`
}

func config() agentspec.AgentConfig {
	return agentspec.AgentConfig{Name: "sample", Description: "Sample agent", Prompt: agentspec.Prompt{
		System: "Answer with JSON", Model: agentspec.Model{Provider: "test", Name: "model", MaxTokens: 100},
	}}
}

func TestTypedContractValidatesActualWireValues(t *testing.T) {
	definition, err := agentspec.NewAgent[request, response](config())
	if err != nil {
		t.Fatal(err)
	}
	for _, input := range []string{`{}`, `null`, `{"text":null}`, `{"text":42}`, `{"text":" "}`, `{"text":"hi","extra":true}`, `{"text":"hi","limit":1.5}`, `{"text":"hi"} {}`} {
		t.Run(input, func(t *testing.T) {
			if err := definition.ValidateInput(json.RawMessage(input)); err == nil {
				t.Fatalf("accepted invalid input: %s", input)
			}
		})
	}
	if err := definition.ValidateInput(json.RawMessage(`{"text":"hello"}`)); err != nil {
		t.Fatal(err)
	}
	if err := definition.ValidateOutput(json.RawMessage(`{"items":["hello"]}`)); err != nil {
		t.Fatal(err)
	}
	for _, output := range []string{`{}`, `{"items":null}`, `{"items":[1]}`} {
		if err := definition.ValidateOutput(json.RawMessage(output)); err == nil {
			t.Fatalf("accepted invalid output: %s", output)
		}
	}
}

func TestManifestMutationCannotChangeDefinition(t *testing.T) {
	definition, err := agentspec.NewAgent[request, response](config())
	if err != nil {
		t.Fatal(err)
	}
	pkg, err := agentspec.NewPackage("example.com/agents/sample", "v0.1.0", "Example package", definition)
	if err != nil {
		t.Fatal(err)
	}
	manifest := pkg.Manifest()
	delete(manifest.Agents[0].InputSchema.Properties, "text")
	manifest.Agents[0].InputSchema.Required[0] = "corrupted"
	*manifest.Agents[0].InputSchema.AdditionalProperties = true
	manifest.Agents[0].Prompt.System = "corrupted"
	if err := definition.ValidateInput(json.RawMessage(`{"text":"hello"}`)); err != nil {
		t.Fatal(err)
	}
	fresh := pkg.Manifest().Agents[0]
	if fresh.InputSchema.Required[0] != "text" || *fresh.InputSchema.AdditionalProperties || fresh.Prompt.System != "Answer with JSON" {
		t.Fatalf("manifest leaked mutable definition state: %+v", fresh)
	}
	data, err := json.Marshal(pkg.Manifest())
	if err != nil || !strings.Contains(string(data), agentspec.ContractVersion) {
		t.Fatalf("manifest is not serializable: %s, %v", data, err)
	}
}

type fakeState struct{}

func (fakeState) ThreadID() string                        { return "thread" }
func (fakeState) GetValue(string) (json.RawMessage, bool) { return nil, false }
func (fakeState) SetValue(string, json.RawMessage) error  { return nil }

func TestToolValidatesBeforeAndAfterExecution(t *testing.T) {
	calls := 0
	tool, err := agentspec.NewTool("search", "Search documents", func(_ context.Context, _ agentspec.State, input request) (response, error) {
		calls++
		return response{}, nil // A nil slice cannot satisfy the advertised array schema.
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := tool.Execute(context.Background(), fakeState{}, json.RawMessage(`{}`)); err == nil || calls != 0 {
		t.Fatalf("invalid input reached tool: calls=%d, err=%v", calls, err)
	}
	if _, err := tool.Execute(context.Background(), fakeState{}, json.RawMessage(`{"text":"hello"}`)); err == nil || calls != 1 {
		t.Fatalf("invalid output accepted: calls=%d, err=%v", calls, err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := tool.Execute(ctx, fakeState{}, json.RawMessage(`{"text":"hello"}`)); err == nil || calls != 1 {
		t.Fatalf("canceled invocation reached tool: calls=%d, err=%v", calls, err)
	}
}

func TestUnsupportedSchemasAndInvalidDefinitionsFailEarly(t *testing.T) {
	type pointerInput struct {
		Text *string `json:"text"`
	}
	type binaryInput struct {
		Data []byte `json:"data"`
	}
	if _, err := agentspec.NewAgent[pointerInput, response](config()); err == nil {
		t.Fatal("accepted pointer schema")
	}
	if _, err := agentspec.NewAgent[binaryInput, response](config()); err == nil {
		t.Fatal("accepted ambiguous binary schema")
	}
	bad := config()
	bad.MaxModelCalls = -1
	if _, err := agentspec.NewAgent[request, response](bad); err == nil {
		t.Fatal("accepted invalid loop budget")
	}
	definition, err := agentspec.NewAgent[request, response](config())
	if err != nil {
		t.Fatal(err)
	}
	for _, version := range []string{"latest", "v1", "v01.0.0", "v1.0.0-01"} {
		if _, err := agentspec.NewPackage("example.com/a", version, "test", definition); err == nil {
			t.Fatalf("accepted invalid version %s", version)
		}
	}
	if _, err := agentspec.NewPackage("example.com/a", "v0.1.0", "test", definition, definition); err == nil {
		t.Fatal("accepted duplicate entry point")
	}
	if _, err := agentspec.NewPackage("example.com/a#b", "v0.1.0", "test", definition); err == nil {
		t.Fatal("accepted ambiguous package namespace")
	}
}
