// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package designer_test

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/UnicoLab/GoLangGraph/agents/designer"
)

type state map[string]json.RawMessage

func (state) ThreadID() string                                   { return "test" }
func (s state) GetValue(key string) (json.RawMessage, bool)      { value, ok := s[key]; return value, ok }
func (s state) SetValue(key string, value json.RawMessage) error { s[key] = value; return nil }

func TestPackageCarriesPromptTypesAndWorkingTool(t *testing.T) {
	pkg, err := designer.Package()
	if err != nil {
		t.Fatal(err)
	}
	manifest := pkg.Manifest()
	if pkg.AgentID("design") != designer.AgentID || manifest.Version != designer.Version || !strings.Contains(manifest.Agents[0].Prompt.System, "sustainable architecture") {
		t.Fatalf("incomplete package: %+v", manifest)
	}
	definition := pkg.Definitions()[0]
	if err := definition.ValidateInput(json.RawMessage(`{"message":" "}`)); err == nil {
		t.Fatal("blank brief accepted")
	}
	if err := definition.ValidateOutput(json.RawMessage(`{"description":"A home","materials":["wood"],"style":"invalid"}`)); err == nil {
		t.Fatal("invalid style accepted")
	}
	values := state{}
	for i := 1; i <= 2; i++ {
		raw, err := definition.Tools()[0].Execute(context.Background(), values, json.RawMessage(`{"focus":"solar energy"}`))
		if err != nil {
			t.Fatal(err)
		}
		var palette designer.Palette
		if err := json.Unmarshal(raw, &palette); err != nil {
			t.Fatal(err)
		}
		if palette.Uses != i || palette.Style != "futuristic" || palette.Materials[0] != "photovoltaic panels" {
			t.Fatalf("unexpected palette: %+v", palette)
		}
	}
}
