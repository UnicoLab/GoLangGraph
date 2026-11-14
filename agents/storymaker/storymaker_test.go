// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package storymaker_test

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/UnicoLab/GoLangGraph/agents/storymaker"
)

type state struct{}

func (state) ThreadID() string                        { return "test" }
func (state) GetValue(string) (json.RawMessage, bool) { return nil, false }
func (state) SetValue(string, json.RawMessage) error  { return nil }

func TestPackageCarriesPromptTypesAndWorkingTool(t *testing.T) {
	pkg, err := storymaker.Package()
	if err != nil {
		t.Fatal(err)
	}
	if pkg.AgentID("story") != storymaker.AgentID || !strings.Contains(pkg.Manifest().Agents[0].Prompt.System, "sustainable habitats") {
		t.Fatal("missing package assets")
	}
	definition := pkg.Definitions()[0]
	if err := definition.ValidateInput(json.RawMessage(`{"story_prompt":"hello"}`)); err == nil {
		t.Fatal("missing design accepted")
	}
	tool := definition.Tools()[0]
	if _, err := tool.Execute(context.Background(), state{}, json.RawMessage(`{"topic":" "}`)); err == nil {
		t.Fatal("blank topic accepted")
	}
	raw, err := tool.Execute(context.Background(), state{}, json.RawMessage(`{"topic":"a shared garden"}`))
	if err != nil {
		t.Fatal(err)
	}
	var outline storymaker.Outline
	if err := json.Unmarshal(raw, &outline); err != nil {
		t.Fatal(err)
	}
	if len(outline.Beats) != 3 || !strings.Contains(outline.Beats[0], "a shared garden") {
		t.Fatalf("invalid outline: %+v", outline)
	}
}
