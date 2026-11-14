// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package habitat

import (
	"encoding/json"
	"testing"

	"github.com/UnicoLab/GoLangGraph/agents/designer"
	"github.com/UnicoLab/GoLangGraph/agents/storymaker"
	"github.com/UnicoLab/GoLangGraph/pkg/agentspec"
)

func TestCompositionDeclaresExactDependencyContracts(t *testing.T) {
	pkg, err := Package()
	if err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(pkg.Manifest())
	if err != nil {
		t.Fatal(err)
	}
	manifest, err := agentspec.ParseManifest(data)
	if err != nil {
		t.Fatal(err)
	}
	agent := manifest.Agents[0]
	if agent.Kind != "handler" || agent.Prompt != nil || len(agent.Subagents) != 2 {
		t.Fatalf("incorrect composition: %+v", agent)
	}
	if agent.Subagents[0].AgentID != designer.AgentID || agent.Subagents[1].AgentID != storymaker.AgentID {
		t.Fatal("dependency identity changed")
	}
	if err := pkg.Definitions()[0].ValidateInput(json.RawMessage(`{"message":""}`)); err == nil {
		t.Fatal("designer input validation lost through alias")
	}
}
