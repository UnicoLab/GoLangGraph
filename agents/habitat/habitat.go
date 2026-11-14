// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

// Package habitat is a reusable composition of two independently versioned
// agents. It depends on their contracts and the SDK, not an execution engine.
package habitat

import (
	"context"

	"github.com/UnicoLab/GoLangGraph/agents/designer"
	"github.com/UnicoLab/GoLangGraph/agents/storymaker"
	"github.com/UnicoLab/GoLangGraph/pkg/agentspec"
)

const (
	Module  = "github.com/UnicoLab/GoLangGraph/agents/habitat"
	Version = "v0.1.0"
	AgentID = Module + "#create"
)

type Input = designer.Input

type Output struct {
	Design designer.Output   `json:"design"`
	Story  storymaker.Output `json:"story"`
}

// Package declares dependencies, but never installs or creates them itself.
// The host registers all three modules; each habitat thread owns two children.
func Package() (agentspec.Package, error) {
	design, err := agentspec.NewSubagent[designer.Input, designer.Output]("designer", "Design the habitat", designer.AgentID, designer.Version)
	if err != nil {
		return agentspec.Package{}, err
	}
	story, err := agentspec.NewSubagent[storymaker.Input, storymaker.Output]("storymaker", "Tell the habitat's story", storymaker.AgentID, storymaker.Version)
	if err != nil {
		return agentspec.Package{}, err
	}
	definition, err := agentspec.NewHandler[Input, Output](agentspec.HandlerConfig{
		Name: "create", Description: "Design a sustainable habitat and tell its story", Subagents: []agentspec.Subagent{design, story},
	}, func(ctx context.Context, execution agentspec.Execution, input Input) (Output, error) {
		design, err := agentspec.CallAgent[designer.Input, designer.Output](ctx, execution, "designer", input)
		if err != nil {
			return Output{}, err
		}
		story, err := agentspec.CallAgent[storymaker.Input, storymaker.Output](ctx, execution, "storymaker", storymaker.Input{
			StoryPrompt: "A resident's first morning in the habitat", DesignDescription: design.Description,
		})
		if err != nil {
			return Output{}, err
		}
		return Output{Design: design, Story: story}, nil
	})
	if err != nil {
		return agentspec.Package{}, err
	}
	return agentspec.NewPackage(Module, Version, "A composable habitat design and storytelling workflow", definition)
}
