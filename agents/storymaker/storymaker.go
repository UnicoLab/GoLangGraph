// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

// Package storymaker packages the storyteller from the ideation examples as a
// reusable Go agent. It has no dependency on the designer module or graph engine.
package storymaker

import (
	"context"
	_ "embed"
	"fmt"
	"strings"

	"github.com/UnicoLab/GoLangGraph/pkg/agentspec"
)

const (
	Module  = "github.com/UnicoLab/GoLangGraph/agents/storymaker"
	Version = "v0.1.0"
	AgentID = Module + "#story"
)

//go:embed prompt.txt
var systemPrompt string

type Input struct {
	StoryPrompt       string `json:"story_prompt"`
	DesignDescription string `json:"design_description"`
}

func (i Input) Validate() error {
	if strings.TrimSpace(i.StoryPrompt) == "" || strings.TrimSpace(i.DesignDescription) == "" {
		return fmt.Errorf("story_prompt and design_description must contain text")
	}
	return nil
}

type Output struct {
	Title  string   `json:"title"`
	Story  string   `json:"story"`
	Themes []string `json:"themes"`
	Moral  string   `json:"moral"`
}

func (o Output) Validate() error {
	if strings.TrimSpace(o.Title) == "" || strings.TrimSpace(o.Story) == "" || strings.TrimSpace(o.Moral) == "" || len(o.Themes) == 0 {
		return fmt.Errorf("title, story, themes and moral are required")
	}
	return nil
}

type OutlineInput struct {
	Topic string `json:"topic"`
}

type Outline struct {
	Beats []string `json:"beats"`
}

func Package() (agentspec.Package, error) {
	tool, err := agentspec.NewTool("story_outline", "Build a three-part narrative outline from a theme.",
		func(_ context.Context, _ agentspec.State, input OutlineInput) (Outline, error) {
			if strings.TrimSpace(input.Topic) == "" {
				return Outline{}, fmt.Errorf("topic must contain text")
			}
			return Outline{Beats: []string{
				"Introduce a resident discovering " + input.Topic,
				"Show a daily challenge resolved through thoughtful habitat design",
				"Connect the resident's experience to the wider community",
			}}, nil
		})
	if err != nil {
		return agentspec.Package{}, err
	}
	definition, err := agentspec.NewAgent[Input, Output](agentspec.AgentConfig{
		Name: "story", Description: "Turn a habitat concept into a human-centered narrative.",
		Prompt: agentspec.Prompt{System: systemPrompt, Model: agentspec.Model{
			Provider: "ollama", Name: "gemma3:1b", Temperature: 0.7, MaxTokens: 1800,
		}},
		Tools: []agentspec.Tool{tool}, MaxModelCalls: 5,
	})
	if err != nil {
		return agentspec.Package{}, err
	}
	return agentspec.NewPackage(Module, Version, "Reusable storyteller for sustainable habitat concepts.", definition)
}
