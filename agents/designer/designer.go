// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

// Package designer packages the habitat designer from the ideation examples as
// a reusable Go agent. Applications supply the model connection and thread store.
package designer

import (
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/UnicoLab/GoLangGraph/pkg/agentspec"
)

const (
	Module  = "github.com/UnicoLab/GoLangGraph/agents/designer"
	Version = "v0.1.0"
	AgentID = Module + "#design"
)

//go:embed prompt.txt
var systemPrompt string

type Input struct {
	Message string `json:"message" description:"A short sustainable habitat brief" minLength:"1" maxLength:"4000"`
}

func (i Input) Validate() error {
	if strings.TrimSpace(i.Message) == "" || len(i.Message) > 4000 {
		return fmt.Errorf("message must contain 1 to 4000 bytes of text")
	}
	return nil
}

type Output struct {
	Description string   `json:"description" minLength:"1"`
	Materials   []string `json:"materials"`
	Style       string   `json:"style" enum:"modern,minimalist,futuristic,organic,industrial"`
}

func (o Output) Validate() error {
	if strings.TrimSpace(o.Description) == "" || len(o.Materials) == 0 {
		return fmt.Errorf("a description and at least one material are required")
	}
	switch o.Style {
	case "modern", "minimalist", "futuristic", "organic", "industrial":
		return nil
	default:
		return fmt.Errorf("unsupported design style %q", o.Style)
	}
}

type PaletteInput struct {
	Focus string `json:"focus" description:"The habitat's energy, water or nature focus" minLength:"1"`
}

type Palette struct {
	Materials []string `json:"materials"`
	Style     string   `json:"style"`
	Uses      int      `json:"uses"`
}

// Package returns a fresh immutable definition, including its prompt, typed
// schemas, default model profile and implementation of the palette tool.
func Package() (agentspec.Package, error) {
	tool, err := agentspec.NewTool("material_palette", "Choose materials for an energy, water or nature-focused habitat.", palette)
	if err != nil {
		return agentspec.Package{}, err
	}
	definition, err := agentspec.NewAgent[Input, Output](agentspec.AgentConfig{
		Name: "design", Description: "Design a sustainable habitat from a short brief.",
		Prompt: agentspec.Prompt{System: systemPrompt, Model: agentspec.Model{
			Provider: "ollama", Name: "gemma3:1b", Temperature: 0.7, MaxTokens: 1500,
		}},
		Tools: []agentspec.Tool{tool}, MaxModelCalls: 5,
	})
	if err != nil {
		return agentspec.Package{}, err
	}
	return agentspec.NewPackage(Module, Version, "Reusable sustainable habitat designer.", definition)
}

func palette(_ context.Context, state agentspec.State, input PaletteInput) (Palette, error) {
	uses := 0
	if raw, exists := state.GetValue("palette_uses"); exists {
		if err := json.Unmarshal(raw, &uses); err != nil {
			return Palette{}, fmt.Errorf("read palette usage: %w", err)
		}
	}
	uses++
	raw, err := json.Marshal(uses)
	if err != nil {
		return Palette{}, err
	}
	if err := state.SetValue("palette_uses", raw); err != nil {
		return Palette{}, err
	}
	result := Palette{Materials: []string{"reclaimed timber", "natural insulation", "living roof"}, Style: "organic", Uses: uses}
	focus := strings.ToLower(input.Focus)
	switch {
	case strings.Contains(focus, "energy"), strings.Contains(focus, "solar"):
		result.Materials = []string{"photovoltaic panels", "recycled steel", "high-performance insulation"}
		result.Style = "futuristic"
	case strings.Contains(focus, "water"), strings.Contains(focus, "rain"):
		result.Materials = []string{"rainwater cistern", "permeable paving", "reclaimed timber"}
		result.Style = "modern"
	}
	return result, nil
}
