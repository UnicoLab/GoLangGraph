// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package main

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/UnicoLab/GoLangGraph/agents/designer"
	"github.com/UnicoLab/GoLangGraph/agents/storymaker"
	"github.com/UnicoLab/GoLangGraph/pkg/llm"
)

// demoProvider replaces model inference only. It asks the runtime to execute the
// actual packaged tools, then incorporates their results into a typed answer.
type demoProvider struct{}

func (demoProvider) Complete(ctx context.Context, _ string, request llm.CompletionRequest) (*llm.CompletionResponse, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(request.Tools) != 1 || len(request.Messages) == 0 {
		return nil, fmt.Errorf("offline demo expects one packaged tool and conversation history")
	}
	var user string
	for _, message := range request.Messages {
		if message.Role == "user" {
			user = message.Content
		}
	}
	last := request.Messages[len(request.Messages)-1]
	name := request.Tools[0].Function.Name
	var args, output any
	switch name {
	case "material_palette":
		var input designer.Input
		if err := json.Unmarshal([]byte(user), &input); err != nil {
			return nil, err
		}
		args = designer.PaletteInput{Focus: input.Message}
		if last.Role == "tool" {
			var palette designer.Palette
			if err := json.Unmarshal([]byte(last.Content), &palette); err != nil {
				return nil, err
			}
			output = designer.Output{
				Description: "An imagined habitat for: " + input.Message + ". Shared outdoor space connects its residents to the landscape.",
				Materials:   palette.Materials, Style: palette.Style,
			}
		}
	case "story_outline":
		var input storymaker.Input
		if err := json.Unmarshal([]byte(user), &input); err != nil {
			return nil, err
		}
		args = storymaker.OutlineInput{Topic: input.StoryPrompt}
		if last.Role == "tool" {
			var outline storymaker.Outline
			if err := json.Unmarshal([]byte(last.Content), &outline); err != nil || len(outline.Beats) == 0 {
				return nil, fmt.Errorf("offline model received an invalid outline")
			}
			output = storymaker.Output{
				Title: "A Shared Morning", Story: input.DesignDescription + " A resident opens the door to meet the neighbors. " + outline.Beats[0] + ". Together they begin tending the garden.",
				Themes: []string{"community", "sustainability"}, Moral: "Thoughtful homes connect people to their surroundings and to each other.",
			}
		}
	default:
		return nil, fmt.Errorf("unsupported offline demo tool %q", name)
	}
	message := llm.Message{Role: "assistant"}
	if output == nil {
		data, err := json.Marshal(args)
		if err != nil {
			return nil, err
		}
		message.ToolCalls = []llm.ToolCall{{ID: fmt.Sprintf("demo-%d", len(request.Messages)), Type: "function",
			Function: llm.FunctionCall{Name: name, Arguments: string(data)}}}
	} else {
		data, err := json.Marshal(output)
		if err != nil {
			return nil, err
		}
		message.Content = string(data)
	}
	return &llm.CompletionResponse{Choices: []llm.Choice{{Message: message}}}, nil
}
