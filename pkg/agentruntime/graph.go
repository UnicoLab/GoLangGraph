// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package agentruntime

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/UnicoLab/GoLangGraph/pkg/agentspec"
	"github.com/UnicoLab/GoLangGraph/pkg/core"
	"github.com/UnicoLab/GoLangGraph/pkg/llm"
)

func (r *Runtime) execute(ctx context.Context, definition agentspec.Definition, thread *Thread, input json.RawMessage, scope *execution) (*Result, error) {
	manifest := definition.Manifest()
	result := scope.result
	graph := core.NewGraph(thread.AgentID)
	defer graph.Close()
	graph.Config.Timeout = 0
	if manifest.Kind == "handler" {
		graph.AddNode("handler", "Execute Go handler", func(ctx context.Context, state *core.BaseState) (*core.BaseState, error) {
			output, err := definition.Handle(ctx, scope, input)
			if err != nil {
				return nil, err
			}
			result.Output = output
			thread.Messages = append(thread.Messages, llm.Message{Role: "assistant", Content: string(output)})
			return state, nil
		})
		if err := graph.SetStartNode("handler"); err != nil {
			return nil, err
		}
		if err := graph.AddEndNode("handler"); err != nil {
			return nil, err
		}
		_, err := graph.Execute(ctx, core.NewBaseState())
		return result, err
	}
	toolDefinitions := make([]llm.ToolDefinition, 0, len(manifest.Tools)+len(manifest.Subagents))
	capabilities := append([]agentspec.ToolManifest(nil), manifest.Tools...)
	for _, child := range manifest.Subagents {
		capabilities = append(capabilities, agentspec.ToolManifest{Name: child.Name, Description: child.Description, InputSchema: child.InputSchema, OutputSchema: child.OutputSchema})
	}
	for _, meta := range capabilities {
		data, err := json.Marshal(meta.InputSchema)
		if err != nil {
			return nil, err
		}
		var parameters map[string]any
		if err := json.Unmarshal(data, &parameters); err != nil {
			return nil, err
		}
		toolDefinitions = append(toolDefinitions, llm.ToolDefinition{
			Type: "function", Function: llm.Function{Name: meta.Name, Description: meta.Description, Parameters: parameters},
		})
	}
	schema, err := json.Marshal(manifest.OutputSchema)
	if err != nil {
		return nil, err
	}
	system := manifest.Prompt.System + "\n\nReturn your final answer as one JSON object matching this schema, without markdown fences:\n" + string(schema)
	var pending []llm.ToolCall
	graph.Config.MaxIterations = 2*manifest.MaxModelCalls + 2
	// Caller cancellation/deadlines govern execution; the model-call budget
	// bounds loops independently of wall-clock time.
	graph.Config.Timeout = 0
	graph.AddNode("model", "Generate response", func(ctx context.Context, state *core.BaseState) (*core.BaseState, error) {
		if result.ModelCalls >= manifest.MaxModelCalls {
			return nil, ErrModelCallLimit
		}
		messages := make([]llm.Message, 0, len(thread.Messages)+1)
		messages = append(messages, llm.Message{Role: "system", Content: system})
		messages = append(messages, thread.Messages...)
		result.ModelCalls++
		scope.event("model_started", thread.Model.Name, "", "")
		response, err := r.providers.Complete(ctx, thread.Model.Provider, llm.CompletionRequest{
			Model: thread.Model.Name, Temperature: thread.Model.Temperature, MaxTokens: thread.Model.MaxTokens,
			Messages: messages, Tools: toolDefinitions,
		})
		if err != nil {
			scope.event("model_failed", thread.Model.Name, "", "")
			return nil, err
		}
		scope.event("model_returned", thread.Model.Name, "", "")
		if response == nil || len(response.Choices) != 1 {
			return nil, fmt.Errorf("provider must return exactly one completion choice")
		}
		choice := response.Choices[0]
		if choice.FinishReason == "length" || choice.FinishReason == "content_filter" {
			return nil, fmt.Errorf("incomplete model response: %s", choice.FinishReason)
		}
		message := choice.Message
		message.Role = "assistant"
		pending = message.ToolCalls
		if len(pending) > 256 {
			return nil, ErrInvocationLimit
		}
		seen := make(map[string]bool)
		for _, call := range pending {
			if call.ID == "" || seen[call.ID] || (call.Type != "" && call.Type != "function") {
				return nil, fmt.Errorf("invalid or duplicate tool call ID/type")
			}
			seen[call.ID] = true
		}
		thread.Messages = append(thread.Messages, message)
		result.Usage.PromptTokens += response.Usage.PromptTokens
		result.Usage.CompletionTokens += response.Usage.CompletionTokens
		result.Usage.TotalTokens += response.Usage.TotalTokens
		if len(pending) == 0 {
			if err := definition.ValidateOutput(json.RawMessage(message.Content)); err != nil {
				return nil, fmt.Errorf("agent output: %w", err)
			}
			result.Output = json.RawMessage(message.Content)
		}
		return state, nil
	})
	graph.AddNode("tools", "Execute package tools", func(ctx context.Context, state *core.BaseState) (*core.BaseState, error) {
		for _, call := range pending {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			output, toolErr := scope.invoke(ctx, call.Function.Name, json.RawMessage(call.Function.Arguments))
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			if errors.Is(toolErr, context.Canceled) || errors.Is(toolErr, context.DeadlineExceeded) || errors.Is(toolErr, ErrRunIndeterminate) || errors.Is(toolErr, ErrInvocationLimit) {
				return nil, toolErr
			}
			if toolErr != nil {
				// Tool errors are observations, allowing the model to correct
				// arguments on its next step within the configured budget.
				output, _ = json.Marshal(struct {
					Error string `json:"error"`
				}{toolErr.Error()})
			}
			thread.Messages = append(thread.Messages, llm.Message{
				Role: "tool", Name: call.Function.Name, ToolCallID: call.ID, Content: string(output),
			})
		}
		return state, nil
	})
	graph.AddNode("done", "Complete turn", func(_ context.Context, state *core.BaseState) (*core.BaseState, error) {
		return state, nil
	})
	if err := graph.AddConditionalEdges("model", func(_ context.Context, _ *core.BaseState) (string, error) {
		if len(pending) > 0 {
			return "tools", nil
		}
		return "done", nil
	}, map[string]string{"tools": "tools", "done": "done"}); err != nil {
		return nil, err
	}
	graph.AddEdge("tools", "model", nil)
	if err := graph.SetStartNode("model"); err != nil {
		return nil, err
	}
	if err := graph.AddEndNode("done"); err != nil {
		return nil, err
	}
	if _, err := graph.Execute(ctx, core.NewBaseState()); err != nil {
		return nil, err
	}
	return result, nil
}

type toolState struct {
	id     string
	values map[string]json.RawMessage
}

func (s *toolState) ThreadID() string { return s.id }

func (s *toolState) GetValue(key string) (json.RawMessage, bool) {
	value, exists := s.values[key]
	return append(json.RawMessage(nil), value...), exists
}

func (s *toolState) SetValue(key string, value json.RawMessage) error {
	if key == "" || !json.Valid(value) {
		return fmt.Errorf("state requires a non-empty key and valid JSON value")
	}
	s.values[key] = append(json.RawMessage(nil), value...)
	return nil
}
