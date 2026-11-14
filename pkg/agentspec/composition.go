// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package agentspec

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
)

// Execution grants a handler access only to its declared tools and child agents.
// The host owns identities, storage and dependency resolution.
type Execution interface {
	State
	InvokeTool(context.Context, string, json.RawMessage) (json.RawMessage, error)
	InvokeAgent(context.Context, string, json.RawMessage) (json.RawMessage, error)
	CheckAgent(string, Schema, Schema) error
}

type SubagentManifest struct {
	Name           string `json:"name"`
	Description    string `json:"description"`
	AgentID        string `json:"agent_id"`
	PackageVersion string `json:"package_version"`
	InputSchema    Schema `json:"input_schema"`
	OutputSchema   Schema `json:"output_schema"`
}

type Subagent struct{ manifest SubagentManifest }

// NewSubagent declares one persistent child per parent thread and alias.
func NewSubagent[I, O any](name, description, agentID, version string) (Subagent, error) {
	module, entry, ok := strings.Cut(agentID, "#")
	if !namePattern.MatchString(name) || strings.TrimSpace(description) == "" || !ok || module == "" || !namePattern.MatchString(entry) || !versionPattern.MatchString(version) {
		return Subagent{}, fmt.Errorf("child requires a name, description, qualified agent ID and exact version")
	}
	if err := ValidateIdentity(module, version); err != nil {
		return Subagent{}, err
	}
	input, output, err := schemas[I, O]()
	if err != nil {
		return Subagent{}, err
	}
	return Subagent{SubagentManifest{name, description, agentID, version, input, output}}, nil
}

func (s Subagent) Manifest() SubagentManifest {
	m := s.manifest
	m.InputSchema, m.OutputSchema = m.InputSchema.clone(), m.OutputSchema.clone()
	return m
}

type HandlerConfig struct {
	Name        string
	Description string
	Tools       []Tool
	Subagents   []Subagent
}

// NewHandler defines a deterministic Go turn with no mandatory model/provider.
// It can compose declared agents and tools using the supplied Execution.
func NewHandler[I, O any](config HandlerConfig, fn func(context.Context, Execution, I) (O, error)) (Definition, error) {
	if !namePattern.MatchString(config.Name) || strings.TrimSpace(config.Description) == "" || fn == nil {
		return Definition{}, fmt.Errorf("handler requires a name, description and function")
	}
	input, output, err := schemas[I, O]()
	if err != nil {
		return Definition{}, err
	}
	m := AgentManifest{Kind: "handler", Name: config.Name, Description: config.Description, InputSchema: input, OutputSchema: output, Tools: []ToolManifest{}}
	seen := make(map[string]bool)
	for _, tool := range config.Tools {
		if tool.execute == nil || seen[tool.manifest.Name] {
			return Definition{}, fmt.Errorf("uninitialized or duplicate tool %q", tool.manifest.Name)
		}
		seen[tool.manifest.Name] = true
		m.Tools = append(m.Tools, tool.Manifest())
	}
	for _, child := range config.Subagents {
		if child.manifest.AgentID == "" || seen[child.manifest.Name] {
			return Definition{}, fmt.Errorf("uninitialized or duplicate child/tool %q", child.manifest.Name)
		}
		seen[child.manifest.Name] = true
		m.Subagents = append(m.Subagents, child.Manifest())
	}
	d := Definition{
		manifest: m, tools: append([]Tool(nil), config.Tools...), subagents: append([]Subagent(nil), config.Subagents...),
		inputType: reflect.TypeOf((*I)(nil)).Elem(), outputType: reflect.TypeOf((*O)(nil)).Elem(),
		validateInput:  func(raw json.RawMessage) error { _, err := decode[I](raw, input); return err },
		validateOutput: func(raw json.RawMessage) error { _, err := decode[O](raw, output); return err },
	}
	d.handler = func(ctx context.Context, state Execution, raw json.RawMessage) (json.RawMessage, error) {
		value, err := decode[I](raw, input)
		if err != nil {
			return nil, err
		}
		result, err := fn(ctx, state, value)
		if err != nil {
			return nil, err
		}
		data, err := json.Marshal(result)
		if err != nil {
			return nil, err
		}
		if err := d.ValidateOutput(data); err != nil {
			return nil, err
		}
		return data, nil
	}
	return d, nil
}

func (d Definition) Handle(ctx context.Context, state Execution, input json.RawMessage) (json.RawMessage, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if state == nil {
		return nil, fmt.Errorf("handler requires an execution state")
	}
	if d.handler == nil {
		return nil, fmt.Errorf("agent is not a Go handler")
	}
	return d.handler(ctx, state, input)
}

// CallAgent encodes typed input and decodes a child's validated output. Declare
// the same types with NewSubagent so registration verifies the boundary.
func CallAgent[I, O any](ctx context.Context, state Execution, name string, input I) (O, error) {
	var output O
	inputSchema, outputSchema, err := schemas[I, O]()
	if err != nil {
		return output, err
	}
	if typeErr := state.CheckAgent(name, inputSchema, outputSchema); typeErr != nil {
		return output, typeErr
	}
	data, err := json.Marshal(input)
	if err != nil {
		return output, err
	}
	if _, validationErr := decode[I](data, inputSchema); validationErr != nil {
		return output, validationErr
	}
	data, err = state.InvokeAgent(ctx, name, data)
	if err != nil {
		return output, err
	}
	return decode[O](data, outputSchema)
}
