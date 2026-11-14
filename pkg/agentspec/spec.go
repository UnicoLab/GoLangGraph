// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

// Package agentspec defines the experimental Go-native agent package contract.
// It depends only on the standard library, not on GoLangGraph's graph engine.
// Definitions are immutable; mutable tool data belongs in the invocation State.
package agentspec

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math"
	"path"
	"reflect"
	"regexp"
	"strings"
)

const ContractVersion = "golanggraph.agent/v1alpha1"

// Model is a package's default model profile. Credentials belong to the host.
type Model struct {
	Provider    string  `json:"provider"`
	Name        string  `json:"name"`
	Temperature float64 `json:"temperature"`
	MaxTokens   int     `json:"max_tokens"`
}

func (m Model) Validate() error {
	if strings.TrimSpace(m.Provider) == "" || strings.TrimSpace(m.Name) == "" {
		return fmt.Errorf("model provider and name are required")
	}
	if math.IsNaN(m.Temperature) || math.IsInf(m.Temperature, 0) || m.Temperature < 0 || m.Temperature > 2 {
		return fmt.Errorf("temperature must be between 0 and 2")
	}
	if m.MaxTokens <= 0 {
		return fmt.Errorf("max_tokens must be positive")
	}
	return nil
}

type Prompt struct {
	System string `json:"system"`
	Model  Model  `json:"model"`
}

// State is scoped to one thread and valid only for the duration of a tool call.
// Values are JSON, copied on access and committed with a successful turn. Tool
// functions must not retain State or capture mutable per-conversation globals.
type State interface {
	ThreadID() string
	GetValue(key string) (json.RawMessage, bool)
	SetValue(key string, value json.RawMessage) error
}

type ToolManifest struct {
	Name         string `json:"name"`
	Description  string `json:"description"`
	InputSchema  Schema `json:"input_schema"`
	OutputSchema Schema `json:"output_schema"`
}

type Tool struct {
	manifest ToolManifest
	execute  func(context.Context, State, json.RawMessage) (json.RawMessage, error)
}

var namePattern = regexp.MustCompile(`^[a-z][a-z0-9_]{0,63}$`)

// NewTool derives schemas from Go types and validates both arguments and results.
// Tool errors are returned to the model by the runtime; cancellation stops a run.
func NewTool[I, O any](name, description string, fn func(context.Context, State, I) (O, error)) (Tool, error) {
	if !namePattern.MatchString(name) || strings.TrimSpace(description) == "" || fn == nil {
		return Tool{}, fmt.Errorf("tool requires a snake_case name, description and function")
	}
	input, output, err := schemas[I, O]()
	if err != nil {
		return Tool{}, err
	}
	tool := Tool{manifest: ToolManifest{name, description, input, output}}
	tool.execute = func(ctx context.Context, state State, raw json.RawMessage) (json.RawMessage, error) {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		args, err := decode[I](raw, input)
		if err != nil {
			return nil, fmt.Errorf("tool %s input: %w", name, err)
		}
		result, err := fn(ctx, state, args)
		if err != nil {
			return nil, err
		}
		data, err := json.Marshal(result)
		if err != nil {
			return nil, err
		}
		if _, err := decode[O](data, output); err != nil {
			return nil, fmt.Errorf("tool %s output: %w", name, err)
		}
		return data, nil
	}
	return tool, nil
}

func (t Tool) Manifest() ToolManifest {
	m := t.manifest
	m.InputSchema, m.OutputSchema = m.InputSchema.clone(), m.OutputSchema.clone()
	return m
}

func (t Tool) Execute(ctx context.Context, state State, input json.RawMessage) (json.RawMessage, error) {
	if t.execute == nil || state == nil {
		return nil, fmt.Errorf("uninitialized tool or missing invocation state")
	}
	return t.execute(ctx, state, input)
}

type AgentManifest struct {
	Kind          string             `json:"kind"`
	Name          string             `json:"name"`
	Description   string             `json:"description"`
	InputSchema   Schema             `json:"input_schema"`
	OutputSchema  Schema             `json:"output_schema"`
	Prompt        *Prompt            `json:"prompt,omitempty"`
	Tools         []ToolManifest     `json:"tools"`
	Subagents     []SubagentManifest `json:"subagents,omitempty"`
	MaxModelCalls int                `json:"max_model_calls"`
}

type AgentConfig struct {
	Name          string
	Description   string
	Prompt        Prompt
	Tools         []Tool
	Subagents     []Subagent
	MaxModelCalls int
}

// Definition describes a prompt/tool agent without holding a live conversation.
type Definition struct {
	manifest       AgentManifest
	tools          []Tool
	subagents      []Subagent
	handler        func(context.Context, Execution, json.RawMessage) (json.RawMessage, error)
	inputType      reflect.Type
	outputType     reflect.Type
	validateInput  func(json.RawMessage) error
	validateOutput func(json.RawMessage) error
}

func NewAgent[I, O any](config AgentConfig) (Definition, error) {
	if !namePattern.MatchString(config.Name) || strings.TrimSpace(config.Description) == "" || strings.TrimSpace(config.Prompt.System) == "" {
		return Definition{}, fmt.Errorf("agent requires a snake_case name, description and system prompt")
	}
	if err := config.Prompt.Model.Validate(); err != nil {
		return Definition{}, err
	}
	if config.MaxModelCalls == 0 {
		config.MaxModelCalls = 8
	}
	if config.MaxModelCalls < 1 || config.MaxModelCalls > 100 {
		return Definition{}, fmt.Errorf("max_model_calls must be between 1 and 100")
	}
	input, output, err := schemas[I, O]()
	if err != nil {
		return Definition{}, err
	}
	manifest := AgentManifest{
		Kind: "prompt", Name: config.Name, Description: config.Description, Prompt: &config.Prompt,
		InputSchema: input, OutputSchema: output, MaxModelCalls: config.MaxModelCalls,
		Tools: make([]ToolManifest, 0, len(config.Tools)),
	}
	seen := make(map[string]bool)
	for _, tool := range config.Tools {
		if tool.execute == nil || seen[tool.manifest.Name] {
			return Definition{}, fmt.Errorf("uninitialized or duplicate tool %q", tool.manifest.Name)
		}
		seen[tool.manifest.Name] = true
		manifest.Tools = append(manifest.Tools, tool.Manifest())
	}
	for _, child := range config.Subagents {
		if child.manifest.AgentID == "" || seen[child.manifest.Name] {
			return Definition{}, fmt.Errorf("uninitialized or duplicate child/tool %q", child.manifest.Name)
		}
		seen[child.manifest.Name] = true
		manifest.Subagents = append(manifest.Subagents, child.Manifest())
	}
	return Definition{
		manifest: manifest, tools: append([]Tool(nil), config.Tools...),
		subagents: append([]Subagent(nil), config.Subagents...),
		inputType: reflect.TypeOf((*I)(nil)).Elem(), outputType: reflect.TypeOf((*O)(nil)).Elem(),
		validateInput:  func(data json.RawMessage) error { _, err := decode[I](data, input); return err },
		validateOutput: func(data json.RawMessage) error { _, err := decode[O](data, output); return err },
	}, nil
}

func schemas[I, O any]() (Schema, Schema, error) {
	input, err := schemaFor(reflect.TypeOf((*I)(nil)).Elem(), make(map[reflect.Type]bool))
	if err != nil {
		return Schema{}, Schema{}, fmt.Errorf("input schema: %w", err)
	}
	output, err := schemaFor(reflect.TypeOf((*O)(nil)).Elem(), make(map[reflect.Type]bool))
	if err != nil {
		return Schema{}, Schema{}, fmt.Errorf("output schema: %w", err)
	}
	if input.Type != "object" || output.Type != "object" {
		return Schema{}, Schema{}, fmt.Errorf("input and output must be structs")
	}
	return input, output, nil
}

func (d Definition) Manifest() AgentManifest {
	m := d.manifest
	if m.Prompt != nil {
		prompt := *m.Prompt
		m.Prompt = &prompt
	}
	m.InputSchema, m.OutputSchema = m.InputSchema.clone(), m.OutputSchema.clone()
	m.Tools = make([]ToolManifest, len(d.tools))
	for i, tool := range d.tools {
		m.Tools[i] = tool.Manifest()
	}
	m.Subagents = make([]SubagentManifest, len(d.subagents))
	for i, child := range d.subagents {
		m.Subagents[i] = child.Manifest()
	}
	return m
}

func (d Definition) Tools() []Tool { return append([]Tool(nil), d.tools...) }

// CheckTypes rejects mismatched typed calls before any work or state writes.
// Callers use the package's exported types (or aliases of those types).
func CheckTypes[I, O any](d Definition) error {
	if d.inputType != reflect.TypeOf((*I)(nil)).Elem() || d.outputType != reflect.TypeOf((*O)(nil)).Elem() {
		return fmt.Errorf("typed invocation requires input %v and output %v", d.inputType, d.outputType)
	}
	return nil
}

// Fingerprint identifies the declarative behavior of an agent. It detects
// changes to prompts, schemas, models and dependencies, not changes to Go code.
func (d Definition) Fingerprint() string {
	data, _ := json.Marshal(d.Manifest())
	return fmt.Sprintf("sha256:%x", sha256.Sum256(data))
}

func (d Definition) ValidateInput(data json.RawMessage) error {
	if d.validateInput == nil {
		return fmt.Errorf("uninitialized definition")
	}
	return d.validateInput(data)
}

func (d Definition) ValidateOutput(data json.RawMessage) error {
	if d.validateOutput == nil {
		return fmt.Errorf("uninitialized definition")
	}
	return d.validateOutput(data)
}

type PackageManifest struct {
	ContractVersion string          `json:"contract_version"`
	Module          string          `json:"module"`
	Version         string          `json:"version"`
	Description     string          `json:"description"`
	Agents          []AgentManifest `json:"agents"`
}

// Package contains code and metadata distributed as an ordinary Go module.
// A version is an exact identity, not a runtime dependency-resolution request.
type Package struct {
	module, version, description string
	definitions                  []Definition
}

var versionPattern = regexp.MustCompile(`^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(-[0-9A-Za-z-]+(\.[0-9A-Za-z-]+)*)?(\+[0-9A-Za-z-]+(\.[0-9A-Za-z-]+)*)?$`)
var modulePattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._~/-]*$`)

// ValidateIdentity checks a portable module path and exact semantic version.
func ValidateIdentity(module, version string) error {
	if !modulePattern.MatchString(module) || path.Clean(module) != module || !strings.Contains(module, "/") {
		return fmt.Errorf("invalid package module %q", module)
	}
	for _, segment := range strings.Split(module, "/") {
		if segment == "." || segment == ".." {
			return fmt.Errorf("invalid package module %q", module)
		}
	}
	if !versionPattern.MatchString(version) {
		return fmt.Errorf("package version must be a v-prefixed semantic version")
	}
	prerelease := strings.SplitN(strings.SplitN(version, "+", 2)[0], "-", 2)
	if len(prerelease) == 2 {
		for _, identifier := range strings.Split(prerelease[1], ".") {
			if len(identifier) > 1 && identifier[0] == '0' && strings.Trim(identifier, "0123456789") == "" {
				return fmt.Errorf("numeric prerelease identifiers cannot have leading zeros")
			}
		}
	}
	return nil
}

func NewPackage(module, version, description string, definitions ...Definition) (Package, error) {
	if err := ValidateIdentity(module, version); err != nil {
		return Package{}, err
	}
	if strings.TrimSpace(description) == "" || len(definitions) == 0 {
		return Package{}, fmt.Errorf("package description and at least one agent are required")
	}
	seen := make(map[string]bool)
	for _, definition := range definitions {
		if definition.validateInput == nil || seen[definition.manifest.Name] {
			return Package{}, fmt.Errorf("uninitialized or duplicate agent %q", definition.manifest.Name)
		}
		seen[definition.manifest.Name] = true
	}
	return Package{module, version, description, append([]Definition(nil), definitions...)}, nil
}

func (p Package) Manifest() PackageManifest {
	m := PackageManifest{ContractVersion, p.module, p.version, p.description, make([]AgentManifest, len(p.definitions))}
	for i, definition := range p.definitions {
		m.Agents[i] = definition.Manifest()
	}
	return m
}

func (p Package) Definitions() []Definition { return append([]Definition(nil), p.definitions...) }

// AgentID qualifies an entry point with its module path. One version of a module
// may be registered in a host; the thread separately pins the package version.
func (p Package) AgentID(name string) string { return p.module + "#" + name }
