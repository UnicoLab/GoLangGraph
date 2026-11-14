// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package agentspec

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
)

// ParseManifest validates introspection data without loading executable code.
// The schema subset is deliberately closed: unknown fields are errors.
func ParseManifest(data []byte) (PackageManifest, error) {
	var manifest PackageManifest
	// Schemas nest properties and schema objects at each level, so their JSON
	// representation needs more depth than an ordinary validated input value.
	if err := validateJSON(data, 128); err != nil {
		return manifest, err
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&manifest); err != nil {
		return manifest, err
	}
	return manifest, manifest.Validate()
}

func (m PackageManifest) Validate() error {
	if err := ValidateIdentity(m.Module, m.Version); err != nil {
		return err
	}
	if m.ContractVersion != ContractVersion || strings.TrimSpace(m.Description) == "" || len(m.Agents) == 0 {
		return fmt.Errorf("manifest requires supported contract, description and agents")
	}
	seen := make(map[string]bool)
	for _, agent := range m.Agents {
		if !namePattern.MatchString(agent.Name) || strings.TrimSpace(agent.Description) == "" || seen[agent.Name] {
			return fmt.Errorf("invalid or duplicate agent %q", agent.Name)
		}
		seen[agent.Name] = true
		switch agent.Kind {
		case "prompt":
			if agent.Prompt == nil || strings.TrimSpace(agent.Prompt.System) == "" || agent.MaxModelCalls < 1 || agent.MaxModelCalls > 100 {
				return fmt.Errorf("agent %s requires a prompt and model-call budget", agent.Name)
			}
			if err := agent.Prompt.Model.Validate(); err != nil {
				return err
			}
		case "handler":
			if agent.Prompt != nil || agent.MaxModelCalls != 0 {
				return fmt.Errorf("handler %s cannot define a prompt or model-call budget", agent.Name)
			}
		default:
			return fmt.Errorf("unsupported agent kind %q", agent.Kind)
		}
		if err := validateBoundary(agent.InputSchema, agent.OutputSchema); err != nil {
			return fmt.Errorf("agent %s: %w", agent.Name, err)
		}
		capabilities := make(map[string]bool)
		for _, tool := range agent.Tools {
			if !namePattern.MatchString(tool.Name) || strings.TrimSpace(tool.Description) == "" || capabilities[tool.Name] {
				return fmt.Errorf("invalid or duplicate tool %q", tool.Name)
			}
			capabilities[tool.Name] = true
			if err := validateBoundary(tool.InputSchema, tool.OutputSchema); err != nil {
				return fmt.Errorf("tool %s: %w", tool.Name, err)
			}
		}
		for _, child := range agent.Subagents {
			module, entry, ok := strings.Cut(child.AgentID, "#")
			if !namePattern.MatchString(child.Name) || strings.TrimSpace(child.Description) == "" || capabilities[child.Name] || !ok || !namePattern.MatchString(entry) {
				return fmt.Errorf("invalid or duplicate child %q", child.Name)
			}
			capabilities[child.Name] = true
			if err := ValidateIdentity(module, child.PackageVersion); err != nil {
				return err
			}
			if err := validateBoundary(child.InputSchema, child.OutputSchema); err != nil {
				return fmt.Errorf("child %s: %w", child.Name, err)
			}
		}
	}
	return nil
}

func validateBoundary(input, output Schema) error {
	for _, schema := range []Schema{input, output} {
		if schema.Type != "object" {
			return fmt.Errorf("input/output schemas must be objects")
		}
		if err := schema.check(0); err != nil {
			return err
		}
	}
	return nil
}

func (s Schema) check(depth int) error {
	if depth > 32 {
		return fmt.Errorf("schema nesting exceeds 32 levels")
	}
	if s.Type != "object" && (len(s.Properties) > 0 || len(s.Required) > 0 || s.AdditionalProperties != nil) {
		return fmt.Errorf("object fields on non-object schema")
	}
	if s.Type != "array" && s.Items != nil {
		return fmt.Errorf("items on non-array schema")
	}
	if s.Type != "string" && (len(s.Enum) > 0 || s.MinLength != nil || s.MaxLength != nil) {
		return fmt.Errorf("string constraints on non-string schema")
	}
	switch s.Type {
	case "object":
		if s.AdditionalProperties == nil || *s.AdditionalProperties {
			return fmt.Errorf("object schemas must disallow additional properties")
		}
		seen := make(map[string]bool)
		for _, name := range s.Required {
			if _, ok := s.Properties[name]; !ok || seen[name] {
				return fmt.Errorf("invalid required field %q", name)
			}
			seen[name] = true
		}
		for _, property := range s.Properties {
			if err := property.check(depth + 1); err != nil {
				return err
			}
		}
	case "array":
		if s.Items == nil {
			return fmt.Errorf("array schema requires items")
		}
		return s.Items.check(depth + 1)
	case "string":
		if (s.MinLength != nil && *s.MinLength < 0) || (s.MaxLength != nil && *s.MaxLength < 0) || (s.MinLength != nil && s.MaxLength != nil && *s.MinLength > *s.MaxLength) {
			return fmt.Errorf("invalid string length constraints")
		}
		seen := make(map[string]bool)
		for _, value := range s.Enum {
			if seen[value] {
				return fmt.Errorf("duplicate enum value")
			}
			seen[value] = true
		}
	case "boolean", "integer", "number":
	default:
		return fmt.Errorf("unsupported schema type %q", s.Type)
	}
	return nil
}
