// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

package main

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"

	"github.com/UnicoLab/GoLangGraph/pkg/agent"
	"github.com/UnicoLab/GoLangGraph/pkg/core"
	"github.com/UnicoLab/GoLangGraph/pkg/llm"
)

// Config structs are written to disk as YAML by `golanggraph multi-agent init`.
// A func-typed field carries no YAML representation, and gopkg.in/yaml.v3
// panics rather than skipping it: adding EarlyExit to AgentConfig with only a
// `json:"-"` tag turned project scaffolding into a crash. Every func-typed
// field in a serialisable config needs `yaml:"-"` as well.
func TestConfigStructs_MarshalToYAMLWithoutPanicking(t *testing.T) {
	exit := func(content string, calls []llm.ToolCall) bool { return true }

	for name, value := range map[string]interface{}{
		"agent.AgentConfig": &agent.AgentConfig{
			ID:        "a1",
			Name:      "scaffolded",
			Type:      agent.AgentTypeChat,
			EarlyExit: exit,
		},
		"llm.CompletionRequest": &llm.CompletionRequest{
			Model:     "m",
			EarlyExit: exit,
		},
		"core.RetryPolicy": &core.RetryPolicy{
			MaxAttempts: 2,
			RetryIf:     func(error) bool { return false },
		},
		"core.Node": &core.Node{
			ID:       "n",
			Function: func(ctx context.Context, s *core.BaseState) (*core.BaseState, error) { return s, nil },
		},
	} {
		t.Run(name, func(t *testing.T) {
			out, err := yaml.Marshal(value)
			require.NoError(t, err, "%s must marshal to YAML", name)
			require.NotContains(t, string(out), "earlyexit",
				"a func field must be omitted, not emitted")
		})
	}
}
