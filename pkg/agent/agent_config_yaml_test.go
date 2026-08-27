// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package agent

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	yaml "gopkg.in/yaml.v3"
)

func TestAgentConfigYAMLUsesDocumentedSnakeCaseAndPreservesLegacySpelling(t *testing.T) {
	var canonical AgentConfig
	require.NoError(t, yaml.Unmarshal([]byte(`
id: canonical
name: Canonical
type: chat
model: model
provider: ollama
system_prompt: Use the configured prompt.
max_tokens: 2048
max_iterations: 7
enable_streaming: true
streaming_mode: stream
interrupt_on: [tool_call]
`), &canonical))
	assert.Equal(t, "Use the configured prompt.", canonical.SystemPrompt)
	assert.Equal(t, 2048, canonical.MaxTokens)
	assert.Equal(t, 7, canonical.MaxIterations)
	assert.True(t, canonical.EnableStreaming)
	assert.Equal(t, "stream", string(canonical.StreamingMode))
	assert.Equal(t, []string{"tool_call"}, canonical.InterruptOn)

	var legacy AgentConfig
	require.NoError(t, yaml.Unmarshal([]byte(`
id: legacy
name: Legacy
type: chat
model: model
provider: ollama
systemprompt: Legacy prompt.
maxtokens: 1024
maxiterations: 3
enablestreaming: true
streamingmode: stream
interrupton: [approval]
`), &legacy))
	assert.Equal(t, "Legacy prompt.", legacy.SystemPrompt)
	assert.Equal(t, 1024, legacy.MaxTokens)
	assert.Equal(t, 3, legacy.MaxIterations)
	assert.True(t, legacy.EnableStreaming)
	assert.Equal(t, "stream", string(legacy.StreamingMode))
	assert.Equal(t, []string{"approval"}, legacy.InterruptOn)

	encoded, err := yaml.Marshal(canonical)
	require.NoError(t, err)
	assert.Contains(t, string(encoded), "system_prompt: Use the configured prompt.")
	assert.Contains(t, string(encoded), "max_tokens: 2048")
	assert.NotContains(t, string(encoded), "systemprompt:")
}
