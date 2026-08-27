// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package agent

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const declarativeAgentConfig = `
name: declarative-agents
version: "1.0"
agents:
  %s:
    id: %s
    name: %s
    type: chat
    model: fake-model
    provider: fake
    max_tokens: 256
`

func writeDeclarativeAgentConfig(t *testing.T, directory, name, agentID string) {
	t.Helper()
	contents := fmt.Sprintf(declarativeAgentConfig, agentID, agentID, agentID)
	if filepath.Ext(name) == ".JSON" {
		contents = fmt.Sprintf(`{"name":"declarative-agents","version":"1.0","agents":{"%[1]s":{"id":"%[1]s","name":"%[1]s","type":"chat","model":"fake-model","provider":"fake","max_tokens":256}}}`, agentID)
	}
	require.NoError(t, os.WriteFile(filepath.Join(directory, name), []byte(contents), 0o600))
}

func TestFileBasedAgentLoaderLoadsDeclarativeConfigurations(t *testing.T) {
	directory := t.TempDir()
	writeDeclarativeAgentConfig(t, directory, "alpha.yaml", "alpha")
	writeDeclarativeAgentConfig(t, directory, "beta.JSON", "beta")
	require.NoError(t, os.WriteFile(filepath.Join(directory, "ignored.txt"), []byte("not a config"), 0o600))

	registry := NewAgentRegistry()
	loader := NewFileBasedAgentLoader(registry)
	require.NoError(t, loader.LoadFromDirectory(directory))

	assert.Equal(t, []string{"alpha", "beta"}, registry.ListDefinitions())
	definition, found := registry.GetDefinition("alpha")
	require.True(t, found)
	assert.Equal(t, "alpha", definition.GetConfig().ID)
}

func TestFileBasedAgentLoaderIsAtomicOnRegistryCollision(t *testing.T) {
	directory := t.TempDir()
	writeDeclarativeAgentConfig(t, directory, "a-existing.yaml", "existing")
	writeDeclarativeAgentConfig(t, directory, "b-new.yaml", "new")

	registry := NewAgentRegistry()
	require.NoError(t, registry.RegisterDefinition("existing", NewBaseAgentDefinition(newTestAgentConfig("existing", "fake"))))

	err := NewFileBasedAgentLoader(registry).LoadFromDirectory(directory)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "already exists")
	_, found := registry.GetDefinition("new")
	assert.False(t, found, "a failed directory reload must not register a subset of agents")
	_, found = registry.GetDefinition("existing")
	assert.True(t, found)
}

func TestFileBasedAgentLoaderRejectsInvalidDirectories(t *testing.T) {
	t.Run("no configuration files", func(t *testing.T) {
		err := NewFileBasedAgentLoader(NewAgentRegistry()).LoadFromDirectory(t.TempDir())
		require.Error(t, err)
		assert.Contains(t, err.Error(), "no agent configuration files")
	})

	t.Run("nil registry", func(t *testing.T) {
		err := NewFileBasedAgentLoader(nil).LoadFromDirectory(t.TempDir())
		require.Error(t, err)
		assert.Contains(t, err.Error(), "not configured")
	})
}
