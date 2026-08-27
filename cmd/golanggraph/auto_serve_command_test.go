// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

package main

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/UnicoLab/GoLangGraph/pkg/agent"
	"github.com/UnicoLab/GoLangGraph/pkg/server"
	"github.com/UnicoLab/GoLangGraph/pkg/tools"
)

// agentConfigYAMLFor returns a one-agent multi-agent config whose agent id is
// unique to the test. Agent definitions live in a process-wide registry, so
// fixtures must not collide between tests.
func agentConfigYAMLFor(t *testing.T) (id, yaml string) {
	t.Helper()

	id = "agent-" + strings.ToLower(strings.NewReplacer("/", "-", "_", "-").Replace(t.Name()))
	return id, fmt.Sprintf(`name: fixture
agents:
  %s:
    id: %s
    name: Fixture Agent
    type: chat
    model: gpt-4
    provider: openai
    systemprompt: "hi"
    maxtokens: 1000
`, id, id)
}

// agentDirFor returns a directory holding one agent configuration unique to
// the test. Tests that do not want the built-in example agents registered as a
// side effect give the server a real agent to load instead.
func agentDirFor(t *testing.T) string {
	t.Helper()

	_, yaml := agentConfigYAMLFor(t)
	dir := t.TempDir()
	writeTestFile(t, dir, "agents.yaml", yaml)
	return dir
}

func defaultAutoServeConfig(port int) *server.AutoServerConfig {
	return &server.AutoServerConfig{
		Host:           "127.0.0.1",
		Port:           port,
		BasePath:       "/api",
		OllamaEndpoint: "http://localhost:11434",
		ServerTimeout:  30 * time.Second,
		MaxRequestSize: 1 << 20,
		LLMProviders:   map[string]interface{}{},
	}
}

// Regression: --timeout and --max-request-size were declared on the command and
// then never read, so neither ever reached the server configuration.
func TestAutoServe_EveryDeclaredFlagReachesTheConfiguration(t *testing.T) {
	t.Cleanup(func() { resetFlags(rootCmd) })

	require.NoError(t, autoServeCmd.ParseFlags([]string{
		"--host", "127.0.0.1",
		"--port", "9123",
		"--base-path", "/v2",
		"--timeout", "45s",
		"--max-request-size", "2048",
		"--ollama-endpoint", "http://ollama.internal:11434",
		"--playground=false",
		"--metrics=false",
		"--env", "staging",
		"--dev",
		"--watch",
		"--agent-dirs", "one,two",
		"--openai-api-key", "sk-test",
	}))

	config, opts, err := autoServeConfigFromFlags(autoServeCmd, []string{"./agents"})
	require.NoError(t, err)

	assert.Equal(t, "127.0.0.1", config.Host)
	assert.Equal(t, 9123, config.Port)
	assert.Equal(t, "/v2", config.BasePath)
	assert.Equal(t, 45*time.Second, config.ServerTimeout, "--timeout must reach the server configuration")
	assert.Equal(t, int64(2048), config.MaxRequestSize, "--max-request-size must reach the server configuration")
	assert.Equal(t, "http://ollama.internal:11434", config.OllamaEndpoint)
	assert.False(t, config.EnablePlayground)
	assert.False(t, config.EnableMetricsAPI)
	assert.Contains(t, config.LLMProviders, "openai")

	assert.Equal(t, "./agents", opts.SourcePath)
	assert.Equal(t, "staging", opts.Env)
	assert.True(t, opts.Dev)
	assert.True(t, opts.Watch)
	assert.Equal(t, []string{"one", "two"}, opts.AgentDirs)
}

// Regression: a source path that did not exist was skipped in silence, and the
// command went on to serve three example agents while reporting that it had
// loaded agents from the operator's path.
func TestPrepareAutoServer_MissingSourcePathFails(t *testing.T) {
	var out bytes.Buffer
	_, err := prepareAutoServer(&out, defaultAutoServeConfig(8080), autoServeOptions{
		SourcePath: filepath.Join(t.TempDir(), "not-here"),
		Env:        "development",
	})

	require.Error(t, err)
	assert.Contains(t, err.Error(), "agent source")
}

// Regression: a file whose extension was not .yaml/.yml was ignored without a
// word, so "auto-serve agents.json" served nothing the operator had asked for.
func TestPrepareAutoServer_UnsupportedSourceExtensionFails(t *testing.T) {
	dir := t.TempDir()
	source := writeTestFile(t, dir, "agents.txt", "not a config")

	_, err := prepareAutoServer(&bytes.Buffer{}, defaultAutoServeConfig(8080), autoServeOptions{
		SourcePath: source,
		Env:        "development",
	})

	require.Error(t, err)
	assert.Contains(t, err.Error(), "unsupported agent source")
}

func TestPrepareAutoServer_LoadsAgentsFromAConfigFile(t *testing.T) {
	id, yaml := agentConfigYAMLFor(t)
	source := writeTestFile(t, t.TempDir(), "agents.yaml", yaml)

	var out bytes.Buffer
	autoServer, err := prepareAutoServer(&out, defaultAutoServeConfig(8080), autoServeOptions{
		SourcePath: source,
		Env:        "development",
	})

	require.NoError(t, err)
	require.NotNil(t, autoServer)
	_, registered := agent.GetGlobalRegistry().GetDefinition(id)
	assert.True(t, registered, "the agent in the config file must be registered")
}

// server.AutoServer.LoadAgentsFromDirectory is a stub that loads nothing, so
// the CLI loads the configuration files in the directory itself.
func TestPrepareAutoServer_LoadsAgentsFromADirectory(t *testing.T) {
	id, yaml := agentConfigYAMLFor(t)
	dir := t.TempDir()
	writeTestFile(t, dir, "agents.yaml", yaml)
	writeTestFile(t, dir, "notes.md", "not a config")

	var out bytes.Buffer
	_, err := prepareAutoServer(&out, defaultAutoServeConfig(8080), autoServeOptions{
		SourcePath: dir,
		Env:        "development",
	})

	require.NoError(t, err)
	_, registered := agent.GetGlobalRegistry().GetDefinition(id)
	assert.True(t, registered, "agents defined in the directory must be registered")
	assert.Contains(t, out.String(), "1 agent config file(s) loaded")
}

func TestPrepareAutoServer_RejectsInvalidPortsAndEnvironments(t *testing.T) {
	_, err := prepareAutoServer(&bytes.Buffer{}, defaultAutoServeConfig(0), autoServeOptions{Env: "development"})
	require.Error(t, err)

	_, err = prepareAutoServer(&bytes.Buffer{}, defaultAutoServeConfig(8080), autoServeOptions{Env: "prod"})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "unknown environment")
}

func TestPrepareAutoServer_ProductionDisablesThePlayground(t *testing.T) {
	config := defaultAutoServeConfig(8080)
	config.EnablePlayground = true

	_, err := prepareAutoServer(&bytes.Buffer{}, config, autoServeOptions{SourcePath: agentDirFor(t), Env: "production"})

	require.NoError(t, err)
	assert.False(t, config.EnablePlayground, "the playground must not be exposed in production")
}

// Regression: --watch printed "👀 File watching enabled (hot-reload)" next to a
// comment saying the watcher "would go here". Nothing ever watched anything.
func TestPrepareAutoServer_UnimplementedFlagsSaySo(t *testing.T) {
	var out bytes.Buffer
	_, err := prepareAutoServer(&out, defaultAutoServeConfig(8080), autoServeOptions{
		SourcePath: agentDirFor(t),
		Env:        "development",
		Watch:      true,
		LogLevel:   "debug",
	})

	require.NoError(t, err)
	assert.Contains(t, out.String(), "--watch is not implemented")
	assert.Contains(t, out.String(), "--log-level is not applied")
	assert.NotContains(t, out.String(), "File watching enabled")
}

// Regression: a plugin that failed to load was reported as a warning and the
// server started anyway, without the agents the operator asked for.
func TestPrepareAutoServer_MissingPluginFails(t *testing.T) {
	dir := agentDirFor(t)

	_, err := prepareAutoServer(&bytes.Buffer{}, defaultAutoServeConfig(8080), autoServeOptions{
		SourcePath: dir,
		Env:        "development",
		Plugins:    []string{filepath.Join(dir, "absent.so")},
	})

	require.Error(t, err)
	assert.Contains(t, err.Error(), "failed to load plugin")
}

func TestPrepareAutoServer_GeneratesDeploymentFiles(t *testing.T) {
	dir := agentDirFor(t)
	config := defaultAutoServeConfig(9090)

	var out bytes.Buffer
	_, err := prepareAutoServer(&out, config, autoServeOptions{
		SourcePath:            dir,
		Env:                   "development",
		GenerateDockerfile:    true,
		GenerateDockerCompose: true,
		GenerateK8s:           true,
	})
	require.NoError(t, err)

	dockerfile, err := os.ReadFile(filepath.Join(dir, "Dockerfile"))
	require.NoError(t, err)
	assert.Contains(t, string(dockerfile), "FROM golang")

	compose, err := os.ReadFile(filepath.Join(dir, "docker-compose.yml"))
	require.NoError(t, err)
	assert.Contains(t, string(compose), "9090:8080", "the configured port must be used")

	manifests, err := os.ReadFile(filepath.Join(dir, "k8s-manifests.yaml"))
	require.NoError(t, err)
	assert.Contains(t, string(manifests), "kind: Deployment")
}

func TestGenerateDeploymentFiles_ReportFailures(t *testing.T) {
	missing := filepath.Join(t.TempDir(), "no-such-dir")

	require.Error(t, generateDockerfileForProject(missing))
	require.Error(t, generateDockerComposeForProject(missing, defaultAutoServeConfig(8080)))
	require.Error(t, generateKubernetesManifests(missing, defaultAutoServeConfig(8080)))
}

// The example agents are registered once per process; a failed registration
// used to be printed and ignored, leaving the server serving fewer agents than
// it announced.
func TestCreateExampleAgents_RegistersEveryAgentAndReportsCollisions(t *testing.T) {
	autoServer := server.NewAutoServer(defaultAutoServeConfig(8080))

	var out bytes.Buffer
	require.NoError(t, createExampleAgents(&out, autoServer))

	for _, id := range []string{"chat", "react", "tools"} {
		_, registered := agent.GetGlobalRegistry().GetDefinition(id)
		assert.True(t, registered, "example agent %q must be registered", id)
	}

	err := createExampleAgents(&bytes.Buffer{}, autoServer)
	require.Error(t, err, "a registration that failed must be reported, not printed and ignored")
}

// Every tool an example agent asks for must exist, or the agent advertises a
// capability it can never use.
func TestExampleAgents_OnlyRequestRegisteredTools(t *testing.T) {
	registry := agent.GetGlobalRegistry()
	definition, ok := registry.GetDefinition("tools")
	if !ok {
		t.Skip("the example agents have not been registered in this run")
	}

	known := map[string]bool{}
	for _, name := range tools.NewToolRegistry().ListTools() {
		known[name] = true
	}
	for _, tool := range definition.GetConfig().Tools {
		assert.True(t, known[tool], "example agent requests unregistered tool %q", tool)
	}
}
