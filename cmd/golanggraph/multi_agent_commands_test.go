// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/spf13/cobra"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// A multi-agent file with no routing and no deployment section: both are
// optional pointers in MultiAgentConfig.
const minimalMultiAgentYAML = `name: minimal
agents:
  a1:
    id: a1
    name: A1
    type: chat
    model: gpt-3.5-turbo
    provider: openai
    systemprompt: "You are A1."
    maxtokens: 1000
`

// A complete file: note that agent fields are spelled the way gopkg.in/yaml.v3
// maps them onto agent.AgentConfig, which carries JSON tags only.
const fullMultiAgentYAML = `name: fleet
version: "1.0.0"
agents:
  alpha:
    id: alpha
    name: Alpha
    type: chat
    model: gpt-4
    provider: openai
    systemprompt: "You are Alpha."
    maxtokens: 1000
    tools:
      - calculator
  beta:
    id: beta
    name: Beta
    type: react
    model: gpt-4
    provider: openai
    systemprompt: "You are Beta."
    maxtokens: 1000
routing:
  type: path
  default_agent: alpha
  rules:
    - id: rule-1
      pattern: /alpha
      agent_id: alpha
      method: POST
    - id: rule-2
      pattern: /beta
      agent_id: beta
      method: POST
deployment:
  type: docker
  environment: development
  replicas: 3
`

// Regression: runMultiAgentValidate printed "✅ Configuration validation
// passed!" and then dereferenced config.Routing and config.Deployment
// unconditionally, so any configuration without those optional sections
// panicked with a nil pointer dereference after reporting success.
func TestMultiAgentValidate_ConfigWithoutRoutingOrDeployment(t *testing.T) {
	path := writeTestFile(t, t.TempDir(), "minimal.yaml", minimalMultiAgentYAML)

	var out bytes.Buffer
	require.NotPanics(t, func() {
		require.NoError(t, runMultiAgentValidate(&out, []string{path}, false, true))
	})
	assert.Contains(t, out.String(), "Routing rules: none configured")
	assert.Contains(t, out.String(), "Deployment type: none configured")
}

func TestMultiAgentValidate_AcceptsACompleteConfiguration(t *testing.T) {
	path := writeTestFile(t, t.TempDir(), "multi.yaml", fullMultiAgentYAML)

	var out bytes.Buffer
	require.NoError(t, runMultiAgentValidate(&out, []string{path}, true, true))
	assert.Contains(t, out.String(), "Agents: 2")
	assert.Contains(t, out.String(), "Routing rules: 2")
}

func TestMultiAgentValidate_MissingAndMalformedFiles(t *testing.T) {
	dir := t.TempDir()

	require.Error(t, runMultiAgentValidate(&bytes.Buffer{}, []string{filepath.Join(dir, "absent.yaml")}, false, true))

	broken := writeTestFile(t, dir, "broken.yaml", "agents: [\n")
	require.Error(t, runMultiAgentValidate(&bytes.Buffer{}, []string{broken}, false, true))

	empty := writeTestFile(t, dir, "noagents.yaml", "name: nobody\n")
	err := runMultiAgentValidate(&bytes.Buffer{}, []string{empty}, false, true)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "at least one agent")
}

// Regression: validateStrictMode was `return nil` with a "add strict validation
// logic here" comment, so --strict reported that strict validation had passed
// without performing any.
func TestMultiAgentValidate_StrictFindsRealProblems(t *testing.T) {
	path := writeTestFile(t, t.TempDir(), "multi.yaml", `name: fleet
agents:
  alpha:
    id: alpha
    name: Alpha
    type: chat
    model: gpt-4
    provider: openai
    maxtokens: 1000
  beta:
    id: wrong-id
    name: Beta
    type: chat
    model: gpt-4
    provider: mystery-inc
    systemprompt: "hi"
    maxtokens: 1000
routing:
  type: path
  rules:
    - id: rule-1
      pattern: /same
      agent_id: alpha
    - id: rule-2
      pattern: /same
      agent_id: beta
`)

	var out bytes.Buffer
	require.NoError(t, runMultiAgentValidate(&out, []string{path}, false, true),
		"these are strict-mode problems only")

	out.Reset()
	err := runMultiAgentValidate(&out, []string{path}, true, true)
	require.Error(t, err)

	reported := out.String()
	assert.Contains(t, reported, "no system prompt")
	assert.Contains(t, reported, "does not match its key")
	assert.Contains(t, reported, "not one of the built-in providers")
	assert.Contains(t, reported, "no default agent")
	assert.Contains(t, reported, `pattern "/same" is claimed by both`)
}

// Regression: validateAgentSchemas was `return nil` while --check-schemas
// defaults to true, so every run reported schema validation it never did.
func TestMultiAgentValidate_SchemaCheckFindsUnknownToolsAndTypes(t *testing.T) {
	dir := t.TempDir()
	badType := writeTestFile(t, dir, "badtype.yaml", `name: fleet
agents:
  beta:
    id: beta
    name: Beta
    type: reactt
    model: gpt-4
    provider: openai
    systemprompt: "hi"
    maxtokens: 1000
`)
	unknownTool := writeTestFile(t, dir, "tool.yaml", `name: fleet
agents:
  alpha:
    id: alpha
    name: Alpha
    type: chat
    model: gpt-4
    provider: openai
    systemprompt: "hi"
    maxtokens: 1000
    tools:
      - no_such_tool
`)

	var out bytes.Buffer
	require.NoError(t, runMultiAgentValidate(&out, []string{badType}, false, false),
		"--check-schemas=false must not run the schema checks")

	out.Reset()
	err := runMultiAgentValidate(&out, []string{badType}, false, true)
	require.Error(t, err, "an agent type the runtime does not implement must be reported")
	assert.Contains(t, out.String(), `unknown type "reactt"`)

	// A tool the CLI does not know may still be registered by the operator's
	// own code, so it is a warning that --strict turns into a failure.
	out.Reset()
	require.NoError(t, runMultiAgentValidate(&out, []string{unknownTool}, false, true))
	assert.Contains(t, out.String(), `tool "no_such_tool" is not registered`)
	require.Error(t, runMultiAgentValidate(&bytes.Buffer{}, []string{unknownTool}, true, true))
}

// A configuration that merely leaves optional keys unset is not broken: the
// runtime defaults them, and so must the validator.
func TestMultiAgentValidate_OmittedOptionalKeysAreDefaulted(t *testing.T) {
	path := writeTestFile(t, t.TempDir(), "multi.yaml", `name: fleet
agents:
  alpha:
    id: alpha
    name: Alpha
    type: chat
    model: gpt-4
    provider: openai
    systemprompt: "hi"
`)

	require.NoError(t, runMultiAgentValidate(&bytes.Buffer{}, []string{path}, false, true))
}

// Regression: deployDocker/deployKubernetes/deployServerless printed
// "Deploying to Docker..." and returned without deploying anything, and the
// command exited 0.
func TestMultiAgentDeploy_DoesNotClaimAnUndoneDeployment(t *testing.T) {
	path := writeTestFile(t, t.TempDir(), "multi.yaml", fullMultiAgentYAML)

	var out bytes.Buffer
	err := runMultiAgentDeploy(&out, []string{path}, "docker", "production", false)

	require.Error(t, err, "a deployment that did not happen must not exit zero")
	assert.ErrorIs(t, err, errNotImplemented)
	assert.NotContains(t, out.String(), "Deploying to Docker...")
}

func TestMultiAgentDeploy_DryRunListsAgentsAndSucceeds(t *testing.T) {
	path := writeTestFile(t, t.TempDir(), "multi.yaml", fullMultiAgentYAML)

	var out bytes.Buffer
	require.NoError(t, runMultiAgentDeploy(&out, []string{path}, "docker", "development", true))

	listing := out.String()
	assert.Contains(t, listing, "alpha")
	assert.Contains(t, listing, "beta")
	assert.Less(t, strings.Index(listing, "alpha"), strings.Index(listing, "beta"),
		"agents must be listed in a stable order")
}

// Regression: `config.Deployment.Type = deploymentType` panicked whenever the
// configuration had no deployment section.
func TestMultiAgentDeploy_ConfigWithoutDeploymentSection(t *testing.T) {
	path := writeTestFile(t, t.TempDir(), "minimal.yaml", minimalMultiAgentYAML)

	require.NotPanics(t, func() {
		require.NoError(t, runMultiAgentDeploy(&bytes.Buffer{}, []string{path}, "docker", "", true))
	})
}

func TestMultiAgentDeploy_UnknownDeploymentTypeIsRejected(t *testing.T) {
	path := writeTestFile(t, t.TempDir(), "multi.yaml", fullMultiAgentYAML)

	err := runMultiAgentDeploy(&bytes.Buffer{}, []string{path}, "carrier-pigeon", "", false)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "unsupported deployment type")
}

// Regression: runGenerateDocker printed "Generating Docker deployment files..."
// and generated nothing, ignoring --output and --multi-service entirely.
func TestMultiAgentGenerateDocker_WritesTheFiles(t *testing.T) {
	dir := t.TempDir()
	path := writeTestFile(t, dir, "multi.yaml", fullMultiAgentYAML)
	outputDir := filepath.Join(dir, "deploy")

	var out bytes.Buffer
	require.NoError(t, runGenerateDocker(&out, []string{path}, outputDir, true))

	compose, err := os.ReadFile(filepath.Join(outputDir, "docker-compose.yml"))
	require.NoError(t, err, "the compose file must actually exist")
	assert.Contains(t, string(compose), "alpha:")
	assert.Contains(t, string(compose), "beta:")

	dockerfile, err := os.ReadFile(filepath.Join(outputDir, "Dockerfile"))
	require.NoError(t, err)
	assert.Contains(t, string(dockerfile), "FROM golang")
}

func TestMultiAgentGenerateDocker_SingleServiceMode(t *testing.T) {
	dir := t.TempDir()
	path := writeTestFile(t, dir, "multi.yaml", fullMultiAgentYAML)
	outputDir := filepath.Join(dir, "deploy")

	require.NoError(t, runGenerateDocker(&bytes.Buffer{}, []string{path}, outputDir, false))

	compose, err := os.ReadFile(filepath.Join(outputDir, "docker-compose.yml"))
	require.NoError(t, err)
	assert.Contains(t, string(compose), "multi-agent:")
	assert.NotContains(t, string(compose), "alpha:", "--multi-service=false must produce one service")
}

func TestMultiAgentGenerateDocker_ReportsAMissingConfig(t *testing.T) {
	dir := t.TempDir()
	err := runGenerateDocker(&bytes.Buffer{}, []string{filepath.Join(dir, "absent.yaml")}, filepath.Join(dir, "deploy"), true)
	require.Error(t, err)
	assert.NoDirExists(t, filepath.Join(dir, "deploy"))
}

// Regression: runGenerateK8s printed "Generating Kubernetes manifests..." and
// generated nothing, ignoring --output and --namespace entirely.
func TestMultiAgentGenerateK8s_WritesManifestsInTheNamespace(t *testing.T) {
	dir := t.TempDir()
	path := writeTestFile(t, dir, "multi.yaml", fullMultiAgentYAML)
	outputDir := filepath.Join(dir, "k8s")

	var out bytes.Buffer
	require.NoError(t, runGenerateK8s(&out, []string{path}, outputDir, "prod"))

	for _, name := range []string{"namespace.yaml", "deployment.yaml", "service.yaml"} {
		content, err := os.ReadFile(filepath.Join(outputDir, name))
		require.NoError(t, err, "%s must be generated", name)
		assert.Contains(t, string(content), "prod", "%s must use the requested namespace", name)
	}

	deployment, err := os.ReadFile(filepath.Join(outputDir, "deployment.yaml"))
	require.NoError(t, err)
	assert.Contains(t, string(deployment), "replicas: 3", "the configured replica count must be used")
}

// Regression: `--watch` fell into `select {}` and blocked forever after
// claiming to be watching for status changes.
func TestMultiAgentStatus_WatchIsReportedAsNotImplemented(t *testing.T) {
	path := writeTestFile(t, t.TempDir(), "multi.yaml", fullMultiAgentYAML)

	done := make(chan error, 1)
	go func() { done <- runMultiAgentStatus(&bytes.Buffer{}, []string{path}, "table", true) }()

	select {
	case err := <-done:
		require.Error(t, err)
		assert.ErrorIs(t, err, errNotImplemented)
	case <-time.After(10 * time.Second):
		t.Fatal("multi-agent status --watch blocked instead of reporting that it is not implemented")
	}
}

func TestMultiAgentStatus_Formats(t *testing.T) {
	path := writeTestFile(t, t.TempDir(), "multi.yaml", fullMultiAgentYAML)

	t.Run("table", func(t *testing.T) {
		var out bytes.Buffer
		require.NoError(t, runMultiAgentStatus(&out, []string{path}, "table", false))
		assert.Contains(t, out.String(), "alpha")
		assert.Contains(t, out.String(), "does not contact a running deployment",
			"the command reads the configuration; it must not imply live status")
	})

	t.Run("json", func(t *testing.T) {
		var out bytes.Buffer
		require.NoError(t, runMultiAgentStatus(&out, []string{path}, "json", false))

		var decoded struct {
			Agents map[string]struct {
				Name string `json:"name"`
			} `json:"agents"`
		}
		start := strings.Index(out.String(), "{")
		require.GreaterOrEqual(t, start, 0)
		require.NoError(t, json.Unmarshal([]byte(out.String()[start:]), &decoded))
		assert.Equal(t, "Alpha", decoded.Agents["alpha"].Name)
	})

	t.Run("unsupported", func(t *testing.T) {
		err := runMultiAgentStatus(&bytes.Buffer{}, []string{path}, "toml", false)
		require.Error(t, err)
	})
}

func TestMultiAgentInit_CreatesAProjectItsOwnValidatorAccepts(t *testing.T) {
	dir := chdirTemp(t)

	var out bytes.Buffer
	require.NoError(t, runMultiAgentInit(&out, []string{"fleet"}, "basic", 2, "yaml", "path"))

	configPath := filepath.Join(dir, "fleet", "configs", "multi-agent.yaml")
	assert.FileExists(t, configPath)
	assert.FileExists(t, filepath.Join(dir, "fleet", "docker-compose.yml"))
	assert.FileExists(t, filepath.Join(dir, "fleet", "k8s", "deployment.yaml"))
	assert.FileExists(t, filepath.Join(dir, "fleet", "k8s", "service.yaml"))
	assert.FileExists(t, filepath.Join(dir, "fleet", "README.md"))
	assert.FileExists(t, filepath.Join(dir, "fleet", "agents", "agent-1", "config.yaml"))
	assert.FileExists(t, filepath.Join(dir, "fleet", "agents", "agent-2", "config.yaml"))

	require.NoError(t, runMultiAgentValidate(&bytes.Buffer{}, []string{configPath}, false, true),
		"the project init generates must pass the validate step init tells you to run")
}

func TestMultiAgentInit_JSONFormat(t *testing.T) {
	dir := chdirTemp(t)

	require.NoError(t, runMultiAgentInit(&bytes.Buffer{}, []string{"fleet"}, "basic", 1, "json", "path"))

	raw, err := os.ReadFile(filepath.Join(dir, "fleet", "configs", "multi-agent.json"))
	require.NoError(t, err)
	var decoded map[string]interface{}
	require.NoError(t, json.Unmarshal(raw, &decoded))
	assert.Contains(t, decoded, "agents")
}

func TestMultiAgentInit_RejectsBadInput(t *testing.T) {
	dir := chdirTemp(t)

	for name, run := range map[string]func() error{
		"escaping name": func() error {
			return runMultiAgentInit(&bytes.Buffer{}, []string{"../out"}, "basic", 1, "yaml", "path")
		},
		"unknown format": func() error { return runMultiAgentInit(&bytes.Buffer{}, []string{"p"}, "basic", 1, "toml", "path") },
		"unknown template": func() error {
			return runMultiAgentInit(&bytes.Buffer{}, []string{"p"}, "nonsense", 1, "yaml", "path")
		},
		"unknown routing": func() error { return runMultiAgentInit(&bytes.Buffer{}, []string{"p"}, "basic", 1, "yaml", "carrier") },
		"zero agents":     func() error { return runMultiAgentInit(&bytes.Buffer{}, []string{"p"}, "basic", 0, "yaml", "path") },
	} {
		t.Run(name, func(t *testing.T) {
			require.Error(t, run())
		})
	}

	assert.NoDirExists(t, filepath.Join(filepath.Dir(dir), "out"))
}

func TestMultiAgentInit_EveryTemplateProducesAValidConfig(t *testing.T) {
	for _, template := range []string{"basic", "microservices", "rag", "workflow"} {
		t.Run(template, func(t *testing.T) {
			dir := chdirTemp(t)
			require.NoError(t, runMultiAgentInit(&bytes.Buffer{}, []string{"fleet"}, template, 3, "yaml", "path"))
			require.NoError(t, runMultiAgentValidate(&bytes.Buffer{}, []string{
				filepath.Join(dir, "fleet", "configs", "multi-agent.yaml"),
			}, false, true))
		})
	}
}

// Regression: directory loading printed "Directory-based loading not yet
// implemented" and then returned nil, so a script checking the exit status was
// told the load had succeeded.
func TestMultiAgentLoad_DirectoryLoadingReportsNotImplemented(t *testing.T) {
	dir := t.TempDir()

	err := runMultiAgentLoad(loadCommand(t, &bytes.Buffer{}), []string{dir})

	require.Error(t, err)
	assert.ErrorIs(t, err, errNotImplemented)
}

func TestMultiAgentLoad_MissingPluginIsReported(t *testing.T) {
	err := runMultiAgentLoad(loadCommand(t, &bytes.Buffer{}), []string{filepath.Join(t.TempDir(), "absent.so")})

	require.Error(t, err)
	assert.Contains(t, err.Error(), "failed to load plugin")
}

func TestMultiAgentServe_RejectsBadInput(t *testing.T) {
	dir := t.TempDir()
	path := writeTestFile(t, dir, "multi.yaml", fullMultiAgentYAML)

	require.Error(t, runMultiAgentServe(context.Background(), &bytes.Buffer{}, []string{path}, "127.0.0.1", 0))
	require.Error(t, runMultiAgentServe(context.Background(), &bytes.Buffer{},
		[]string{filepath.Join(dir, "absent.yaml")}, "127.0.0.1", 8080))
}

// The multi-agent server used to build a MultiAgentManager and then serve an
// unrelated, empty server while advertising per-agent endpoints. It now serves
// the agents from the configuration, so a bind failure must be reported rather
// than announced as a running server.
func TestMultiAgentServe_ReportsABindFailure(t *testing.T) {
	path := writeTestFile(t, t.TempDir(), "multi.yaml", uniqueMultiAgentYAML(t))

	var out bytes.Buffer
	err := runMultiAgentServe(context.Background(), &out, []string{path}, "127.0.0.1", occupiedPort(t))

	require.Error(t, err)
	assert.Contains(t, err.Error(), "cannot bind")
	assert.Contains(t, out.String(), "Registered 2 agent(s)")
	assert.NotContains(t, out.String(), "listening")
}

// uniqueMultiAgentYAML returns a configuration whose agent IDs are unique to
// the test: agent definitions are registered in a process-wide registry, and
// re-registering the same ID fails.
func uniqueMultiAgentYAML(t *testing.T) string {
	t.Helper()

	prefix := strings.ToLower(strings.ReplaceAll(t.Name(), "/", "-"))
	return strings.NewReplacer(
		"alpha", prefix+"-alpha",
		"beta", prefix+"-beta",
	).Replace(fullMultiAgentYAML)
}

// loadCommand returns the registered "multi-agent load" command with its
// output redirected for the duration of the test.
func loadCommand(t *testing.T, out *bytes.Buffer) *cobra.Command {
	t.Helper()

	for _, sub := range multiAgentCmd.Commands() {
		if sub.Name() != "load" {
			continue
		}
		sub.SetOut(out)
		t.Cleanup(func() { sub.SetOut(nil) })
		return sub
	}
	t.Fatal("the multi-agent load command is not registered")
	return nil
}
