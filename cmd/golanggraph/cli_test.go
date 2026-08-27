// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"go/parser"
	"go/token"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/spf13/cobra"
	"github.com/spf13/pflag"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// chdirTemp moves into a fresh temporary directory for the duration of a test.
// Commands such as init and docker build write relative to the working
// directory, and tests must not write outside their own temp dir.
func chdirTemp(t *testing.T) string {
	t.Helper()

	dir := t.TempDir()
	previous, err := os.Getwd()
	require.NoError(t, err)
	require.NoError(t, os.Chdir(dir))
	t.Cleanup(func() {
		require.NoError(t, os.Chdir(previous))
	})
	return dir
}

// writeTestFile writes a fixture file below dir.
func writeTestFile(t *testing.T, dir, name, content string) string {
	t.Helper()

	path := filepath.Join(dir, name)
	require.NoError(t, os.MkdirAll(filepath.Dir(path), 0o750))
	require.NoError(t, os.WriteFile(path, []byte(content), 0o600))
	return path
}

// occupiedPort returns a port with a listener held open for the test, so a
// bind attempt against it is guaranteed to fail.
func occupiedPort(t *testing.T) int {
	t.Helper()

	listener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	t.Cleanup(func() { _ = listener.Close() })
	return listener.Addr().(*net.TCPAddr).Port
}

// freePort returns a port nothing is listening on.
func freePort(t *testing.T) int {
	t.Helper()

	listener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	port := listener.Addr().(*net.TCPAddr).Port
	require.NoError(t, listener.Close())
	return port
}

const validAgentYAML = `name: "support-agent"
type: "chat"
model: "gpt-3.5-turbo"
provider: "openai"
system_prompt: "You are a helpful assistant."
temperature: 0.5
max_tokens: 1000
tools:
  - calculator
`

// ---------------------------------------------------------------------------
// validate
// ---------------------------------------------------------------------------

// Regression: runValidate checked only that the file existed and then printed
// "Configuration validation completed successfully!" -- unparseable YAML was
// reported as a valid configuration.
func TestValidate_MalformedYAMLIsRejected(t *testing.T) {
	dir := t.TempDir()
	path := writeTestFile(t, dir, "broken.yaml", "this is not: [valid yaml at all\n  - x\n")

	var out bytes.Buffer
	err := runValidate(&out, []string{path}, false)

	require.Error(t, err, "malformed YAML must not be reported as valid")
	assert.Contains(t, err.Error(), "parsing")
	assert.NotContains(t, out.String(), "is valid")
}

func TestValidate_MissingFileIsRejected(t *testing.T) {
	var out bytes.Buffer
	err := runValidate(&out, []string{filepath.Join(t.TempDir(), "absent.yaml")}, false)

	require.Error(t, err)
	assert.NotContains(t, out.String(), "is valid")
}

func TestValidate_EmptyFileIsRejected(t *testing.T) {
	dir := t.TempDir()
	path := writeTestFile(t, dir, "empty.yaml", "   \n")

	err := runValidate(&bytes.Buffer{}, []string{path}, false)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "empty")
}

func TestValidate_UnsupportedExtensionIsRejected(t *testing.T) {
	dir := t.TempDir()
	path := writeTestFile(t, dir, "agent.txt", validAgentYAML)

	err := runValidate(&bytes.Buffer{}, []string{path}, false)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "unsupported config extension")
}

func TestValidate_AcceptsAValidConfiguration(t *testing.T) {
	dir := t.TempDir()
	path := writeTestFile(t, dir, "agent.yaml", validAgentYAML)

	var out bytes.Buffer
	require.NoError(t, runValidate(&out, []string{path}, true))
	assert.Contains(t, out.String(), "is valid")
}

func TestValidate_AcceptsJSON(t *testing.T) {
	dir := t.TempDir()
	path := writeTestFile(t, dir, "agent.json", `{
      "name": "json-agent",
      "type": "chat",
      "model": "gpt-4",
      "provider": "openai",
      "system_prompt": "hello",
      "max_tokens": 500
    }`)

	require.NoError(t, runValidate(&bytes.Buffer{}, []string{path}, true))
}

func TestValidate_MissingRequiredFieldsAreReported(t *testing.T) {
	dir := t.TempDir()
	path := writeTestFile(t, dir, "agent.yaml", "name: incomplete\ntype: chat\n")

	var out bytes.Buffer
	err := runValidate(&out, []string{path}, false)

	require.Error(t, err)
	assert.Contains(t, out.String(), "model is required")
}

// The framework's AgentConfig carries JSON tags only, so a YAML decode straight
// into it drops snake_case keys silently. A too-low max_tokens must therefore
// be seen and rejected; if the key were dropped the default of 1000 would
// quietly make this configuration "valid".
func TestValidate_SnakeCaseKeysAreActuallyRead(t *testing.T) {
	dir := t.TempDir()
	path := writeTestFile(t, dir, "agent.yaml", `name: low-tokens
type: chat
model: gpt-4
provider: openai
system_prompt: "hi"
max_tokens: 50
`)

	var out bytes.Buffer
	err := runValidate(&out, []string{path}, false)

	require.Error(t, err, "max_tokens must be read from the file, not defaulted")
	assert.Contains(t, out.String(), "MaxTokens too low")
}

// The runtime silently falls back to a chat graph for an unknown agent type,
// so validation has to catch the typo.
func TestValidate_UnknownAgentTypeIsRejected(t *testing.T) {
	dir := t.TempDir()
	path := writeTestFile(t, dir, "agent.yaml", `name: typo
type: reactt
model: gpt-4
provider: openai
max_tokens: 500
`)

	var out bytes.Buffer
	err := runValidate(&out, []string{path}, false)

	require.Error(t, err)
	assert.Contains(t, out.String(), "unknown agent type")
}

func TestValidate_StrictTurnsWarningsIntoFailures(t *testing.T) {
	dir := t.TempDir()
	path := writeTestFile(t, dir, "agent.yaml", `name: warns
type: chat
model: gpt-4
provider: openai
system_prompt: "hi"
max_tokens: 500
tools:
  - no_such_tool
`)

	var out bytes.Buffer
	require.NoError(t, runValidate(&out, []string{path}, false), "an unknown tool is only a warning")
	assert.Contains(t, out.String(), "not registered")

	out.Reset()
	err := runValidate(&out, []string{path}, true)
	require.Error(t, err, "--strict must fail on warnings")
	assert.Contains(t, err.Error(), "warning")
}

func TestValidate_MultiAgentFileChecksRoutingTargets(t *testing.T) {
	dir := t.TempDir()
	path := writeTestFile(t, dir, "multi.yaml", `name: fleet
agents:
  first:
    name: First
    type: chat
    model: gpt-4
    provider: openai
    system_prompt: "hi"
    max_tokens: 500
routing:
  default_agent: first
  rules:
    - id: rule-1
      pattern: /first
      agent_id: first
    - id: rule-2
      pattern: /ghost
      agent_id: ghost
`)

	var out bytes.Buffer
	err := runValidate(&out, []string{path}, false)

	require.Error(t, err)
	assert.Contains(t, out.String(), `rule targets agent "ghost"`)
}

func TestValidate_ToolsMayBeObjects(t *testing.T) {
	// The init template writes `tools: [{name: calculator, enabled: true}]`.
	configs, err := loadAgentConfigs(writeTestFile(t, t.TempDir(), "agent.yaml", `name: obj-tools
type: chat
model: gpt-4
provider: openai
max_tokens: 500
tools:
  - name: calculator
    enabled: true
  - name: web_search
    enabled: false
`))
	require.NoError(t, err)
	require.Len(t, configs, 1)
	assert.Equal(t, []string{"calculator"}, configs[0].Tools, "disabled tools must not be requested")
}

func TestValidate_RejectsWronglyTypedValues(t *testing.T) {
	dir := t.TempDir()
	path := writeTestFile(t, dir, "agent.yaml", "name: bad\ntype: chat\nmodel: gpt-4\nprovider: openai\ntemperature: hot\n")

	err := runValidate(&bytes.Buffer{}, []string{path}, false)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "temperature")
}

// ---------------------------------------------------------------------------
// init
// ---------------------------------------------------------------------------

func TestInit_CreatesAProjectThatBuilds(t *testing.T) {
	dir := chdirTemp(t)

	var out bytes.Buffer
	require.NoError(t, runInit(&out, []string{"demo"}, "basic", false))

	for _, name := range []string{"go.mod", "main.go", "README.md", ".gitignore", "docker-compose.yml", filepath.Join("configs", "agent-config.yaml")} {
		assert.FileExists(t, filepath.Join(dir, "demo", name), "init must actually write %s", name)
	}

	// The generated program has to be real Go, not a sketch.
	source, err := os.ReadFile(filepath.Join(dir, "demo", "main.go"))
	require.NoError(t, err)
	_, err = parser.ParseFile(token.NewFileSet(), "main.go", source, parser.AllErrors)
	require.NoError(t, err, "generated main.go must parse")
	assert.Contains(t, string(source), "package main")
	assert.Contains(t, string(source), "GoLangGraph/pkg/agent")

	gomod, err := os.ReadFile(filepath.Join(dir, "demo", "go.mod"))
	require.NoError(t, err)
	assert.Contains(t, string(gomod), "module demo")

	// And the configuration it ships with has to survive its own validator.
	require.NoError(t, runValidate(&bytes.Buffer{}, []string{filepath.Join(dir, "demo", "configs", "agent-config.yaml")}, false))
}

// Regression: "golanggraph init ../escape" created and populated a directory
// outside the working directory.
func TestInit_RejectsNamesThatEscapeTheWorkingDirectory(t *testing.T) {
	dir := chdirTemp(t)
	parent := filepath.Dir(dir)

	for _, name := range []string{"../escape", "../../escape", "/tmp/escape-abs"} {
		err := runInit(&bytes.Buffer{}, []string{name}, "basic", false)
		require.Error(t, err, "init %q must be refused", name)
		assert.NoDirExists(t, filepath.Join(parent, "escape"))
	}
	assert.NoDirExists(t, "/tmp/escape-abs")
}

// Regression: an unknown template silently produced the basic project while
// reporting the template the operator had asked for.
func TestInit_RejectsUnknownTemplate(t *testing.T) {
	dir := chdirTemp(t)

	err := runInit(&bytes.Buffer{}, []string{"demo"}, "nonsense", false)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "unknown template")
	assert.NoDirExists(t, filepath.Join(dir, "demo"))
}

func TestInit_DoesNotOverwriteAnExistingProject(t *testing.T) {
	dir := chdirTemp(t)
	require.NoError(t, runInit(&bytes.Buffer{}, []string{"demo"}, "basic", false))

	marker := writeTestFile(t, dir, filepath.Join("demo", "keep.txt"), "precious")

	err := runInit(&bytes.Buffer{}, []string{"demo"}, "basic", false)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "--force")

	content, err := os.ReadFile(marker)
	require.NoError(t, err)
	assert.Equal(t, "precious", string(content))

	require.NoError(t, runInit(&bytes.Buffer{}, []string{"demo"}, "basic", true), "--force must proceed")
}

func TestInit_TemplatesWriteTheirOwnConfigs(t *testing.T) {
	for _, tc := range []struct {
		template string
		expected []string
	}{
		{"basic", []string{"configs/agent-config.yaml"}},
		{"advanced", []string{"configs/agent-config.yaml", "configs/advanced-config.yaml"}},
		{"rag", []string{"configs/agent-config.yaml", "configs/advanced-config.yaml", "configs/rag-config.yaml"}},
	} {
		t.Run(tc.template, func(t *testing.T) {
			dir := chdirTemp(t)
			require.NoError(t, runInit(&bytes.Buffer{}, []string{"demo"}, tc.template, false))
			for _, rel := range tc.expected {
				assert.FileExists(t, filepath.Join(dir, "demo", filepath.FromSlash(rel)))
			}
		})
	}
}

func TestInit_ReportsWriteFailures(t *testing.T) {
	dir := chdirTemp(t)

	// A file where the project directory should go makes every write fail.
	writeTestFile(t, dir, "demo", "in the way")

	err := runInit(&bytes.Buffer{}, []string{"demo"}, "basic", true)
	require.Error(t, err, "a failing scaffold must not report success")
}

// ---------------------------------------------------------------------------
// debug visualize
// ---------------------------------------------------------------------------

const graphYAML = `name: my-pipeline
start_node: ingest
end_nodes: [publish]
nodes:
  - id: ingest
    name: Ingest
  - id: publish
    name: Publish
edges:
  - from: ingest
    to: publish
`

// Regression: visualize accepted a graph file argument and rendered a
// hard-coded three-node sample graph regardless of it.
func TestVisualize_RendersTheGraphFileGiven(t *testing.T) {
	dir := t.TempDir()
	path := writeTestFile(t, dir, "graph.yaml", graphYAML)

	var out bytes.Buffer
	require.NoError(t, runVisualize(&out, []string{path}, "mermaid", ""))

	rendered := out.String()
	assert.Contains(t, rendered, "ingest")
	assert.Contains(t, rendered, "publish")
	assert.NotContains(t, rendered, "process", "the built-in sample graph must not be rendered instead")
}

// Regression: the json branch produced the literal string "JSON output not
// implemented yet", wrote it to the output file and reported success.
func TestVisualize_JSONFormatProducesJSON(t *testing.T) {
	dir := t.TempDir()
	path := writeTestFile(t, dir, "graph.yaml", graphYAML)
	output := filepath.Join(dir, "topology.json")

	var out bytes.Buffer
	require.NoError(t, runVisualize(&out, []string{path}, "json", output))

	raw, err := os.ReadFile(output)
	require.NoError(t, err)
	assert.NotContains(t, string(raw), "not implemented")

	var topology struct {
		Nodes []struct {
			ID          string `json:"id"`
			IsStartNode bool   `json:"is_start_node"`
		} `json:"nodes"`
		Edges []struct {
			From string `json:"from"`
			To   string `json:"to"`
		} `json:"edges"`
	}
	require.NoError(t, json.Unmarshal(raw, &topology))
	require.Len(t, topology.Nodes, 2)
	require.Len(t, topology.Edges, 1)
	assert.Equal(t, "ingest", topology.Edges[0].From)
}

func TestVisualize_UnknownFormatFailsBeforeWriting(t *testing.T) {
	dir := t.TempDir()
	path := writeTestFile(t, dir, "graph.yaml", graphYAML)
	output := filepath.Join(dir, "out.txt")

	err := runVisualize(&bytes.Buffer{}, []string{path}, "svg", output)

	require.Error(t, err)
	assert.NoFileExists(t, output, "no file may be written for a format that cannot be produced")
}

func TestVisualize_MalformedGraphFilesAreRejected(t *testing.T) {
	dir := t.TempDir()

	for name, content := range map[string]string{
		"broken.yaml":  "nodes: [oh dear\n",
		"nonodes.yaml": "name: empty\n",
		"badedge.yaml": "nodes:\n  - id: a\nedges:\n  - from: a\n    to: ghost\n",
		"badstart.yaml": `nodes:
  - id: a
start_node: ghost
`,
	} {
		t.Run(name, func(t *testing.T) {
			path := writeTestFile(t, dir, name, content)
			require.Error(t, runVisualize(&bytes.Buffer{}, []string{path}, "mermaid", ""))
		})
	}
}

func TestVisualize_WithoutAFileSaysItIsASample(t *testing.T) {
	var out bytes.Buffer
	require.NoError(t, runVisualize(&out, nil, "mermaid", ""))
	assert.Contains(t, out.String(), "built-in sample graph")
}

// ---------------------------------------------------------------------------
// test
// ---------------------------------------------------------------------------

// Regression: runTests ignored its argument entirely, built one hard-coded
// agent and printed "All tests completed successfully".
func TestTestCommand_ChecksTheConfigurationGiven(t *testing.T) {
	dir := t.TempDir()
	good := writeTestFile(t, dir, "good.yaml", validAgentYAML)
	bad := writeTestFile(t, dir, "bad.yaml", "name: broken\ntype: chat\nprovider: openai\n")

	var out bytes.Buffer
	require.NoError(t, runTests(&out, []string{good}))
	assert.Contains(t, out.String(), "support-agent")

	require.Error(t, runTests(&bytes.Buffer{}, []string{bad}), "an invalid configuration must fail the test command")
	require.Error(t, runTests(&bytes.Buffer{}, []string{filepath.Join(dir, "absent.yaml")}))
}

func TestTestCommand_SelfTestPasses(t *testing.T) {
	var out bytes.Buffer
	require.NoError(t, runTests(&out, nil))
	assert.Contains(t, out.String(), "self-test")
}

// ---------------------------------------------------------------------------
// migrate
// ---------------------------------------------------------------------------

// Regression: the migrate flags were declared on the command but read back
// through viper, which they were never bound to, so every value arrived empty
// and "golanggraph migrate --db-type postgres" died with
// "Unsupported database type: ".
func TestMigrate_FlagsReachTheCommand(t *testing.T) {
	out, err := executeRootCommand(t, "migrate", "--db-type", "bogus", "--db-host", "db.example.com", "--db-port", "6000")

	require.Error(t, err)
	assert.Contains(t, err.Error(), `"bogus"`, "the --db-type value must reach the command")
	assert.Contains(t, out, "db.example.com:6000", "the --db-host and --db-port values must reach the command")
}

func TestMigrate_UnsupportedTypeIsRejected(t *testing.T) {
	err := runMigrations(&bytes.Buffer{}, migrateOptions{Type: "mysql", Host: "localhost", Port: 3306})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "unsupported database type")
}

func TestMigrate_MissingHostIsRejected(t *testing.T) {
	require.Error(t, runMigrations(&bytes.Buffer{}, migrateOptions{Type: "postgres", Port: 5432}))
	require.Error(t, runMigrations(&bytes.Buffer{}, migrateOptions{Type: "postgres", Host: "localhost", Port: 0}))
}

// A database that cannot be reached must fail: the operator is about to deploy
// against the schema this command claims to have created.
func TestMigrate_UnreachableDatabaseFails(t *testing.T) {
	port := freePort(t)

	t.Run("postgres", func(t *testing.T) {
		err := runMigrations(&bytes.Buffer{}, migrateOptions{
			Type: "postgres", Host: "127.0.0.1", Port: port, Database: "golanggraph", Username: "postgres",
		})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "postgres migration failed")
	})

	t.Run("redis", func(t *testing.T) {
		err := runMigrations(&bytes.Buffer{}, migrateOptions{Type: "redis", Host: "127.0.0.1", Port: port})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "redis check failed")
	})
}

// ---------------------------------------------------------------------------
// docker build / deploy
// ---------------------------------------------------------------------------

// stubRunCommand replaces the external command runner for a test and records
// the command lines it was asked to run.
func stubRunCommand(t *testing.T, err error) *[][]string {
	t.Helper()

	var recorded [][]string
	original := runCommand
	runCommand = func(ctx context.Context, out io.Writer, name string, args ...string) error {
		recorded = append(recorded, append([]string{name}, args...))
		return err
	}
	t.Cleanup(func() { runCommand = original })
	return &recorded
}

// Regression: docker build printed the command it would have run and reported
// "Docker build command prepared", so a pipeline calling it produced no image
// and no error.
func TestDockerBuild_ActuallyRunsDocker(t *testing.T) {
	dir := chdirTemp(t)
	config := writeTestFile(t, dir, "agent-config.yaml", validAgentYAML)
	recorded := stubRunCommand(t, nil)

	var out bytes.Buffer
	require.NoError(t, runDockerBuild(context.Background(), &out, []string{config}, dockerBuildOptions{
		Tag: "demo:1", Platform: "linux/amd64",
	}))

	require.Len(t, *recorded, 1, "docker must be executed")
	argv := (*recorded)[0]
	assert.Equal(t, "docker", argv[0])
	assert.Contains(t, argv, "build")
	assert.Contains(t, argv, "demo:1")
	assert.Contains(t, argv, "linux/amd64")
	assert.FileExists(t, filepath.Join(dir, "Dockerfile.agent"))
}

func TestDockerBuild_DryRunDoesNotRunDocker(t *testing.T) {
	dir := chdirTemp(t)
	config := writeTestFile(t, dir, "agent-config.yaml", validAgentYAML)
	recorded := stubRunCommand(t, nil)

	var out bytes.Buffer
	require.NoError(t, runDockerBuild(context.Background(), &out, []string{config}, dockerBuildOptions{DryRun: true}))

	assert.Empty(t, *recorded)
	assert.Contains(t, out.String(), "Dry run")
}

func TestDockerBuild_ReportsDockerFailure(t *testing.T) {
	dir := chdirTemp(t)
	config := writeTestFile(t, dir, "agent-config.yaml", validAgentYAML)
	stubRunCommand(t, fmt.Errorf("exit status 1"))

	err := runDockerBuild(context.Background(), &bytes.Buffer{}, []string{config}, dockerBuildOptions{})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "docker build failed")
}

func TestDockerBuild_DoesNotOverwriteAnExistingDockerfile(t *testing.T) {
	dir := chdirTemp(t)
	config := writeTestFile(t, dir, "agent-config.yaml", validAgentYAML)
	writeTestFile(t, dir, "Dockerfile.agent", "FROM scratch\n# hand written\n")
	stubRunCommand(t, nil)

	require.NoError(t, runDockerBuild(context.Background(), &bytes.Buffer{}, []string{config}, dockerBuildOptions{}))

	content, err := os.ReadFile(filepath.Join(dir, "Dockerfile.agent"))
	require.NoError(t, err)
	assert.Contains(t, string(content), "hand written")
}

func TestDockerBuild_RejectsAnInvalidConfiguration(t *testing.T) {
	dir := chdirTemp(t)
	broken := writeTestFile(t, dir, "broken.yaml", "nope: [\n")
	recorded := stubRunCommand(t, nil)

	require.Error(t, runDockerBuild(context.Background(), &bytes.Buffer{}, []string{broken}, dockerBuildOptions{}))
	assert.Empty(t, *recorded, "docker must not be invoked for a configuration that does not parse")
}

func TestDockerBuild_MissingCustomDockerfileIsReported(t *testing.T) {
	dir := chdirTemp(t)
	config := writeTestFile(t, dir, "agent-config.yaml", validAgentYAML)
	stubRunCommand(t, nil)

	err := runDockerBuild(context.Background(), &bytes.Buffer{}, []string{config},
		dockerBuildOptions{Dockerfile: filepath.Join(dir, "absent.Dockerfile")})
	require.Error(t, err)
}

func TestDockerBuild_DistrolessUsesItsOwnDockerfile(t *testing.T) {
	dir := chdirTemp(t)
	config := writeTestFile(t, dir, "agent-config.yaml", validAgentYAML)
	recorded := stubRunCommand(t, nil)

	require.NoError(t, runDockerBuild(context.Background(), &bytes.Buffer{}, []string{config}, dockerBuildOptions{Distroless: true}))

	assert.FileExists(t, filepath.Join(dir, "Dockerfile.distroless"))
	assert.Contains(t, (*recorded)[0], "Dockerfile.distroless")
}

// Regression: "deploy docker" printed "Docker deployment completed for config:
// X!" for any argument, including a path that did not exist, having done
// nothing at all.
func TestDeployDocker_DoesNotClaimAnUndoneDeployment(t *testing.T) {
	dir := t.TempDir()

	t.Run("missing config", func(t *testing.T) {
		var out bytes.Buffer
		err := runDeployDocker(&out, []string{filepath.Join(dir, "absent.yaml")})
		require.Error(t, err)
		assert.NotContains(t, out.String(), "completed")
	})

	t.Run("valid config", func(t *testing.T) {
		config := writeTestFile(t, dir, "agent.yaml", validAgentYAML)
		var out bytes.Buffer
		err := runDeployDocker(&out, []string{config})
		require.Error(t, err, "a deployment that did not happen must not exit zero")
		assert.ErrorIs(t, err, errNotImplemented)
		assert.NotContains(t, out.String(), "completed")
	})
}

// ---------------------------------------------------------------------------
// serve / dev
// ---------------------------------------------------------------------------

// Regression: the server was started in a goroutine whose only error handling
// was log.Fatalf while the caller had already printed "Server started on
// host:port", so a failure to bind was announced as a success.
func TestServer_ReportsABindFailure(t *testing.T) {
	port := occupiedPort(t)

	var out bytes.Buffer
	err := runServer(context.Background(), &out, serverOptions{Host: "127.0.0.1", Port: port})

	require.Error(t, err)
	assert.Contains(t, err.Error(), "cannot bind")
	assert.NotContains(t, out.String(), "listening")
}

func TestServer_RejectsAnInvalidPort(t *testing.T) {
	require.Error(t, runServer(context.Background(), &bytes.Buffer{}, serverOptions{Host: "127.0.0.1", Port: 0}))
	require.Error(t, runServer(context.Background(), &bytes.Buffer{}, serverOptions{Host: "127.0.0.1", Port: 70000}))
}

func TestServer_ServesAndShutsDownCleanly(t *testing.T) {
	port := freePort(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var out bytes.Buffer
	done := make(chan error, 1)
	go func() {
		done <- runServer(ctx, &out, serverOptions{Host: "127.0.0.1", Port: port, StaticDir: t.TempDir()})
	}()

	url := fmt.Sprintf("http://127.0.0.1:%d/api/v1/health", port)
	client := &http.Client{Timeout: time.Second}

	var reached bool
	for i := 0; i < 100 && !reached; i++ {
		resp, err := client.Get(url)
		if err == nil {
			_ = resp.Body.Close()
			reached = true
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	require.True(t, reached, "the server must answer on the port it reported")

	cancel()
	select {
	case err := <-done:
		require.NoError(t, err)
	case <-time.After(35 * time.Second):
		t.Fatal("the server did not shut down when the context was canceled")
	}
}

// Regression: the dev command declared its own --host/--port flags but
// runDevServer read host and port from viper, where only the serve command's
// flags were bound, so "golanggraph dev --port 3000" started on 8080.
func TestDev_UsesItsOwnPortFlag(t *testing.T) {
	port := occupiedPort(t)

	_, err := executeRootCommand(t, "dev", "--host", "127.0.0.1", "--port", fmt.Sprint(port))

	require.Error(t, err)
	assert.Contains(t, err.Error(), fmt.Sprintf("127.0.0.1:%d", port),
		"the dev command must bind the port its own flag names")
}

func TestDev_LoadsTheAgentConfigFlag(t *testing.T) {
	dir := t.TempDir()
	broken := writeTestFile(t, dir, "broken.yaml", "agents: [\n")

	err := runServer(context.Background(), &bytes.Buffer{}, serverOptions{
		Host: "127.0.0.1", Port: freePort(t), Dev: true, AgentConfig: broken,
	})

	require.Error(t, err, "--agent-config must be read, not ignored")
	assert.Contains(t, err.Error(), "agent config")
}

// Regression: --log-level was declared on the dev command and never read.
func TestDev_ValidatesAndAppliesTheLogLevel(t *testing.T) {
	err := runServer(context.Background(), &bytes.Buffer{}, serverOptions{
		Host: "127.0.0.1", Port: freePort(t), Dev: true, LogLevel: "shouting",
	})

	require.Error(t, err, "an unusable log level must be reported, not ignored")
	assert.Contains(t, err.Error(), "invalid log level")
}

func TestDev_SaysHotReloadIsNotImplemented(t *testing.T) {
	// Point at an occupied port so the run stops right after the notice.
	var out bytes.Buffer
	_ = runServer(context.Background(), &out, serverOptions{
		Host: "127.0.0.1", Port: occupiedPort(t), Dev: true, HotReload: true,
	})

	assert.Contains(t, out.String(), "hot-reload is not implemented")
}

// ---------------------------------------------------------------------------
// command wiring
// ---------------------------------------------------------------------------

// executeRootCommand runs the real cobra command tree with the given arguments
// and returns its output. Flag values persist on the shared command objects, so
// the flags are restored afterwards.
func executeRootCommand(t *testing.T, args ...string) (string, error) {
	t.Helper()

	var out bytes.Buffer
	rootCmd.SetOut(&out)
	rootCmd.SetErr(&out)
	rootCmd.SetArgs(args)
	t.Cleanup(func() {
		rootCmd.SetOut(nil)
		rootCmd.SetErr(nil)
		rootCmd.SetArgs(nil)
		resetFlags(rootCmd)
	})

	err := rootCmd.Execute()
	return out.String(), err
}

// resetFlags restores every flag in the tree to its default value.
func resetFlags(cmd *cobra.Command) {
	cmd.Flags().VisitAll(func(f *pflag.Flag) {
		if f.Changed {
			_ = f.Value.Set(f.DefValue)
			f.Changed = false
		}
	})
	for _, sub := range cmd.Commands() {
		resetFlags(sub)
	}
}

// Every leaf command must do something: a command with neither a Run nor
// subcommands only prints its own help.
func TestRootCommand_EveryLeafCommandIsRunnable(t *testing.T) {
	var walk func(cmd *cobra.Command, path string)
	walk = func(cmd *cobra.Command, path string) {
		if len(cmd.Commands()) == 0 {
			assert.True(t, cmd.Run != nil || cmd.RunE != nil, "%s has nothing to run", path)
			return
		}
		for _, sub := range cmd.Commands() {
			walk(sub, path+" "+sub.Name())
		}
	}
	walk(rootCmd, rootCmd.Name())
}

func TestRootCommand_KnownCommandsArePresent(t *testing.T) {
	names := map[string]bool{}
	for _, cmd := range rootCmd.Commands() {
		names[cmd.Name()] = true
	}
	for _, expected := range []string{
		"auto-serve", "debug", "deploy", "dev", "docker", "health",
		"init", "migrate", "multi-agent", "serve", "test", "validate",
	} {
		assert.True(t, names[expected], "command %q is missing", expected)
	}
}

func TestSafeProjectDir(t *testing.T) {
	for _, name := range []string{"", "   ", "..", "../x", "/abs"} {
		_, err := safeProjectDir(name)
		assert.Error(t, err, "%q must be refused", name)
	}
	for input, expected := range map[string]string{
		"demo":       "demo",
		"./demo":     "demo",
		"nested/app": filepath.Join("nested", "app"),
	} {
		got, err := safeProjectDir(input)
		require.NoError(t, err)
		assert.Equal(t, expected, got)
	}
}

func TestCheckAddressAvailable(t *testing.T) {
	require.NoError(t, checkAddressAvailable("127.0.0.1", freePort(t)))

	err := checkAddressAvailable("127.0.0.1", occupiedPort(t))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "cannot bind")
}

func TestWriteFileChecked_ReportsFailure(t *testing.T) {
	dir := t.TempDir()
	require.NoError(t, writeFileChecked(filepath.Join(dir, "ok.txt"), "content"))

	err := writeFileChecked(filepath.Join(dir, "ok.txt", "nested.txt"), "content")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "failed to write")
}

func TestNormalizeKey(t *testing.T) {
	for _, spelling := range []string{"system_prompt", "systemPrompt", "system-prompt", "SystemPrompt"} {
		assert.Equal(t, "systemprompt", normalizeKey(spelling))
	}
}

func TestLoadAgentConfigs_MultiAgentFilesAreOrdered(t *testing.T) {
	path := writeTestFile(t, t.TempDir(), "multi.yaml", `name: fleet
agents:
  zeta:
    name: Zeta
    type: chat
    model: m
    provider: ollama
  alpha:
    name: Alpha
    type: chat
    model: m
    provider: ollama
`)

	configs, err := loadAgentConfigs(path)
	require.NoError(t, err)
	require.Len(t, configs, 2)
	assert.Equal(t, "alpha", configs[0].Key, "agents must be reported in a stable order")
	assert.Equal(t, "zeta", configs[1].Key)
}

func TestLoadAgentConfigs_RejectsMalformedAgentsSection(t *testing.T) {
	dir := t.TempDir()
	for name, content := range map[string]string{
		"list.yaml":  "name: fleet\nagents:\n  - not-a-map\n",
		"empty.yaml": "name: fleet\nagents: {}\n",
		"scalar.yaml": `name: fleet
agents:
  one: "just a string"
`,
	} {
		t.Run(name, func(t *testing.T) {
			_, err := loadAgentConfigs(writeTestFile(t, dir, name, content))
			require.Error(t, err)
		})
	}
}
