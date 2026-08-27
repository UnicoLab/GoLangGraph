// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package main

import (
	"context"
	"fmt"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strings"
	"syscall"
	"time"

	"github.com/fsnotify/fsnotify"
	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"

	"github.com/UnicoLab/GoLangGraph/pkg/agent"
	"github.com/UnicoLab/GoLangGraph/pkg/server"
)

// autoServeCmd represents the auto-serve command
var autoServeCmd = &cobra.Command{
	Use:   "auto-serve [config-file-or-directory]",
	Short: "Auto-generate and serve a multi-agent system with minimal configuration",
	Long: `Auto-serve automatically discovers agent definitions and generates a complete
multi-agent system with REST endpoints, web interfaces, and schema validation.

This command embodies the GoLangGraph vision: define your agents with minimal code
and get a production-ready deployment automatically.

Features:
- Automatic endpoint generation for each agent
- Dynamic web chat interface
- Schema validation and API documentation
- Metrics and monitoring endpoints
- Development mode with an interactive playground and optional configuration reload
- Production-ready deployment

Examples:
  # Serve agents from current directory
  golanggraph auto-serve

  # Serve agents from a config file
  golanggraph auto-serve agents.yaml

  # Serve agents from a directory with custom port
  golanggraph auto-serve ./agents --port 3000

  # Enable development mode
  golanggraph auto-serve --dev

  # Deploy to production
  golanggraph auto-serve --env production --host 0.0.0.0`,
	Args: cobra.MaximumNArgs(1),
	RunE: runAutoServe,
}

func init() {
	rootCmd.AddCommand(autoServeCmd)

	// Server configuration
	autoServeCmd.Flags().StringP("host", "H", "0.0.0.0", "Host to bind to")
	autoServeCmd.Flags().IntP("port", "p", 8080, "Port to bind to")
	autoServeCmd.Flags().String("base-path", "/api", "Base path for agent endpoints")

	// Feature toggles
	autoServeCmd.Flags().Bool("web-ui", true, "Enable web chat interface")
	autoServeCmd.Flags().Bool("playground", true, "Enable API playground")
	autoServeCmd.Flags().Bool("schema-api", true, "Enable schema API endpoints")
	autoServeCmd.Flags().Bool("metrics", true, "Enable metrics endpoints")
	autoServeCmd.Flags().Bool("cors", true, "Enable CORS support")
	autoServeCmd.Flags().Bool("schema-validation", true, "Enable schema validation")

	// LLM configuration
	autoServeCmd.Flags().String("ollama-endpoint", "http://localhost:11434", "Ollama endpoint URL")
	autoServeCmd.Flags().String("openai-api-key", "", "OpenAI API key")
	autoServeCmd.Flags().String("anthropic-api-key", "", "Anthropic API key")

	// Development features
	autoServeCmd.Flags().Bool("dev", false, "Enable development mode")
	autoServeCmd.Flags().Bool("watch", false, "Watch agent configuration files and hot-reload")

	// Production features
	autoServeCmd.Flags().String("env", "development", "Environment (development, staging, production)")
	autoServeCmd.Flags().String("log-level", "info", "Log level (panic, fatal, error, warn, info, debug, trace)")
	autoServeCmd.Flags().Duration("timeout", 30*time.Second, "Request read/write timeout")
	autoServeCmd.Flags().Int64("max-request-size", 10*1024*1024, "Maximum request size in bytes")

	// Docker and deployment
	autoServeCmd.Flags().Bool("generate-dockerfile", false, "Generate Dockerfile for deployment")
	autoServeCmd.Flags().Bool("generate-docker-compose", false, "Generate docker-compose.yml")
	autoServeCmd.Flags().Bool("generate-k8s", false, "Generate Kubernetes manifests")

	// Plugin support
	autoServeCmd.Flags().StringSlice("plugins", []string{}, "Load agent plugins")
	autoServeCmd.Flags().StringSlice("agent-dirs", []string{}, "Additional agent directories")
}

// autoServeOptions is the resolved configuration of an auto-serve run.
type autoServeOptions struct {
	SourcePath            string
	Env                   string
	Dev                   bool
	Watch                 bool
	LogLevel              string
	AgentDirs             []string
	Plugins               []string
	GenerateDockerfile    bool
	GenerateDockerCompose bool
	GenerateK8s           bool
}

func runAutoServe(cmd *cobra.Command, args []string) error {
	out := cmd.OutOrStdout()

	config, opts, err := autoServeConfigFromFlags(cmd, args)
	if err != nil {
		return err
	}

	ctx, stop := signal.NotifyContext(cmd.Context(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	return runAutoServeWithContext(ctx, out, config, opts)
}

// runAutoServeWithContext runs the server and, when requested, replaces the
// whole AutoServer after a configuration change. Replacing rather than
// regenerating routes is intentional: endpoint generation mutates route and
// agent maps and is unsafe once requests may be in flight.
func runAutoServeWithContext(ctx context.Context, out io.Writer, config *server.AutoServerConfig, opts autoServeOptions) error {
	var changes <-chan struct{}
	var watchErrors <-chan error
	var stopWatching func()
	if opts.Watch {
		var err error
		changes, watchErrors, stopWatching, err = watchAutoServeSources(ctx, opts.SourcePath, opts.AgentDirs)
		if err != nil {
			return err
		}
		defer stopWatching()
		_, _ = fmt.Fprintln(out, "👀 Watching agent configuration files for changes")
	}

	printedURLs := false
	for {
		autoServer, err := prepareAutoServer(out, config, opts)
		if err != nil {
			return err
		}
		if !printedURLs {
			// Do not advertise a URL when another process already owns the
			// address. Start performs the authoritative bind immediately after
			// this preflight check.
			if err := checkAddressAvailable(config.Host, config.Port); err != nil {
				return err
			}
			printAutoServeURLs(out, config)
			printedURLs = true
		}

		serveCtx, stopServing := context.WithCancel(ctx)
		served := make(chan error, 1)
		go func() { served <- autoServer.Start(serveCtx) }()

	waitForReload:
		for {
			select {
			case err := <-served:
				stopServing()
				if err != nil {
					return fmt.Errorf("server failed: %w", err)
				}
				return nil
			case <-ctx.Done():
				stopServing()
				if err := <-served; err != nil {
					return fmt.Errorf("server failed: %w", err)
				}
				_, _ = fmt.Fprintln(out, "✅ Server stopped gracefully")
				return nil
			case err := <-watchErrors:
				if err != nil {
					_, _ = fmt.Fprintf(out, "Auto-serve watcher error: %v\n", err)
				}
			case <-changes:
				_, _ = fmt.Fprintln(out, "🔄 Agent configuration changed; restarting server")
				stopServing()
				if err := <-served; err != nil {
					return fmt.Errorf("server failed during reload: %w", err)
				}
				break waitForReload
			}
		}
	}
}

// autoServeConfigFromFlags turns the command's flags into a server
// configuration and run options.
//
// Keeping this out of the serving loop makes it testable that every declared
// flag reaches the configuration: --timeout and --max-request-size were
// declared and then never read, so the server ran with neither applied.
func autoServeConfigFromFlags(cmd *cobra.Command, args []string) (*server.AutoServerConfig, autoServeOptions, error) {
	config := &server.AutoServerConfig{
		LLMProviders: make(map[string]interface{}),
		Middleware:   []string{"cors", "logging", "recovery"},
	}
	opts := autoServeOptions{SourcePath: "."}
	if len(args) > 0 {
		opts.SourcePath = args[0]
	}

	// Flag reads are checked: a mistyped flag name used to be swallowed by
	// `value, _ := cmd.Flags().Get...` and silently produce a zero value.
	flags := cmd.Flags()
	var err error
	if config.Host, err = flags.GetString("host"); err != nil {
		return nil, opts, err
	}
	if config.Port, err = flags.GetInt("port"); err != nil {
		return nil, opts, err
	}
	if config.BasePath, err = flags.GetString("base-path"); err != nil {
		return nil, opts, err
	}
	if config.EnableWebUI, err = flags.GetBool("web-ui"); err != nil {
		return nil, opts, err
	}
	if config.EnablePlayground, err = flags.GetBool("playground"); err != nil {
		return nil, opts, err
	}
	if config.EnableSchemaAPI, err = flags.GetBool("schema-api"); err != nil {
		return nil, opts, err
	}
	if config.EnableMetricsAPI, err = flags.GetBool("metrics"); err != nil {
		return nil, opts, err
	}
	if config.EnableCORS, err = flags.GetBool("cors"); err != nil {
		return nil, opts, err
	}
	if config.SchemaValidation, err = flags.GetBool("schema-validation"); err != nil {
		return nil, opts, err
	}
	if config.OllamaEndpoint, err = flags.GetString("ollama-endpoint"); err != nil {
		return nil, opts, err
	}
	if config.ServerTimeout, err = flags.GetDuration("timeout"); err != nil {
		return nil, opts, err
	}
	if config.MaxRequestSize, err = flags.GetInt64("max-request-size"); err != nil {
		return nil, opts, err
	}

	openaiKey, err := flags.GetString("openai-api-key")
	if err != nil {
		return nil, opts, err
	}
	if openaiKey != "" {
		config.LLMProviders["openai"] = map[string]string{"api_key": openaiKey}
	}
	anthropicKey, err := flags.GetString("anthropic-api-key")
	if err != nil {
		return nil, opts, err
	}
	if anthropicKey != "" {
		config.LLMProviders["anthropic"] = map[string]string{"api_key": anthropicKey}
	}

	if opts.Env, err = flags.GetString("env"); err != nil {
		return nil, opts, err
	}
	if opts.Dev, err = flags.GetBool("dev"); err != nil {
		return nil, opts, err
	}
	if opts.Watch, err = flags.GetBool("watch"); err != nil {
		return nil, opts, err
	}
	if opts.LogLevel, err = flags.GetString("log-level"); err != nil {
		return nil, opts, err
	}
	config.LogLevel = opts.LogLevel
	if opts.AgentDirs, err = flags.GetStringSlice("agent-dirs"); err != nil {
		return nil, opts, err
	}
	if opts.Plugins, err = flags.GetStringSlice("plugins"); err != nil {
		return nil, opts, err
	}
	if opts.GenerateDockerfile, err = flags.GetBool("generate-dockerfile"); err != nil {
		return nil, opts, err
	}
	if opts.GenerateDockerCompose, err = flags.GetBool("generate-docker-compose"); err != nil {
		return nil, opts, err
	}
	if opts.GenerateK8s, err = flags.GetBool("generate-k8s"); err != nil {
		return nil, opts, err
	}

	return config, opts, nil
}

// prepareAutoServer resolves the configuration, loads the agents and generates
// any deployment files, without binding a port. Keeping this separate from the
// serving loop is what makes the command testable.
func prepareAutoServer(out io.Writer, config *server.AutoServerConfig, opts autoServeOptions) (*server.AutoServer, error) {
	_, _ = fmt.Fprintf(out, "🚀 GoLangGraph Auto-Serve\n")

	if config.Port <= 0 || config.Port > 65535 {
		return nil, fmt.Errorf("invalid port %d", config.Port)
	}
	switch opts.Env {
	case "development", "staging", "production":
	default:
		return nil, fmt.Errorf("unknown environment %q (want development, staging or production)", opts.Env)
	}

	if opts.Env == "production" {
		config.EnablePlayground = false
		_, _ = fmt.Fprintf(out, "🔒 Production mode: playground disabled\n")
	}
	if opts.Dev {
		_, _ = fmt.Fprintf(out, "🛠️  Development mode enabled\n")
	}
	if opts.LogLevel == "" {
		opts.LogLevel = logrus.InfoLevel.String()
	}
	if _, err := logrus.ParseLevel(opts.LogLevel); err != nil {
		return nil, fmt.Errorf("invalid log level %q: %w", opts.LogLevel, err)
	}
	config.LogLevel = opts.LogLevel

	// An auto-serve run must not inherit definitions left in the global
	// registry by a different server. It also means a reload starts with only
	// the definitions present in the newly-read files.
	autoServer := server.NewAutoServerWithRegistry(config, agent.NewAgentRegistry())
	registry := autoServer.Registry()
	before := len(registry.ListDefinitions())

	// A source path that does not exist used to be skipped in silence: the
	// command then served three example agents while reporting that it had
	// loaded agents from the operator's path.
	sources := append([]string{opts.SourcePath}, opts.AgentDirs...)
	for i, source := range sources {
		info, err := os.Stat(source)
		if err != nil {
			if i == 0 && opts.SourcePath == "." {
				// No explicit path given and no current directory: nothing to load.
				continue
			}
			return nil, fmt.Errorf("agent source %s: %w", source, err)
		}

		if info.IsDir() {
			loaded, err := loadAgentsFromDirectory(out, autoServer, source)
			if err != nil {
				return nil, err
			}
			_, _ = fmt.Fprintf(out, "📁 %s: %d agent config file(s) loaded\n", source, loaded)
			continue
		}

		switch ext := strings.ToLower(filepath.Ext(source)); ext {
		case ".yaml", ".yml", ".json":
			if err := autoServer.LoadAgentsFromConfig(source); err != nil {
				return nil, fmt.Errorf("failed to load agents from config: %w", err)
			}
			_, _ = fmt.Fprintf(out, "📄 %s: loaded\n", source)
		default:
			// Previously any other extension was ignored without a word.
			return nil, fmt.Errorf("unsupported agent source %s: want a directory or a .yaml, .yml or .json file", source)
		}
	}

	for _, pluginPath := range opts.Plugins {
		_, _ = fmt.Fprintf(out, "🔌 Loading plugin: %s\n", pluginPath)
		if err := registry.LoadFromPlugin(pluginPath); err != nil {
			return nil, fmt.Errorf("failed to load plugin %s: %w", pluginPath, err)
		}
	}

	definitions := registry.ListDefinitions()
	if len(definitions) == 0 {
		_, _ = fmt.Fprintf(out, "📝 No agents found; registering the built-in example agents\n")
		if err := createExampleAgents(out, autoServer); err != nil {
			return nil, err
		}
	} else {
		_, _ = fmt.Fprintf(out, "🤖 %d agent(s) registered (%d from this run)\n",
			len(definitions), len(definitions)-before)
	}

	if opts.GenerateDockerfile {
		if err := generateDockerfileForProject(opts.SourcePath); err != nil {
			return nil, fmt.Errorf("failed to generate Dockerfile: %w", err)
		}
		_, _ = fmt.Fprintf(out, "🐳 Generated Dockerfile\n")
	}
	if opts.GenerateDockerCompose {
		if err := generateDockerComposeForProject(opts.SourcePath, config); err != nil {
			return nil, fmt.Errorf("failed to generate docker-compose.yml: %w", err)
		}
		_, _ = fmt.Fprintf(out, "🐳 Generated docker-compose.yml\n")
	}
	if opts.GenerateK8s {
		if err := generateKubernetesManifests(opts.SourcePath, config); err != nil {
			return nil, fmt.Errorf("failed to generate Kubernetes manifests: %w", err)
		}
		_, _ = fmt.Fprintf(out, "☸️  Generated Kubernetes manifests\n")
	}

	return autoServer, nil
}

// loadAgentsFromDirectory registers the agents defined by the configuration
// files in a directory.
//
// Configuration files in a directory are loaded in a stable order. A malformed
// unrelated YAML file is reported but does not prevent the valid agent configs
// in that directory from being served.
func loadAgentsFromDirectory(out io.Writer, autoServer *server.AutoServer, dir string) (int, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return 0, err
	}

	names := make([]string, 0, len(entries))
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		switch strings.ToLower(filepath.Ext(entry.Name())) {
		case ".yaml", ".yml", ".json":
			names = append(names, entry.Name())
		}
	}
	sort.Strings(names)

	loaded := 0
	for _, name := range names {
		path := filepath.Join(dir, name)
		if err := autoServer.LoadAgentsFromConfig(path); err != nil {
			// Not every YAML file in a directory is an agent config; report it
			// and carry on rather than refusing to start.
			_, _ = fmt.Fprintf(out, "   ⚠️  %s: %v\n", path, err)
			continue
		}
		loaded++
	}
	return loaded, nil
}

// watchAutoServeSources reports changes to the same top-level configuration
// files that auto-serve loads. It watches directories rather than individual
// files so atomic-save editors (write a temporary file then rename it) work
// reliably. The loader is non-recursive, therefore watching nested directories
// would only create misleading reload expectations.
func watchAutoServeSources(ctx context.Context, sourcePath string, agentDirs []string) (<-chan struct{}, <-chan error, func(), error) {
	sources := append([]string{sourcePath}, agentDirs...)
	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		return nil, nil, nil, fmt.Errorf("create auto-serve watcher: %w", err)
	}

	watchedDirs := make(map[string]struct{})
	watchedFiles := make(map[string]struct{})
	for _, source := range sources {
		absPath, err := filepath.Abs(source)
		if err != nil {
			_ = watcher.Close()
			return nil, nil, nil, fmt.Errorf("resolve agent source for watch: %w", err)
		}
		absPath = filepath.Clean(absPath)
		info, err := os.Stat(absPath)
		if err != nil {
			_ = watcher.Close()
			return nil, nil, nil, fmt.Errorf("watch agent source %s: %w", source, err)
		}

		dir := absPath
		if !info.IsDir() {
			dir = filepath.Dir(absPath)
			watchedFiles[absPath] = struct{}{}
		}
		if _, exists := watchedDirs[dir]; exists {
			continue
		}
		if err := watcher.Add(dir); err != nil {
			_ = watcher.Close()
			return nil, nil, nil, fmt.Errorf("watch agent source directory %s: %w", dir, err)
		}
		watchedDirs[dir] = struct{}{}
	}

	changes := make(chan struct{}, 1)
	errors := make(chan error, 1)
	done := make(chan struct{})
	go func() {
		defer close(done)
		defer func() { _ = watcher.Close() }()

		var debounce <-chan time.Time
		var timer *time.Timer
		for {
			select {
			case <-ctx.Done():
				if timer != nil {
					timer.Stop()
				}
				return
			case event, ok := <-watcher.Events:
				if !ok {
					return
				}
				if event.Op&(fsnotify.Write|fsnotify.Create|fsnotify.Rename|fsnotify.Remove) == 0 || !isAutoServeConfigChange(event.Name, watchedDirs, watchedFiles) {
					continue
				}
				if timer == nil {
					timer = time.NewTimer(150 * time.Millisecond)
					debounce = timer.C
					continue
				}
				if !timer.Stop() {
					select {
					case <-timer.C:
					default:
					}
				}
				timer.Reset(150 * time.Millisecond)
			case <-debounce:
				timer = nil
				debounce = nil
				select {
				case changes <- struct{}{}:
				default:
				}
			case err, ok := <-watcher.Errors:
				if !ok {
					return
				}
				select {
				case errors <- err:
				default:
				}
			}
		}
	}()

	return changes, errors, func() { <-done }, nil
}

func isAutoServeConfigChange(path string, watchedDirs, watchedFiles map[string]struct{}) bool {
	path = filepath.Clean(path)
	if _, ok := watchedFiles[path]; ok {
		return true
	}
	if _, ok := watchedDirs[filepath.Dir(path)]; !ok {
		return false
	}
	switch strings.ToLower(filepath.Ext(path)) {
	case ".yaml", ".yml", ".json":
		return true
	default:
		return false
	}
}

// printAutoServeURLs prints where to reach the server.
func printAutoServeURLs(out io.Writer, config *server.AutoServerConfig) {
	host := config.Host
	if host == "0.0.0.0" || host == "" {
		host = "localhost"
	}
	baseURL := fmt.Sprintf("http://%s:%d", host, config.Port)

	_, _ = fmt.Fprintf(out, "\n🌐 URLs:\n")
	if config.EnableWebUI {
		_, _ = fmt.Fprintf(out, "   💬 Chat Interface: %s/chat\n", baseURL)
	}
	if config.EnablePlayground {
		_, _ = fmt.Fprintf(out, "   🏗️  API Playground: %s/playground\n", baseURL)
	}
	_, _ = fmt.Fprintf(out, "   📋 System Health: %s/health\n", baseURL)
	_, _ = fmt.Fprintf(out, "   🤖 List Agents: %s/agents\n", baseURL)
	if config.EnableSchemaAPI {
		_, _ = fmt.Fprintf(out, "   📄 API Schemas: %s/schemas\n", baseURL)
	}
	if config.EnableMetricsAPI {
		_, _ = fmt.Fprintf(out, "   📊 Metrics: %s/metrics\n", baseURL)
	}
	_, _ = fmt.Fprintf(out, "\n")
}

// defaultExampleModel is the model the example agents are configured with; it
// is small enough to run under a local Ollama.
const defaultExampleModel = "gemma3:1b"

// createExampleAgents registers the demonstration agents used when no agent
// definitions were found.
func createExampleAgents(out io.Writer, autoServer *server.AutoServer) error {
	examples := []struct {
		id, name, prompt string
		agentType        agent.AgentType
		tools            []string
	}{
		{"chat", "Chat Agent", "You are a helpful AI assistant. Provide clear and concise responses.", agent.AgentTypeChat, nil},
		{"react", "ReAct Agent", "You are a reasoning agent that can think and act. Break down complex problems step by step.", agent.AgentTypeReAct, []string{"calculator", "web_search"}},
		// Tool names must match the registry: "http" resolved to nothing, so
		// the example agent advertised a tool it could never call.
		{"tools", "Tool Agent", "You are a specialized agent that excels at using tools to accomplish tasks.", agent.AgentTypeTool, []string{"file_read", "file_write", "shell", "http_request"}},
	}

	for _, example := range examples {
		config := agent.DefaultAgentConfig()
		config.ID = example.id
		config.Name = example.name
		config.Type = example.agentType
		config.SystemPrompt = example.prompt
		// DefaultAgentConfig leaves the provider and model empty, and the
		// registry rejects a definition without a model. Registration therefore
		// failed for all three agents while the command still reported
		// "Created 3 example agents", leaving the server serving none of them.
		config.Provider = "ollama"
		config.Model = defaultExampleModel
		if example.tools != nil {
			config.Tools = example.tools
		}

		// Registration failures used to be printed and ignored, leaving the
		// server serving fewer agents than it announced.
		if err := autoServer.RegisterAgent(example.id, agent.NewBaseAgentDefinition(config)); err != nil {
			return fmt.Errorf("failed to register the %s example agent: %w", example.id, err)
		}
	}

	_, _ = fmt.Fprintf(out, "   ✅ Registered example agents: chat, react, tools\n")
	return nil
}

// generateDockerfileForProject generates a Dockerfile for the project
func generateDockerfileForProject(projectPath string) error {
	dockerfileContent := `# Auto-generated Dockerfile by GoLangGraph
FROM golang:1.23-alpine AS builder

WORKDIR /app
RUN apk add --no-cache git ca-certificates

COPY . .
RUN go mod tidy && go mod download
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o main .

FROM alpine:latest
RUN apk --no-cache add ca-certificates tzdata
WORKDIR /root/

COPY --from=builder /app/main .

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:8080/health || exit 1

CMD ["./main", "auto-serve", "--host", "0.0.0.0", "--port", "8080"]
`

	dockerfilePath := filepath.Join(projectPath, "Dockerfile")
	return writeFileChecked(dockerfilePath, dockerfileContent)
}

// generateDockerComposeForProject generates a docker-compose.yml
func generateDockerComposeForProject(projectPath string, config *server.AutoServerConfig) error {
	composeContent := fmt.Sprintf(`# Auto-generated docker-compose.yml by GoLangGraph
version: '3.8'

services:
  golanggraph-agents:
    build: .
    ports:
      - "%d:8080"
    environment:
      - OLLAMA_ENDPOINT=%s
    volumes:
      - ./data:/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "wget", "--no-verbose", "--tries=1", "--spider", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - golanggraph-network

  # Optional: Include Ollama service
  # ollama:
  #   image: ollama/ollama:latest
  #   ports:
  #     - "11434:11434"
  #   volumes:
  #     - ollama-data:/root/.ollama
  #   restart: unless-stopped
  #   networks:
  #     - golanggraph-network

networks:
  golanggraph-network:
    driver: bridge

volumes:
  ollama-data:
`, config.Port, config.OllamaEndpoint)

	composePath := filepath.Join(projectPath, "docker-compose.yml")
	return writeFileChecked(composePath, composeContent)
}

// generateKubernetesManifests generates Kubernetes deployment manifests
func generateKubernetesManifests(projectPath string, config *server.AutoServerConfig) error {
	manifestContent := fmt.Sprintf(`# Auto-generated Kubernetes manifests by GoLangGraph
apiVersion: apps/v1
kind: Deployment
metadata:
  name: golanggraph-agents
  labels:
    app: golanggraph-agents
spec:
  replicas: 2
  selector:
    matchLabels:
      app: golanggraph-agents
  template:
    metadata:
      labels:
        app: golanggraph-agents
    spec:
      containers:
      - name: golanggraph-agents
        image: golanggraph-agents:latest
        ports:
        - containerPort: 8080
        env:
        - name: OLLAMA_ENDPOINT
          value: "%s"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: golanggraph-agents-service
spec:
  selector:
    app: golanggraph-agents
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: LoadBalancer
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: golanggraph-agents-ingress
  annotations:
    kubernetes.io/ingress.class: nginx
spec:
  rules:
  - host: agents.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: golanggraph-agents-service
            port:
              number: 80
`, config.OllamaEndpoint)

	manifestPath := filepath.Join(projectPath, "k8s-manifests.yaml")
	return writeFileChecked(manifestPath, manifestContent)
}
