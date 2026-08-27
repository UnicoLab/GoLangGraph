// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/fsnotify/fsnotify"
	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
	yaml "gopkg.in/yaml.v3"

	"github.com/UnicoLab/GoLangGraph/pkg/agent"
	"github.com/UnicoLab/GoLangGraph/pkg/core"
	"github.com/UnicoLab/GoLangGraph/pkg/debug"
	"github.com/UnicoLab/GoLangGraph/pkg/llm"
	"github.com/UnicoLab/GoLangGraph/pkg/persistence"
	"github.com/UnicoLab/GoLangGraph/pkg/server"
	"github.com/UnicoLab/GoLangGraph/pkg/tools"
)

// errNotImplemented marks a command that cannot do what its name claims. Such a
// command must fail loudly: several commands here used to print a success
// message ("Docker deployment completed", "Deploying to Docker...") and exit 0
// without doing any work at all, which an operator would act on.
var errNotImplemented = errors.New("not implemented")

var (
	cfgFile string
	verbose bool
)

// synchronizedWriter makes command output safe when a command's watcher and
// serving goroutine both report progress. CLI callers commonly use bytes.Buffer
// in tests, but the wrapper also protects any non-concurrent io.Writer supplied
// by an embedding application.
type synchronizedWriter struct {
	mu sync.Mutex
	w  io.Writer
}

func (w *synchronizedWriter) Write(p []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.w.Write(p)
}

func synchronizeWriter(w io.Writer) io.Writer {
	if _, ok := w.(*synchronizedWriter); ok {
		return w
	}
	return &synchronizedWriter{w: w}
}

// rootCmd represents the base command when called without any subcommands
var rootCmd = &cobra.Command{
	Use:   "golanggraph",
	Short: "GoLangGraph - A Go implementation of LangGraph",
	Long: `GoLangGraph is a comprehensive Go implementation of the LangGraph framework
for building stateful, multi-agent conversational AI applications.

This CLI provides tools for:
- Building and packaging agents into Docker containers
- Running a development server
- Managing database migrations
- Visualizing graph execution
- Validating and testing agent configurations
- Generating deployment artifacts`,
}

// serveCmd represents the serve command
var serveCmd = &cobra.Command{
	Use:   "serve",
	Short: "Start the GoLangGraph server",
	Long: `Start the GoLangGraph HTTP server with REST API endpoints and WebSocket support.
The server provides:
- REST API for agent and graph management
- WebSocket endpoints for real-time streaming
- Visual debugging interface
- Health monitoring endpoints`,
	RunE: func(cmd *cobra.Command, args []string) error {
		ctx, stop := signal.NotifyContext(cmd.Context(), syscall.SIGINT, syscall.SIGTERM)
		defer stop()
		agentConfig, err := cmd.Flags().GetString("agent-config")
		if err != nil {
			return err
		}
		return runServer(ctx, cmd.OutOrStdout(), serverOptions{
			Host:        viper.GetString("host"),
			Port:        viper.GetInt("port"),
			StaticDir:   viper.GetString("static-dir"),
			CORS:        viper.GetBool("enable-cors"),
			AgentConfig: agentConfig,
		})
	},
}

// migrateCmd represents the migrate command
var migrateCmd = &cobra.Command{
	Use:   "migrate",
	Short: "Run database migrations",
	Long: `Run database migrations to set up the required schema for state persistence.

For postgres this creates the threads, checkpoints, sessions and document tables
if they do not exist. Redis has no schema; for redis this only verifies that the
server is reachable.`,
	RunE: func(cmd *cobra.Command, args []string) error {
		opts := migrateOptions{}
		var err error
		// Defect: these flags were declared on this command but the values were
		// read back through viper, which they were never bound to. Every value
		// came back empty, so "golanggraph migrate --db-host db.example.com"
		// ignored the host entirely and "--db-type postgres" reached the switch
		// as "" and died with `Unsupported database type: `. Read the flags.
		if opts.Type, err = cmd.Flags().GetString("db-type"); err != nil {
			return err
		}
		if opts.Host, err = cmd.Flags().GetString("db-host"); err != nil {
			return err
		}
		if opts.Port, err = cmd.Flags().GetInt("db-port"); err != nil {
			return err
		}
		if opts.Database, err = cmd.Flags().GetString("db-name"); err != nil {
			return err
		}
		if opts.Username, err = cmd.Flags().GetString("db-user"); err != nil {
			return err
		}
		if opts.Password, err = cmd.Flags().GetString("db-password"); err != nil {
			return err
		}
		if opts.SSLMode, err = cmd.Flags().GetString("db-sslmode"); err != nil {
			return err
		}
		return runMigrations(cmd.OutOrStdout(), opts)
	},
}

// debugCmd represents the debug command
var debugCmd = &cobra.Command{
	Use:   "debug",
	Short: "Debug tools for graph visualization and analysis",
	Long:  `Provides debugging tools for analyzing graph execution and visualizing agent behavior.`,
}

// visualizeCmd represents the visualize command
var visualizeCmd = &cobra.Command{
	Use:   "visualize [graph-file]",
	Short: "Visualize a graph structure",
	Long: `Generate visual representations of graph structures in various formats (mermaid, dot, json).

The graph file is a JSON or YAML document describing the graph:

  name: my-graph
  start_node: start
  end_nodes: [finish]
  nodes:
    - id: start
      name: Start
    - id: finish
      name: Finish
  edges:
    - from: start
      to: finish

Without a graph file a built-in sample graph is rendered, and the output says so.`,
	Args: cobra.MaximumNArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		format, err := cmd.Flags().GetString("format")
		if err != nil {
			return err
		}
		output, err := cmd.Flags().GetString("output")
		if err != nil {
			return err
		}
		return runVisualize(cmd.OutOrStdout(), args, format, output)
	},
}

// testCmd represents the test command
var testCmd = &cobra.Command{
	Use:   "test [config-file]",
	Short: "Test agent configurations and graph execution",
	Long: `Test an agent configuration by building the agent it describes and validating
the execution graph that results.

With a configuration file the agent in that file is built and checked. Without
one, a built-in self-test exercises agent construction and graph validation so
that a broken installation is detected. No LLM calls are made either way.`,
	Args: cobra.MaximumNArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		return runTests(cmd.OutOrStdout(), args)
	},
}

// healthCmd represents the health command
var healthCmd = &cobra.Command{
	Use:   "health",
	Short: "Check system health and component status",
	Long: `Check the health status of GoLangGraph components.

With --server, probes a running server's HTTP health endpoint; this is what a
container health check should use. Otherwise it probes the dependencies that
are actually configured (POSTGRES_HOST, REDIS_HOST, OLLAMA_URL) and the local
disk and memory.

Missing optional provider credentials are reported as warnings and do not fail
the check unless --strict is given.`,
	Run: func(cmd *cobra.Command, args []string) {
		serverURL, _ := cmd.Flags().GetString("server")
		strict, _ := cmd.Flags().GetBool("strict")
		timeout, _ := cmd.Flags().GetDuration("timeout")

		if serverURL == "" {
			serverURL = os.Getenv("GOLANGGRAPH_SERVER_URL")
		}

		os.Exit(runHealthCheck(healthOptions{
			ServerURL: serverURL,
			Strict:    strict,
			Timeout:   timeout,
		}))
	},
}

// buildCmd represents the build command.
//
// It has no subcommands of its own: invoking it used to print the help text and
// exit 0, which reads as "the build succeeded". Point at the command that does
// the work and fail.
var buildCmd = &cobra.Command{
	Use:   "build",
	Short: "Build and package agents for deployment",
	Long: `Build and package agents into deployable artifacts including Docker containers.

Container images are built by "golanggraph docker build", which supports both
regular and distroless variants.`,
	RunE: func(cmd *cobra.Command, args []string) error {
		return fmt.Errorf("nothing to build here: use 'golanggraph docker build [agent-config]'")
	},
}

// dockerCmd represents the docker command
var dockerCmd = &cobra.Command{
	Use:   "docker",
	Short: "Docker packaging commands",
	Long:  `Commands for packaging agents into Docker containers for production deployment.`,
}

// dockerBuildCmd represents the docker build command
var dockerBuildCmd = &cobra.Command{
	Use:   "build [agent-config]",
	Short: "Build Docker container for agent",
	Long: `Build a Docker container for deploying an agent to production.
Supports both regular and distroless variants for different deployment needs.`,
	Args: cobra.MaximumNArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		opts := dockerBuildOptions{}
		var err error
		if opts.Distroless, err = cmd.Flags().GetBool("distroless"); err != nil {
			return err
		}
		if opts.Tag, err = cmd.Flags().GetString("tag"); err != nil {
			return err
		}
		if opts.Dockerfile, err = cmd.Flags().GetString("dockerfile"); err != nil {
			return err
		}
		if opts.Platform, err = cmd.Flags().GetString("platform"); err != nil {
			return err
		}
		if opts.DryRun, err = cmd.Flags().GetBool("dry-run"); err != nil {
			return err
		}
		if opts.ContextDir, err = cmd.Flags().GetString("context"); err != nil {
			return err
		}
		return runDockerBuild(cmd.Context(), cmd.OutOrStdout(), args, opts)
	},
}

// devCmd represents the dev command
var devCmd = &cobra.Command{
	Use:   "dev",
	Short: "Start a development server",
	Long: `Start a development server for testing and debugging agents.

Includes an interactive debugging interface and an agent playground. With
--hot-reload and --agent-config, the server atomically replaces configured
agents whenever that configuration file changes.`,
	RunE: func(cmd *cobra.Command, args []string) error {
		ctx, stop := signal.NotifyContext(cmd.Context(), syscall.SIGINT, syscall.SIGTERM)
		defer stop()

		// Defect: dev declares its own --host/--port/--hot-reload/--log-level
		// flags, but runDevServer read host and port from viper, where only the
		// *serve* command's flags are bound. "golanggraph dev --port 3000"
		// therefore started on 8080. Read this command's own flags.
		opts := serverOptions{Dev: true, CORS: true, StaticDir: "./static"}
		var err error
		if opts.Host, err = cmd.Flags().GetString("host"); err != nil {
			return err
		}
		if opts.Port, err = cmd.Flags().GetInt("port"); err != nil {
			return err
		}
		if opts.AgentConfig, err = cmd.Flags().GetString("agent-config"); err != nil {
			return err
		}
		if opts.HotReload, err = cmd.Flags().GetBool("hot-reload"); err != nil {
			return err
		}
		if opts.LogLevel, err = cmd.Flags().GetString("log-level"); err != nil {
			return err
		}
		debug, err := cmd.Flags().GetBool("debug")
		if err != nil {
			return err
		}
		opts.Dev = debug
		return runServer(ctx, cmd.OutOrStdout(), opts)
	},
}

// validateCmd represents the validate command
var validateCmd = &cobra.Command{
	Use:   "validate [config-file]",
	Short: "Validate agent configuration",
	Long: `Validate agent configuration files and graph definitions for correctness.

Both single-agent files and multi-agent files (a top-level "agents:" map) are
understood, in YAML or JSON. The file is parsed, required fields are checked,
value ranges are checked, tool names are resolved against the built-in tool
registry and the resulting agent graph is built and validated.

With --strict, warnings (unknown keys, missing system prompt, unknown provider)
are treated as errors.`,
	Args: cobra.MaximumNArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		strict, err := cmd.Flags().GetBool("strict")
		if err != nil {
			return err
		}
		return runValidate(cmd.OutOrStdout(), args, strict)
	},
}

// deployCmd represents the deploy command
var deployCmd = &cobra.Command{
	Use:   "deploy",
	Short: "Deploy agents to production",
	Long:  `Deploy agents to various production environments including Docker, Kubernetes, and cloud platforms.`,
}

// deployDockerCmd represents the deploy docker command
var deployDockerCmd = &cobra.Command{
	Use:   "docker [agent-config]",
	Short: "Deploy agent using Docker",
	Long: `Build and run an agent Docker container with production-ready defaults.

The command validates the agent configuration, builds the image, then starts a
named container publishing its internal port 8080. Use --dry-run to inspect the
build and run commands without changing Docker state.`,
	Args: cobra.MaximumNArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		opts := deployDockerOptions{}
		var err error
		if opts.Tag, err = cmd.Flags().GetString("tag"); err != nil {
			return err
		}
		if opts.ContainerName, err = cmd.Flags().GetString("name"); err != nil {
			return err
		}
		if opts.Port, err = cmd.Flags().GetInt("port"); err != nil {
			return err
		}
		if opts.ContextDir, err = cmd.Flags().GetString("context"); err != nil {
			return err
		}
		if opts.DryRun, err = cmd.Flags().GetBool("dry-run"); err != nil {
			return err
		}
		return runDeployDockerWithOptions(cmd.Context(), cmd.OutOrStdout(), args, opts)
	},
}

// initCmd represents the init command
var initCmd = &cobra.Command{
	Use:   "init [project-name]",
	Short: "Initialize a new GoLangGraph project",
	Long: `Initialize a new GoLangGraph project with example configurations and templates.

The project name is used as a directory name below the current directory; names
that escape it (absolute paths, "..") are rejected. An existing non-empty
directory is not overwritten unless --force is given.`,
	Args: cobra.MaximumNArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		template, err := cmd.Flags().GetString("template")
		if err != nil {
			return err
		}
		force, err := cmd.Flags().GetBool("force")
		if err != nil {
			return err
		}
		return runInit(cmd.OutOrStdout(), args, template, force)
	},
}

func init() {
	cobra.OnInitialize(initConfig)

	// A command that fails at runtime should report the failure, not bury it
	// under a page of usage text, and main() prints the error itself.
	rootCmd.SilenceUsage = true
	rootCmd.SilenceErrors = true

	// Global flags
	rootCmd.PersistentFlags().StringVar(&cfgFile, "config", "", "config file (default is $HOME/.golanggraph.yaml)")
	rootCmd.PersistentFlags().BoolVarP(&verbose, "verbose", "v", false, "verbose output")

	// Serve command flags
	serveCmd.Flags().StringP("host", "H", "0.0.0.0", "Host to bind to")
	serveCmd.Flags().IntP("port", "p", 8080, "Port to bind to")
	serveCmd.Flags().String("static-dir", "./static", "Static files directory")
	serveCmd.Flags().Bool("enable-cors", true, "Enable CORS")
	serveCmd.Flags().String("agent-config", "", "Agent configuration file to load before serving")

	// Dev command flags
	devCmd.Flags().StringP("host", "H", "localhost", "Host to bind to")
	devCmd.Flags().IntP("port", "p", 8080, "Port to bind to")
	devCmd.Flags().String("agent-config", "", "Agent configuration file")
	devCmd.Flags().Bool("hot-reload", false, "Reload --agent-config when its file changes")
	devCmd.Flags().Bool("debug", true, "Enable the server's development mode (debug interface and playground)")
	devCmd.Flags().String("log-level", "info", "Log level (debug, info, warn, error)")

	// Docker build command flags
	dockerBuildCmd.Flags().BoolP("distroless", "d", false, "Build distroless container")
	dockerBuildCmd.Flags().StringP("tag", "t", "", "Docker image tag")
	dockerBuildCmd.Flags().String("dockerfile", "", "Custom Dockerfile path")
	dockerBuildCmd.Flags().String("platform", "", "Target platform (e.g., linux/amd64,linux/arm64)")
	dockerBuildCmd.Flags().Bool("dry-run", false, "Print the docker command without running it")
	dockerBuildCmd.Flags().String("context", ".", "Docker build context directory")

	// Docker deployment flags
	deployDockerCmd.Flags().StringP("tag", "t", "", "Docker image tag")
	deployDockerCmd.Flags().String("name", "golanggraph-agent", "Name for the deployed container")
	deployDockerCmd.Flags().IntP("port", "p", 8080, "Host port to publish for the agent")
	deployDockerCmd.Flags().String("context", ".", "Docker build context directory")
	deployDockerCmd.Flags().Bool("dry-run", false, "Print Docker commands without building or running")

	// Validate command flags
	validateCmd.Flags().BoolP("strict", "s", false, "Enable strict validation")

	// Init command flags
	initCmd.Flags().StringP("template", "t", "basic", "Project template (basic, advanced, rag)")
	initCmd.Flags().Bool("force", false, "Overwrite an existing project directory")

	// Migrate command flags
	migrateCmd.Flags().String("db-type", "postgres", "Database type (postgres, redis)")
	migrateCmd.Flags().String("db-host", "localhost", "Database host")
	migrateCmd.Flags().Int("db-port", 5432, "Database port")
	migrateCmd.Flags().String("db-name", "golanggraph", "Database name")
	migrateCmd.Flags().String("db-user", "postgres", "Database user")
	migrateCmd.Flags().String("db-password", "", "Database password")
	migrateCmd.Flags().String("db-sslmode", "disable", "PostgreSQL sslmode (disable, require, verify-full)")

	// Visualize command flags
	visualizeCmd.Flags().StringP("format", "f", "mermaid", "Output format (mermaid, dot, json)")
	visualizeCmd.Flags().StringP("output", "o", "", "Output file (default: stdout)")

	// Add subcommands
	rootCmd.AddCommand(initCmd)
	rootCmd.AddCommand(buildCmd)
	rootCmd.AddCommand(dockerCmd)
	rootCmd.AddCommand(deployCmd)
	rootCmd.AddCommand(devCmd)
	rootCmd.AddCommand(validateCmd)
	rootCmd.AddCommand(serveCmd)
	rootCmd.AddCommand(migrateCmd)
	rootCmd.AddCommand(debugCmd)
	rootCmd.AddCommand(testCmd)
	healthCmd.Flags().String("server", "", "probe a running server's health endpoint instead of local dependencies")
	healthCmd.Flags().Bool("strict", false, "treat warnings as failures")
	healthCmd.Flags().Duration("timeout", 3*time.Second, "per-probe timeout")
	rootCmd.AddCommand(healthCmd)

	// Add nested commands
	dockerCmd.AddCommand(dockerBuildCmd)
	deployCmd.AddCommand(deployDockerCmd)
	debugCmd.AddCommand(visualizeCmd)

	// Bind flags to viper
	_ = viper.BindPFlag("host", serveCmd.Flags().Lookup("host"))
	_ = viper.BindPFlag("port", serveCmd.Flags().Lookup("port"))
	_ = viper.BindPFlag("static-dir", serveCmd.Flags().Lookup("static-dir"))
	_ = viper.BindPFlag("enable-cors", serveCmd.Flags().Lookup("enable-cors"))
}

// initConfig reads in config file and ENV variables.
func initConfig() {
	if cfgFile != "" {
		// Use config file from the flag.
		viper.SetConfigFile(cfgFile)
	} else {
		// Find home directory.
		home, err := os.UserHomeDir()
		cobra.CheckErr(err)

		// Search config in home directory with name ".golanggraph" (without extension).
		viper.AddConfigPath(home)
		viper.AddConfigPath(".")
		viper.SetConfigType("yaml")
		viper.SetConfigName(".golanggraph")
	}

	viper.AutomaticEnv() // read in environment variables that match

	// If a config file is found, read it in.
	if err := viper.ReadInConfig(); err == nil && verbose {
		_, _ = fmt.Fprintln(os.Stderr, "Using config file:", viper.ConfigFileUsed())
	}
}

// serverOptions describes a serve or dev run.
type serverOptions struct {
	Host      string
	Port      int
	StaticDir string
	CORS      bool
	// Dev enables the development mode of the server.
	Dev bool
	// AgentConfig is an optional agent configuration file whose agents are
	// created on the server before it starts serving.
	AgentConfig string
	// HotReload watches --agent-config and atomically replaces its agents when
	// its file changes.
	HotReload bool
	// LogLevel is applied to the server's logger.
	LogLevel string
}

// runServer starts the HTTP server and blocks until ctx is canceled.
//
// Two defects are fixed here. The server used to be started in a goroutine
// whose only error handling was log.Fatalf, while the caller had already
// printed "Server started on host:port" -- so a failure to bind was announced
// as a success. And the dev command's own flags were ignored (see devCmd).
func runServer(ctx context.Context, out io.Writer, opts serverOptions) error {
	out = synchronizeWriter(out)
	if opts.Port <= 0 || opts.Port > 65535 {
		return fmt.Errorf("invalid port %d", opts.Port)
	}
	if opts.Host == "" {
		opts.Host = "0.0.0.0"
	}
	if opts.StaticDir == "" {
		opts.StaticDir = "./static"
	}

	// --log-level was declared on the dev command and never read.
	if opts.LogLevel != "" {
		if _, err := logrus.ParseLevel(opts.LogLevel); err != nil {
			return fmt.Errorf("invalid log level %q (want debug, info, warn or error)", opts.LogLevel)
		}
	}

	config := &server.ServerConfig{
		Host:           opts.Host,
		Port:           opts.Port,
		ReadTimeout:    30 * time.Second,
		WriteTimeout:   30 * time.Second,
		MaxHeaderBytes: 1 << 20,
		EnableCORS:     opts.CORS,
		StaticDir:      opts.StaticDir,
		DevMode:        opts.Dev,
		LogLevel:       opts.LogLevel,
	}

	if opts.Dev {
		_, _ = fmt.Fprintln(out, "Starting GoLangGraph development server...")
	} else {
		_, _ = fmt.Fprintln(out, "Starting GoLangGraph server...")
	}

	srv := server.NewServer(config)

	agentManager, err := initializeComponents(out, srv)
	if err != nil {
		return fmt.Errorf("failed to initialize components: %w", err)
	}

	// --agent-config used to be declared and never read. Load it for real.
	if opts.AgentConfig != "" {
		if err := replaceAgentsFromConfig(out, agentManager, opts.AgentConfig); err != nil {
			return err
		}
	}

	// Fail before announcing success if the address cannot be bound.
	if err := checkAddressAvailable(opts.Host, opts.Port); err != nil {
		return err
	}

	watchCtx, stopWatching := context.WithCancel(ctx)
	var waitForWatcher func()
	stopWatcher := func() {
		stopWatching()
		if waitForWatcher != nil {
			waitForWatcher()
		}
	}
	defer stopWatcher()
	if opts.HotReload {
		if opts.AgentConfig == "" {
			return fmt.Errorf("--hot-reload requires --agent-config")
		}
		wait, err := watchAgentConfig(watchCtx, out, agentManager, opts.AgentConfig)
		if err != nil {
			return err
		}
		waitForWatcher = wait
	}

	errCh := make(chan error, 1)
	go func() { errCh <- srv.Start() }()

	_, _ = fmt.Fprintf(out, "Server listening on %s:%d\n", config.Host, config.Port)
	_, _ = fmt.Fprintf(out, "Health check: http://%s:%d/api/v1/health\n", config.Host, config.Port)
	if opts.Dev {
		_, _ = fmt.Fprintf(out, "Debug interface: http://%s:%d/debug\n", config.Host, config.Port)
		_, _ = fmt.Fprintf(out, "Agent playground: http://%s:%d/playground\n", config.Host, config.Port)
	}

	select {
	case err := <-errCh:
		if err != nil {
			return fmt.Errorf("server failed to start: %w", err)
		}
		return nil
	case <-ctx.Done():
		// The watcher writes progress messages through the same writer as the
		// server. Join it before printing shutdown output so callers that use
		// a non-concurrent writer (including bytes.Buffer in tests) stay safe.
		stopWatcher()
	}

	_, _ = fmt.Fprintln(out, "Shutting down server...")

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := srv.Stop(shutdownCtx); err != nil {
		return fmt.Errorf("server forced to shutdown: %w", err)
	}

	_, _ = fmt.Fprintln(out, "Server exited")
	return nil
}

// replaceAgentsFromConfig parses a complete replacement set before atomically
// swapping it into the manager. It is shared by the initial load and reload
// path so both reject the same malformed configuration.
func replaceAgentsFromConfig(out io.Writer, manager *server.AgentManager, configPath string) error {
	configs, err := loadAgentConfigs(configPath)
	if err != nil {
		return fmt.Errorf("agent config %s: %w", configPath, err)
	}

	agentConfigs := make([]*agent.AgentConfig, 0, len(configs))
	for _, cfg := range configs {
		agentConfigs = append(agentConfigs, cfg.toAgentConfig())
	}
	if err := manager.ReplaceAgents(agentConfigs); err != nil {
		return fmt.Errorf("agent config %s: %w", configPath, err)
	}
	for _, cfg := range configs {
		_, _ = fmt.Fprintf(out, "Loaded agent %s (%s)\n", cfg.Name, cfg.Type)
	}
	return nil
}

// watchAgentConfig watches both the configuration file's directory and its
// filename. Watching the directory is important because editors commonly save
// by replacing a temporary file, which invalidates a file-only watch.
func watchAgentConfig(ctx context.Context, out io.Writer, manager *server.AgentManager, configPath string) (func(), error) {
	absPath, err := filepath.Abs(configPath)
	if err != nil {
		return nil, fmt.Errorf("resolve agent config for hot-reload: %w", err)
	}
	absPath = filepath.Clean(absPath)

	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		return nil, fmt.Errorf("create hot-reload watcher: %w", err)
	}
	if err := watcher.Add(filepath.Dir(absPath)); err != nil {
		_ = watcher.Close()
		return nil, fmt.Errorf("watch agent config directory: %w", err)
	}

	_, _ = fmt.Fprintf(out, "Watching agent config for changes: %s\n", absPath)
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
				if filepath.Clean(event.Name) != absPath || event.Op&(fsnotify.Write|fsnotify.Create|fsnotify.Rename) == 0 {
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
				if err := replaceAgentsFromConfig(out, manager, absPath); err != nil {
					_, _ = fmt.Fprintf(out, "Hot-reload failed; keeping existing agents: %v\n", err)
					continue
				}
				_, _ = fmt.Fprintln(out, "Reloaded agent configuration")
			case err, ok := <-watcher.Errors:
				if !ok {
					return
				}
				_, _ = fmt.Fprintf(out, "Hot-reload watcher error: %v\n", err)
			}
		}
	}()
	return func() { <-done }, nil
}

// checkAddressAvailable reports whether the server can bind host:port. Without
// this the CLI announced a listening server whose bind had already failed.
func checkAddressAvailable(host string, port int) error {
	listener, err := net.Listen("tcp", net.JoinHostPort(host, fmt.Sprintf("%d", port)))
	if err != nil {
		return fmt.Errorf("cannot bind %s:%d: %w", host, port, err)
	}
	return listener.Close()
}

// initializeComponents wires the LLM providers, tools and managers onto the
// server and returns the agent manager it installed.
//
// Provider construction errors used to be discarded with `if err == nil`, so a
// misconfigured provider silently disappeared, and the built-in tools were
// re-registered on a registry that already contains them, discarding the
// "already registered" errors that came back.
func initializeComponents(out io.Writer, srv *server.Server) (*server.AgentManager, error) {
	llmManager := llm.NewProviderManager()

	if apiKey := os.Getenv("OPENAI_API_KEY"); apiKey != "" {
		openaiProvider, err := llm.NewOpenAIProvider(&llm.ProviderConfig{
			APIKey:   apiKey,
			Endpoint: "https://api.openai.com/v1",
		})
		if err != nil {
			return nil, fmt.Errorf("openai provider: %w", err)
		}
		if err := llmManager.RegisterProvider("openai", openaiProvider); err != nil {
			return nil, fmt.Errorf("register openai provider: %w", err)
		}
	}

	ollamaURL := os.Getenv("OLLAMA_URL")
	if ollamaURL == "" {
		ollamaURL = "http://localhost:11434"
	}
	ollamaProvider, err := llm.NewOllamaProvider(&llm.ProviderConfig{Endpoint: ollamaURL})
	if err != nil {
		return nil, fmt.Errorf("ollama provider: %w", err)
	}
	if err := llmManager.RegisterProvider("ollama", ollamaProvider); err != nil {
		return nil, fmt.Errorf("register ollama provider: %w", err)
	}

	// NewToolRegistry already registers the built-in tools.
	toolRegistry := tools.NewToolRegistry()

	sessionManager := persistence.NewSessionManager(nil)
	agentManager := server.NewAgentManager(llmManager, toolRegistry)

	srv.SetLLMManager(llmManager)
	srv.SetToolRegistry(toolRegistry)
	srv.SetAgentManager(agentManager)
	srv.SetSessionManager(sessionManager)

	_, _ = fmt.Fprintf(out, "Providers: %s | Tools: %d\n",
		strings.Join(llmManager.ListProviders(), ", "), len(toolRegistry.ListTools()))

	return agentManager, nil
}

// migrateOptions describes a migrate run.
type migrateOptions struct {
	Type     string
	Host     string
	Port     int
	Database string
	Username string
	Password string
	SSLMode  string
}

func runMigrations(out io.Writer, opts migrateOptions) error {
	if opts.Host == "" {
		return errors.New("database host is required (--db-host)")
	}
	if opts.Port <= 0 {
		return fmt.Errorf("invalid database port %d", opts.Port)
	}

	_, _ = fmt.Fprintf(out, "Running database migrations against %s %s:%d...\n", opts.Type, opts.Host, opts.Port)

	switch opts.Type {
	case "postgres", "postgresql", "pgvector":
		sslMode := opts.SSLMode
		if sslMode == "" {
			sslMode = "disable"
		}
		config := &persistence.DatabaseConfig{
			Type:     persistence.DatabaseType(opts.Type),
			Host:     opts.Host,
			Port:     opts.Port,
			Database: opts.Database,
			Username: opts.Username,
			Password: opts.Password,
			SSLMode:  sslMode,
			// Without a connect timeout a wrong host hangs for minutes.
			ConnectionParams: map[string]string{"connect_timeout": "10"},
		}

		// NewPostgresCheckpointer runs the schema migration (CREATE TABLE IF
		// NOT EXISTS ...) as part of construction.
		checkpointer, err := persistence.NewPostgresCheckpointer(config)
		if err != nil {
			return fmt.Errorf("postgres migration failed: %w", err)
		}
		defer func() {
			if cerr := checkpointer.Close(); cerr != nil {
				_, _ = fmt.Fprintf(out, "warning: failed to close database connection: %v\n", cerr)
			}
		}()

		_, _ = fmt.Fprintln(out, "PostgreSQL schema is up to date (threads, checkpoints, sessions, documents)")
		return nil

	case "redis":
		config := &persistence.DatabaseConfig{
			Type:     persistence.DatabaseTypeRedis,
			Host:     opts.Host,
			Port:     opts.Port,
			Password: opts.Password,
		}

		checkpointer, err := persistence.NewRedisCheckpointer(config)
		if err != nil {
			return fmt.Errorf("redis check failed: %w", err)
		}
		defer func() {
			if cerr := checkpointer.Close(); cerr != nil {
				_, _ = fmt.Fprintf(out, "warning: failed to close redis connection: %v\n", cerr)
			}
		}()

		// Redis is schemaless: there is nothing to migrate. Saying "Redis setup
		// completed successfully" implied work that never happened.
		_, _ = fmt.Fprintln(out, "Redis has no schema to migrate; the server is reachable")
		return nil

	default:
		return fmt.Errorf("unsupported database type: %q (want postgres, pgvector or redis)", opts.Type)
	}
}

// graphFile is the on-disk description of a graph to visualize.
type graphFile struct {
	Name      string          `json:"name" yaml:"name"`
	StartNode string          `json:"start_node" yaml:"start_node"`
	EndNodes  []string        `json:"end_nodes" yaml:"end_nodes"`
	Nodes     []graphFileNode `json:"nodes" yaml:"nodes"`
	Edges     []graphFileEdge `json:"edges" yaml:"edges"`
}

type graphFileNode struct {
	ID   string `json:"id" yaml:"id"`
	Name string `json:"name" yaml:"name"`
}

type graphFileEdge struct {
	From string `json:"from" yaml:"from"`
	To   string `json:"to" yaml:"to"`
}

// loadGraphFromFile builds a graph from a JSON or YAML description.
//
// The visualize command used to accept a graph file argument and then render a
// hard-coded three-node sample graph regardless of it, so an operator inspected
// a diagram that had nothing to do with their graph.
func loadGraphFromFile(path string) (*core.Graph, error) {
	data, err := os.ReadFile(path) // #nosec G304 -- operator-supplied path is the point of the command
	if err != nil {
		return nil, err
	}

	var spec graphFile
	switch ext := strings.ToLower(filepath.Ext(path)); ext {
	case ".json":
		if err := json.Unmarshal(data, &spec); err != nil {
			return nil, fmt.Errorf("parsing %s: %w", path, err)
		}
	case ".yaml", ".yml":
		if err := yaml.Unmarshal(data, &spec); err != nil {
			return nil, fmt.Errorf("parsing %s: %w", path, err)
		}
	default:
		return nil, fmt.Errorf("unsupported graph file extension %q (want .json, .yaml or .yml)", ext)
	}

	if len(spec.Nodes) == 0 {
		return nil, fmt.Errorf("%s defines no nodes", path)
	}

	name := spec.Name
	if name == "" {
		name = strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
	}

	graph := core.NewGraph(name)
	for _, node := range spec.Nodes {
		nodeName := node.Name
		if nodeName == "" {
			nodeName = node.ID
		}
		// Visualization only needs the topology, so every node gets an
		// identity function.
		graph.AddNode(node.ID, nodeName, func(ctx context.Context, state *core.BaseState) (*core.BaseState, error) {
			return state, nil
		})
	}
	for _, edge := range spec.Edges {
		graph.AddEdge(edge.From, edge.To, nil)
	}

	start := spec.StartNode
	if start == "" {
		start = spec.Nodes[0].ID
	}
	if err := graph.SetStartNode(start); err != nil {
		return nil, err
	}
	for _, end := range spec.EndNodes {
		if err := graph.AddEndNode(end); err != nil {
			return nil, err
		}
	}

	if err := graph.Validate(); err != nil {
		return nil, fmt.Errorf("graph in %s is invalid: %w", path, err)
	}
	return graph, nil
}

func runVisualize(out io.Writer, args []string, format, output string) error {
	graph := createSampleGraph()
	source := "built-in sample graph"

	if len(args) > 0 {
		loaded, err := loadGraphFromFile(args[0])
		if err != nil {
			return err
		}
		graph = loaded
		source = args[0]
	}

	visualizer := debug.NewGraphVisualizer(nil, nil)
	topology := visualizer.GetGraphTopology(graph)

	var result string
	switch format {
	case "mermaid":
		result = visualizer.GenerateMermaidDiagram(topology)
	case "dot":
		result = visualizer.GenerateDotDiagram(topology)
	case "json":
		// This branch used to emit the literal string "JSON output not
		// implemented yet" -- and then write it to the output file and report
		// "Visualization saved to <file>".
		encoded, err := json.MarshalIndent(topology, "", "  ")
		if err != nil {
			return fmt.Errorf("failed to encode topology: %w", err)
		}
		result = string(encoded)
	default:
		return fmt.Errorf("unsupported format %q (want mermaid, dot or json)", format)
	}

	_, _ = fmt.Fprintf(out, "Visualizing %s in %s format...\n", source, format)
	if len(args) == 0 {
		_, _ = fmt.Fprintln(out, "(no graph file given; pass one to visualize your own graph)")
	}

	if output != "" {
		if err := os.WriteFile(output, []byte(result), 0600); err != nil {
			return fmt.Errorf("failed to write output file: %w", err)
		}
		_, _ = fmt.Fprintf(out, "Visualization saved to %s\n", output)
		return nil
	}

	_, _ = fmt.Fprintln(out, result)
	return nil
}

// createSampleGraph builds the graph rendered when no graph file is given.
func createSampleGraph() *core.Graph {
	graph := core.NewGraph("sample-graph")

	// Add some sample nodes
	graph.AddNode("start", "Start", func(ctx context.Context, state *core.BaseState) (*core.BaseState, error) {
		return state, nil
	})

	graph.AddNode("process", "Process", func(ctx context.Context, state *core.BaseState) (*core.BaseState, error) {
		return state, nil
	})

	graph.AddNode("end", "End", func(ctx context.Context, state *core.BaseState) (*core.BaseState, error) {
		return state, nil
	})

	// Add edges
	graph.AddEdge("start", "process", nil)
	graph.AddEdge("process", "end", nil)

	// Set start and end nodes. The nodes were added immediately above, so these
	// cannot fail; Validate() would surface it if they ever did.
	_ = graph.SetStartNode("start")
	_ = graph.AddEndNode("end")

	return graph
}

// runTests builds the agents described by a configuration file (or a built-in
// one) and validates the graphs they produce.
//
// The command used to ignore any argument, build one hard-coded agent and print
// "All tests completed successfully", which claimed far more than it checked.
func runTests(out io.Writer, args []string) error {
	llmManager := llm.NewProviderManager()
	toolRegistry := tools.NewToolRegistry()

	var configs []*agentFileConfig
	if len(args) > 0 {
		_, _ = fmt.Fprintf(out, "Testing agent configuration %s...\n", args[0])
		loaded, err := loadAgentConfigs(args[0])
		if err != nil {
			return err
		}
		configs = loaded
	} else {
		_, _ = fmt.Fprintln(out, "No configuration given; running the built-in self-test...")
		configs = []*agentFileConfig{{
			Name:         "test-agent",
			Type:         string(agent.AgentTypeChat),
			Model:        "gpt-3.5-turbo",
			Provider:     "openai",
			SystemPrompt: "You are a helpful assistant for testing.",
			Temperature:  0.7,
			MaxTokens:    1000,
		}}
	}

	for _, cfg := range configs {
		agentConfig := cfg.toAgentConfig()
		// NewAgent falls back to defaults when handed an invalid config, so
		// check the configuration itself before trusting the agent it returns.
		if err := agentConfig.Validate(); err != nil {
			return fmt.Errorf("agent %s: %w", cfg.Name, err)
		}

		built := agent.NewAgent(agentConfig, llmManager, toolRegistry)
		if got := built.GetConfig().Name; got != cfg.Name {
			return fmt.Errorf("agent %s: configuration was rejected by the agent runtime (got name %q)", cfg.Name, got)
		}

		graph := built.GetGraph()
		if graph == nil {
			return fmt.Errorf("agent %s: no execution graph was built", cfg.Name)
		}
		if err := graph.Validate(); err != nil {
			return fmt.Errorf("agent %s: graph validation failed: %w", cfg.Name, err)
		}

		_, _ = fmt.Fprintf(out, "  ✓ %s (%s, %s/%s): graph valid\n",
			cfg.Name, agentConfig.Type, agentConfig.Provider, agentConfig.Model)
	}

	_, _ = fmt.Fprintf(out, "%d agent configuration(s) built and validated. No LLM calls were made.\n", len(configs))
	return nil
}

// safeProjectDir turns a project name into a directory below the working
// directory, rejecting names that escape it.
//
// "golanggraph init ../../etc/whatever" used to create and populate directories
// anywhere the process could write.
func safeProjectDir(name string) (string, error) {
	if strings.TrimSpace(name) == "" {
		return "", errors.New("project name must not be empty")
	}
	if filepath.IsAbs(name) {
		return "", fmt.Errorf("project name %q must be relative to the current directory", name)
	}
	cleaned := filepath.Clean(name)
	if cleaned == "." || cleaned == ".." || strings.HasPrefix(cleaned, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("project name %q escapes the current directory", name)
	}
	return cleaned, nil
}

// projectTemplates lists the templates init understands.
var projectTemplates = []string{"basic", "advanced", "rag"}

func runInit(out io.Writer, args []string, template string, force bool) error {
	projectName := "golanggraph-agent"
	if len(args) > 0 {
		projectName = args[0]
	}

	dir, err := safeProjectDir(projectName)
	if err != nil {
		return err
	}

	// An unknown template used to fall through to the basic one while still
	// reporting the template the operator asked for.
	valid := false
	for _, t := range projectTemplates {
		if t == template {
			valid = true
			break
		}
	}
	if !valid {
		return fmt.Errorf("unknown template %q (want one of: %s)", template, strings.Join(projectTemplates, ", "))
	}

	if entries, readErr := os.ReadDir(dir); readErr == nil && len(entries) > 0 && !force {
		return fmt.Errorf("directory %s already exists and is not empty (use --force to overwrite)", dir)
	}

	_, _ = fmt.Fprintf(out, "Creating project %s from the %s template...\n", dir, template)

	for _, sub := range []string{"", "configs", "agents", "tools", "static", "tests"} {
		if mkErr := os.MkdirAll(filepath.Join(dir, sub), 0750); mkErr != nil {
			return fmt.Errorf("failed to create directory %s: %w", filepath.Join(dir, sub), mkErr)
		}
	}

	// Every project gets a buildable Go program: init used to produce a
	// directory of YAML with no code in it and then tell the operator to run it.
	files := []struct {
		path    string
		content string
	}{
		{"go.mod", projectGoMod(filepath.Base(dir))},
		{"main.go", projectMainGo(filepath.Base(dir))},
		{"README.md", projectReadme(filepath.Base(dir), template)},
		{".gitignore", "/" + filepath.Base(dir) + "\n*.exe\n.env\n"},
	}
	for _, f := range files {
		if writeErr := writeFileChecked(filepath.Join(dir, f.path), f.content); writeErr != nil {
			return writeErr
		}
	}

	switch template {
	case "advanced":
		err = createAdvancedTemplate(dir)
	case "rag":
		err = createRAGTemplate(dir)
	default:
		err = createBasicTemplate(dir)
	}
	if err != nil {
		return err
	}

	_, _ = fmt.Fprintf(out, "Project %s initialized.\n", dir)
	_, _ = fmt.Fprintf(out, "Next steps:\n")
	_, _ = fmt.Fprintf(out, "  cd %s\n", dir)
	_, _ = fmt.Fprintf(out, "  go mod tidy && go run .\n")
	_, _ = fmt.Fprintf(out, "  golanggraph validate configs/agent-config.yaml\n")
	return nil
}

// writeFileChecked writes a generated project file and reports failure. Silent
// os.WriteFile errors are how scaffolding ends up claiming files it never made.
func writeFileChecked(path, content string) error {
	if err := os.WriteFile(path, []byte(content), 0600); err != nil {
		return fmt.Errorf("failed to write %s: %w", path, err)
	}
	return nil
}

func projectGoMod(name string) string {
	return fmt.Sprintf(`module %s

go 1.23.0

require github.com/UnicoLab/GoLangGraph v0.0.0

// The framework is not published to a module proxy yet; point this at your
// checkout (or delete both lines once it is).
replace github.com/UnicoLab/GoLangGraph => ../GoLangGraph
`, name)
}

func projectMainGo(name string) string {
	return fmt.Sprintf(`// Command %s is a GoLangGraph agent generated by "golanggraph init".
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/UnicoLab/GoLangGraph/pkg/agent"
	"github.com/UnicoLab/GoLangGraph/pkg/llm"
	"github.com/UnicoLab/GoLangGraph/pkg/tools"
)

func main() {
	endpoint := os.Getenv("OLLAMA_URL")
	if endpoint == "" {
		endpoint = "http://localhost:11434"
	}

	provider, err := llm.NewOllamaProvider(&llm.ProviderConfig{
		Endpoint: endpoint,
		Timeout:  60 * time.Second,
	})
	if err != nil {
		log.Fatalf("failed to create the ollama provider: %%v", err)
	}

	providers := llm.NewProviderManager()
	if err := providers.RegisterProvider("ollama", provider); err != nil {
		log.Fatalf("failed to register the ollama provider: %%v", err)
	}

	config := agent.DefaultAgentConfig()
	config.Name = %q
	config.Type = agent.AgentTypeChat
	config.Provider = "ollama"
	config.Model = "gemma3:1b"
	config.SystemPrompt = "You are a helpful assistant."

	if err := config.Validate(); err != nil {
		log.Fatalf("invalid agent configuration: %%v", err)
	}

	assistant := agent.NewAgent(config, providers, tools.NewToolRegistry())

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	execution, err := assistant.Execute(ctx, "Say hello in one sentence.")
	if err != nil {
		log.Fatalf("agent execution failed: %%v", err)
	}

	fmt.Println(execution.Output)
}
`, name, name)
}

func projectReadme(name, template string) string {
	return fmt.Sprintf(`# %s

A GoLangGraph project generated with the %q template.

## Run

	go mod tidy
	go run .

The generated agent talks to a local Ollama (set OLLAMA_URL to point elsewhere).

## Layout

- main.go               the agent program
- configs/              agent configuration files
- docker-compose.yml    postgres and redis for state persistence

## Validate the configuration

	golanggraph validate configs/agent-config.yaml
`, name, template)
}

// dockerBuildOptions describes a docker build run.
type dockerBuildOptions struct {
	Distroless bool
	Tag        string
	Dockerfile string
	Platform   string
	DryRun     bool
	ContextDir string
}

// runCommand executes an external command. It is a package variable so tests
// can observe the command line without needing docker installed.
var runCommand = func(ctx context.Context, out io.Writer, name string, args ...string) error {
	cmd := exec.CommandContext(ctx, name, args...) // #nosec G204 -- arguments are built from validated flags
	cmd.Stdout = out
	cmd.Stderr = out
	return cmd.Run()
}

// runDockerBuild builds the container image.
//
// It used to print the command it would have run and then report "Docker build
// command prepared" and exit 0, so a build pipeline calling it produced no
// image and no error. It now runs docker (use --dry-run to only print), and
// fails when docker is missing or the build fails.
func runDockerBuild(ctx context.Context, out io.Writer, args []string, opts dockerBuildOptions) error {
	if ctx == nil {
		ctx = context.Background()
	}

	configFile := "agent-config.yaml"
	if len(args) > 0 {
		configFile = args[0]
	}
	// Building an image around a configuration that does not parse just moves
	// the failure to production.
	if _, err := loadAgentConfigs(configFile); err != nil {
		return fmt.Errorf("agent config %s: %w", configFile, err)
	}

	tag := opts.Tag
	if tag == "" {
		tag = "golanggraph-agent:latest"
	}
	contextDir := opts.ContextDir
	if contextDir == "" {
		contextDir = "."
	}

	var dockerfilePath string
	switch {
	case opts.Dockerfile != "":
		dockerfilePath = opts.Dockerfile
		if _, err := os.Stat(dockerfilePath); err != nil {
			return fmt.Errorf("dockerfile %s: %w", dockerfilePath, err)
		}
	case opts.Distroless:
		dockerfilePath = "Dockerfile.distroless"
		if err := ensureDockerfile(out, dockerfilePath, distrolessDockerfile); err != nil {
			return err
		}
	default:
		dockerfilePath = "Dockerfile.agent"
		if err := ensureDockerfile(out, dockerfilePath, agentDockerfile); err != nil {
			return err
		}
	}

	dockerArgs := []string{"build", "-f", dockerfilePath, "-t", tag}
	if opts.Platform != "" {
		dockerArgs = append(dockerArgs, "--platform", opts.Platform)
	}
	dockerArgs = append(dockerArgs, contextDir)

	_, _ = fmt.Fprintf(out, "docker %s\n", strings.Join(dockerArgs, " "))

	if opts.DryRun {
		_, _ = fmt.Fprintln(out, "Dry run: the image was not built.")
		return nil
	}

	if err := runCommand(ctx, out, "docker", dockerArgs...); err != nil {
		return fmt.Errorf("docker build failed: %w", err)
	}

	_, _ = fmt.Fprintf(out, "Built image %s\n", tag)
	return nil
}

// ensureDockerfile writes the generated Dockerfile, leaving an existing one
// alone: silently overwriting an operator's Dockerfile loses their changes.
func ensureDockerfile(out io.Writer, path, content string) error {
	if _, err := os.Stat(path); err == nil {
		_, _ = fmt.Fprintf(out, "Using existing %s\n", path)
		return nil
	} else if !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("cannot inspect %s: %w", path, err)
	}

	if err := writeFileChecked(path, content); err != nil {
		return err
	}
	_, _ = fmt.Fprintf(out, "Generated %s\n", path)
	return nil
}

// agentFileConfig is one agent as it appears in a configuration file. The
// framework's AgentConfig carries JSON tags only, so decoding YAML straight
// into it drops every snake_case key ("system_prompt", "max_tokens") without a
// word of complaint. This type accepts both spellings.
type agentFileConfig struct {
	Key           string // map key in a multi-agent file, empty for single-agent files
	ID            string
	Name          string
	Type          string
	Model         string
	Provider      string
	SystemPrompt  string
	Temperature   float64
	MaxTokens     int
	MaxIterations int
	Tools         []string
	UnknownKeys   []string
}

// toAgentConfig converts to the framework configuration, filling in the
// framework defaults for anything the file left out.
func (c *agentFileConfig) toAgentConfig() *agent.AgentConfig {
	config := agent.DefaultAgentConfig()
	if c.ID != "" {
		config.ID = c.ID
	}
	config.Name = c.Name
	if c.Type != "" {
		config.Type = agent.AgentType(c.Type)
	}
	config.Model = c.Model
	config.Provider = c.Provider
	config.SystemPrompt = c.SystemPrompt
	if c.Temperature != 0 {
		config.Temperature = c.Temperature
	}
	if c.MaxTokens != 0 {
		config.MaxTokens = c.MaxTokens
	}
	if c.MaxIterations != 0 {
		config.MaxIterations = c.MaxIterations
	}
	if c.Tools != nil {
		config.Tools = c.Tools
	}
	return config
}

// knownAgentKeys are the keys understood inside an agent block.
var knownAgentKeys = map[string]bool{
	"id": true, "name": true, "type": true, "model": true, "provider": true,
	"systemprompt": true, "temperature": true, "maxtokens": true,
	"maxiterations": true, "tools": true, "enablestreaming": true,
	"streamingmode": true, "timeout": true, "metadata": true,
	"description": true, "enabled": true,
}

// knownTopLevelKeys are the keys understood beside the agent definition in a
// configuration file.
var knownTopLevelKeys = map[string]bool{
	"agents": true, "routing": true, "deployment": true, "shared": true,
	"version": true, "database": true, "vectorstore": true, "rag": true,
	"documentloaders": true, "workflow": true, "server": true,
}

// normalizeKey folds "system_prompt", "system-prompt" and "systemPrompt" onto
// one spelling so a configuration is not silently half-read.
func normalizeKey(key string) string {
	return strings.ToLower(strings.NewReplacer("_", "", "-", "", " ", "").Replace(key))
}

// decodeConfigFile reads a YAML or JSON configuration file into a generic map.
func decodeConfigFile(path string) (map[string]interface{}, error) {
	data, err := os.ReadFile(path) // #nosec G304 -- reading the operator's configuration file is the command
	if err != nil {
		return nil, err
	}
	if len(strings.TrimSpace(string(data))) == 0 {
		return nil, fmt.Errorf("%s is empty", path)
	}

	raw := map[string]interface{}{}
	switch ext := strings.ToLower(filepath.Ext(path)); ext {
	case ".json":
		if err := json.Unmarshal(data, &raw); err != nil {
			return nil, fmt.Errorf("parsing %s: %w", path, err)
		}
	case ".yaml", ".yml":
		if err := yaml.Unmarshal(data, &raw); err != nil {
			return nil, fmt.Errorf("parsing %s: %w", path, err)
		}
	default:
		return nil, fmt.Errorf("unsupported config extension %q (want .yaml, .yml or .json)", ext)
	}
	if raw == nil {
		return nil, fmt.Errorf("%s does not contain a configuration object", path)
	}
	return raw, nil
}

// parseAgentBlock reads one agent definition out of a decoded map.
func parseAgentBlock(key string, raw map[string]interface{}) (*agentFileConfig, error) {
	cfg := &agentFileConfig{Key: key}

	for rawKey, value := range raw {
		switch name := normalizeKey(rawKey); name {
		case "id":
			cfg.ID = fmt.Sprintf("%v", value)
		case "name":
			cfg.Name = fmt.Sprintf("%v", value)
		case "type":
			cfg.Type = fmt.Sprintf("%v", value)
		case "model":
			cfg.Model = fmt.Sprintf("%v", value)
		case "provider":
			cfg.Provider = fmt.Sprintf("%v", value)
		case "systemprompt":
			cfg.SystemPrompt = fmt.Sprintf("%v", value)
		case "temperature":
			f, err := toFloat(value)
			if err != nil {
				return nil, fmt.Errorf("temperature: %w", err)
			}
			cfg.Temperature = f
		case "maxtokens":
			n, err := toInt(value)
			if err != nil {
				return nil, fmt.Errorf("max_tokens: %w", err)
			}
			cfg.MaxTokens = n
		case "maxiterations":
			n, err := toInt(value)
			if err != nil {
				return nil, fmt.Errorf("max_iterations: %w", err)
			}
			cfg.MaxIterations = n
		case "tools":
			names, err := toToolNames(value)
			if err != nil {
				return nil, fmt.Errorf("tools: %w", err)
			}
			cfg.Tools = names
		default:
			if !knownAgentKeys[name] && !knownTopLevelKeys[name] {
				cfg.UnknownKeys = append(cfg.UnknownKeys, rawKey)
			}
		}
	}

	sort.Strings(cfg.UnknownKeys)
	if cfg.Name == "" {
		cfg.Name = key
	}
	return cfg, nil
}

func toFloat(value interface{}) (float64, error) {
	switch v := value.(type) {
	case float64:
		return v, nil
	case float32:
		return float64(v), nil
	case int:
		return float64(v), nil
	case int64:
		return float64(v), nil
	default:
		return 0, fmt.Errorf("want a number, got %T (%v)", value, value)
	}
}

func toInt(value interface{}) (int, error) {
	switch v := value.(type) {
	case int:
		return v, nil
	case int64:
		return int(v), nil
	case float64:
		if v != float64(int(v)) {
			return 0, fmt.Errorf("want a whole number, got %v", v)
		}
		return int(v), nil
	default:
		return 0, fmt.Errorf("want a whole number, got %T (%v)", value, value)
	}
}

// toToolNames accepts both `tools: [calculator]` and the list-of-objects form
// `tools: [{name: calculator, enabled: true}]` that the init template writes.
func toToolNames(value interface{}) ([]string, error) {
	items, ok := value.([]interface{})
	if !ok {
		return nil, fmt.Errorf("want a list, got %T", value)
	}

	var names []string
	for _, item := range items {
		switch entry := item.(type) {
		case string:
			names = append(names, entry)
		case map[string]interface{}:
			enabled := true
			var name string
			for k, v := range entry {
				switch normalizeKey(k) {
				case "name":
					name = fmt.Sprintf("%v", v)
				case "enabled":
					if b, ok := v.(bool); ok {
						enabled = b
					}
				}
			}
			if name == "" {
				return nil, fmt.Errorf("tool entry %v has no name", entry)
			}
			if enabled {
				names = append(names, name)
			}
		default:
			return nil, fmt.Errorf("want a tool name or {name, enabled}, got %T", item)
		}
	}
	return names, nil
}

// loadAgentConfigs parses every agent defined in a configuration file. Both a
// single-agent file and a multi-agent file (top-level "agents:") are accepted.
func loadAgentConfigs(path string) ([]*agentFileConfig, error) {
	raw, err := decodeConfigFile(path)
	if err != nil {
		return nil, err
	}

	for key, value := range raw {
		if normalizeKey(key) != "agents" {
			continue
		}
		agentsMap, ok := value.(map[string]interface{})
		if !ok {
			return nil, fmt.Errorf("%s: \"agents\" must be a map of agent id to agent definition", path)
		}
		if len(agentsMap) == 0 {
			return nil, fmt.Errorf("%s: \"agents\" is empty", path)
		}

		ids := make([]string, 0, len(agentsMap))
		for id := range agentsMap {
			ids = append(ids, id)
		}
		sort.Strings(ids)

		configs := make([]*agentFileConfig, 0, len(ids))
		for _, id := range ids {
			block, ok := agentsMap[id].(map[string]interface{})
			if !ok {
				return nil, fmt.Errorf("%s: agent %q is not a mapping", path, id)
			}
			cfg, parseErr := parseAgentBlock(id, block)
			if parseErr != nil {
				return nil, fmt.Errorf("%s: agent %q: %w", path, id, parseErr)
			}
			configs = append(configs, cfg)
		}
		return configs, nil
	}

	cfg, err := parseAgentBlock("", raw)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return []*agentFileConfig{cfg}, nil
}

// validationReport collects everything found in a configuration.
type validationReport struct {
	Errors   []string
	Warnings []string
}

func (r *validationReport) errorf(format string, args ...interface{}) {
	r.Errors = append(r.Errors, fmt.Sprintf(format, args...))
}

func (r *validationReport) warnf(format string, args ...interface{}) {
	r.Warnings = append(r.Warnings, fmt.Sprintf(format, args...))
}

// knownProviders are the providers the framework ships with.
var knownProviders = map[string]bool{"openai": true, "ollama": true, "gemini": true}

// validateAgentConfigs checks parsed agents for real problems: required fields,
// value ranges, known agent types, resolvable tools, and a graph that builds.
//
// The validate command used to check only that the file existed and then print
// "Configuration validation completed successfully!" -- it reported unparseable
// YAML as valid.
func validateAgentConfigs(configs []*agentFileConfig) *validationReport {
	report := &validationReport{}
	toolRegistry := tools.NewToolRegistry()
	known := map[string]bool{}
	for _, name := range toolRegistry.ListTools() {
		known[name] = true
	}
	llmManager := llm.NewProviderManager()

	for _, cfg := range configs {
		label := cfg.Name
		if cfg.Key != "" {
			label = cfg.Key
		}
		if label == "" {
			label = "agent"
		}

		agentConfig := cfg.toAgentConfig()
		if err := agentConfig.Validate(); err != nil {
			report.errorf("%s: %v", label, err)
			continue
		}

		// An unrecognized type is not rejected by the framework: it silently
		// builds a chat agent, so "type: reactt" would run as something else.
		switch agentConfig.Type {
		case agent.AgentTypeChat, agent.AgentTypeReAct, agent.AgentTypeTool:
		default:
			report.errorf("%s: unknown agent type %q (want chat, react or tool)", label, agentConfig.Type)
			continue
		}

		if !knownProviders[strings.ToLower(agentConfig.Provider)] {
			report.warnf("%s: provider %q is not one of the built-in providers", label, agentConfig.Provider)
		}
		if agentConfig.SystemPrompt == "" {
			report.warnf("%s: no system prompt", label)
		}
		for _, tool := range agentConfig.Tools {
			if !known[tool] {
				report.warnf("%s: tool %q is not registered", label, tool)
			}
		}
		for _, key := range cfg.UnknownKeys {
			report.warnf("%s: unknown key %q", label, key)
		}

		built := agent.NewAgent(agentConfig, llmManager, toolRegistry)
		graph := built.GetGraph()
		if graph == nil {
			report.errorf("%s: no execution graph was built", label)
			continue
		}
		if err := graph.Validate(); err != nil {
			report.errorf("%s: execution graph is invalid: %v", label, err)
		}
	}

	return report
}

func runValidate(out io.Writer, args []string, strict bool) error {
	configFile := "agent-config.yaml"
	if len(args) > 0 {
		configFile = args[0]
	}

	_, _ = fmt.Fprintf(out, "Validating %s (strict: %t)...\n", configFile, strict)

	configs, err := loadAgentConfigs(configFile)
	if err != nil {
		return err
	}

	report := validateAgentConfigs(configs)

	// Routing rules that point at agents which do not exist are a deployment
	// outage waiting to happen, so check them here too.
	if raw, err := decodeConfigFile(configFile); err == nil {
		checkRouting(raw, configs, report)
	}

	for _, warning := range report.Warnings {
		_, _ = fmt.Fprintf(out, "  ⚠ %s\n", warning)
	}
	for _, problem := range report.Errors {
		_, _ = fmt.Fprintf(out, "  ✗ %s\n", problem)
	}

	if len(report.Errors) > 0 {
		return fmt.Errorf("%s is invalid: %d problem(s)", configFile, len(report.Errors))
	}
	if strict && len(report.Warnings) > 0 {
		return fmt.Errorf("%s has %d warning(s) and --strict is set", configFile, len(report.Warnings))
	}

	_, _ = fmt.Fprintf(out, "✅ %s is valid: %d agent(s), %d warning(s)\n", configFile, len(configs), len(report.Warnings))
	return nil
}

// checkRouting verifies that routing rules reference agents that exist.
func checkRouting(raw map[string]interface{}, configs []*agentFileConfig, report *validationReport) {
	ids := map[string]bool{}
	for _, cfg := range configs {
		if cfg.Key != "" {
			ids[cfg.Key] = true
		}
		if cfg.ID != "" {
			ids[cfg.ID] = true
		}
	}
	if len(ids) == 0 {
		return
	}

	routing, ok := raw["routing"].(map[string]interface{})
	if !ok {
		return
	}

	if def, isString := routing["default_agent"].(string); isString && def != "" && !ids[def] {
		report.errorf("routing: default agent %q is not defined", def)
	}

	rules, ok := routing["rules"].([]interface{})
	if !ok {
		return
	}
	patterns := map[string]string{}
	for _, item := range rules {
		rule, ok := item.(map[string]interface{})
		if !ok {
			continue
		}
		agentID, _ := rule["agent_id"].(string)
		if agentID != "" && !ids[agentID] {
			report.errorf("routing: rule targets agent %q, which is not defined", agentID)
		}
		if pattern, ok := rule["pattern"].(string); ok && pattern != "" {
			if previous, clash := patterns[pattern]; clash {
				report.warnf("routing: pattern %q is used by both %q and %q", pattern, previous, agentID)
			}
			patterns[pattern] = agentID
		}
	}
}

// runDeployDocker refuses to claim a deployment it cannot perform.
//
// This command used to print "Docker deployment completed for config: X!" for
// any argument at all -- including a path that does not exist -- while doing
// nothing whatsoever.
func runDeployDocker(out io.Writer, args []string) error {
	return runDeployDockerWithOptions(context.Background(), out, args, deployDockerOptions{})
}

type deployDockerOptions struct {
	Tag           string
	ContainerName string
	Port          int
	ContextDir    string
	DryRun        bool
}

func runDeployDockerWithOptions(ctx context.Context, out io.Writer, args []string, opts deployDockerOptions) error {
	if ctx == nil {
		ctx = context.Background()
	}
	configFile := "agent-config.yaml"
	if len(args) > 0 {
		configFile = args[0]
	}

	configs, err := loadAgentConfigs(configFile)
	if err != nil {
		return fmt.Errorf("agent config %s: %w", configFile, err)
	}

	report := validateAgentConfigs(configs)
	if len(report.Errors) > 0 {
		for _, problem := range report.Errors {
			_, _ = fmt.Fprintf(out, "  ✗ %s\n", problem)
		}
		return fmt.Errorf("%s is invalid: %d problem(s)", configFile, len(report.Errors))
	}
	configPath, err := filepath.Abs(configFile)
	if err != nil {
		return fmt.Errorf("resolve agent config %s: %w", configFile, err)
	}

	_, _ = fmt.Fprintf(out, "%s is valid (%d agent(s)).\n", configFile, len(configs))

	tag := opts.Tag
	if tag == "" {
		tag = "golanggraph-agent:latest"
	}
	name := opts.ContainerName
	if name == "" {
		name = "golanggraph-agent"
	}
	if strings.HasPrefix(tag, "-") || strings.HasPrefix(name, "-") || strings.ContainsAny(name, " \t\n") {
		return fmt.Errorf("invalid Docker image tag or container name")
	}
	port := opts.Port
	if port == 0 {
		port = 8080
	}
	if port < 1 || port > 65535 {
		return fmt.Errorf("invalid host port %d", port)
	}

	if err := runDockerBuild(ctx, out, args, dockerBuildOptions{Tag: tag, ContextDir: opts.ContextDir, DryRun: opts.DryRun}); err != nil {
		return err
	}
	// The image stays generic, while this deployment uses the config that was
	// just validated. An absolute bind mount is required by Docker and lets
	// operators rotate the config without rebuilding the application image.
	runArgs := []string{
		"run", "--detach", "--name", name,
		"--publish", fmt.Sprintf("%d:8080", port),
		"--volume", configPath + ":/app/configs/agent-config.yaml:ro",
		tag, "serve", "--agent-config", "/app/configs/agent-config.yaml",
	}
	_, _ = fmt.Fprintf(out, "docker %s\n", strings.Join(runArgs, " "))
	if opts.DryRun {
		_, _ = fmt.Fprintln(out, "Dry run: the container was not started.")
		return nil
	}
	if err := runCommand(ctx, out, "docker", runArgs...); err != nil {
		return fmt.Errorf("docker run failed: %w", err)
	}
	_, _ = fmt.Fprintf(out, "Deployed container %s at http://127.0.0.1:%d\n", name, port)
	return nil
}
func createBasicTemplate(projectName string) error {
	// Create basic agent configuration
	agentConfig := `name: "basic-agent"
type: "chat"
model: "gpt-3.5-turbo"
provider: "openai"
system_prompt: "You are a helpful assistant."
temperature: 0.7
max_tokens: 1000

tools:
  - name: "calculator"
    enabled: true
  - name: "web_search"
    enabled: false

database:
  type: "postgres"
  host: "localhost"
  port: 5432
  database: "golanggraph"
  username: "postgres"
  password: "password"
`

	if err := writeFileChecked(filepath.Join(projectName, "configs", "agent-config.yaml"), agentConfig); err != nil {
		return err
	}

	// Create docker-compose for development
	dockerCompose := `version: '3.8'
services:
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: golanggraph
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
`

	return writeFileChecked(filepath.Join(projectName, "docker-compose.yml"), dockerCompose)
}

func createAdvancedTemplate(projectName string) error {
	if err := createBasicTemplate(projectName); err != nil {
		return err
	}

	// Add advanced configuration
	advancedConfig := `name: "advanced-agent"
type: "multi-agent"
model: "gpt-4"
provider: "openai"
system_prompt: "You are an advanced AI assistant with multiple capabilities."
temperature: 0.7
max_tokens: 2000

agents:
  - name: "research-agent"
    type: "research"
    tools: ["web_search", "document_reader"]
  - name: "analysis-agent"
    type: "analysis"
    tools: ["calculator", "data_analyzer"]
  - name: "synthesis-agent"
    type: "synthesis"
    tools: ["summarizer", "report_generator"]

workflow:
  start_node: "research-agent"
  edges:
    - from: "research-agent"
      to: "analysis-agent"
    - from: "analysis-agent"
      to: "synthesis-agent"
  end_node: "synthesis-agent"

tools:
  - name: "web_search"
    enabled: true
    config:
      api_key: "${SEARCH_API_KEY}"
  - name: "document_reader"
    enabled: true
  - name: "calculator"
    enabled: true
  - name: "data_analyzer"
    enabled: true
  - name: "summarizer"
    enabled: true
  - name: "report_generator"
    enabled: true

database:
  type: "postgres"
  host: "localhost"
  port: 5432
  database: "golanggraph"
  username: "postgres"
  password: "password"

vector_store:
  type: "pgvector"
  host: "localhost"
  port: 5432
  database: "vectordb"
  username: "postgres"
  password: "password"
  dimensions: 1536
`

	return writeFileChecked(filepath.Join(projectName, "configs", "advanced-config.yaml"), advancedConfig)
}

func createRAGTemplate(projectName string) error {
	if err := createAdvancedTemplate(projectName); err != nil {
		return err
	}

	// Add RAG-specific configuration
	ragConfig := `name: "rag-agent"
type: "rag"
model: "gpt-4"
provider: "openai"
system_prompt: "You are a RAG-enabled AI assistant that can retrieve and analyze information from documents."
temperature: 0.7
max_tokens: 2000

rag:
  enabled: true
  chunk_size: 1000
  chunk_overlap: 200
  similarity_threshold: 0.7
  max_chunks: 5
  embedding_model: "text-embedding-ada-002"

vector_store:
  type: "pgvector"
  host: "localhost"
  port: 5432
  database: "vectordb"
  username: "postgres"
  password: "password"
  dimensions: 1536
  collection_name: "documents"

document_loaders:
  - type: "pdf"
    enabled: true
  - type: "text"
    enabled: true
  - type: "markdown"
    enabled: true
  - type: "web"
    enabled: true

tools:
  - name: "vector_search"
    enabled: true
  - name: "document_loader"
    enabled: true
  - name: "web_search"
    enabled: true
  - name: "summarizer"
    enabled: true

database:
  type: "postgres"
  host: "localhost"
  port: 5432
  database: "golanggraph"
  username: "postgres"
  password: "password"
`

	return writeFileChecked(filepath.Join(projectName, "configs", "rag-config.yaml"), ragConfig)
}

// agentDockerfile is the Dockerfile generated for a normal build.
const agentDockerfile = `# Production Dockerfile for GoLangGraph Agent
FROM golang:1.21-alpine AS builder

# Set working directory
WORKDIR /app

# Install dependencies
RUN apk add --no-cache git ca-certificates tzdata

# Copy go mod files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy source code
COPY . .

# Build the binary
ARG VERSION=production
RUN CGO_ENABLED=0 GOOS=linux go build \
    -ldflags="-w -s -X main.version=${VERSION}" \
    -a -installsuffix cgo \
    -o golanggraph-agent \
    ./cmd/golanggraph

# Production stage
FROM alpine:latest

# Install ca-certificates for HTTPS requests
RUN apk --no-cache add ca-certificates tzdata

# Create non-root user
RUN addgroup -g 1001 -S golanggraph && \
    adduser -u 1001 -S golanggraph -G golanggraph

WORKDIR /app

# Copy the binary from builder stage
COPY --from=builder /app/golanggraph-agent .

# Copy configuration files
COPY configs/ ./configs/
COPY static/ ./static/

# Create necessary directories
RUN mkdir -p ./logs ./data

# Change ownership to non-root user
RUN chown -R golanggraph:golanggraph /app

# Switch to non-root user
USER golanggraph

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD ./golanggraph-agent health || exit 1

# Run the agent
ENTRYPOINT ["./golanggraph-agent"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8080"]
`

// distrolessDockerfile is the Dockerfile generated for --distroless builds.
const distrolessDockerfile = `# Distroless Dockerfile for GoLangGraph Agent
FROM golang:1.21-alpine AS builder

# Set working directory
WORKDIR /app

# Install dependencies
RUN apk add --no-cache git ca-certificates tzdata

# Copy go mod files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy source code
COPY . .

# Build the binary
ARG VERSION=production
RUN CGO_ENABLED=0 GOOS=linux go build \
    -ldflags="-w -s -X main.version=${VERSION}" \
    -a -installsuffix cgo \
    -o golanggraph-agent \
    ./cmd/golanggraph

# Distroless production stage
FROM gcr.io/distroless/static:nonroot

# Copy the binary from builder stage
COPY --from=builder /app/golanggraph-agent /

# Copy configuration files
COPY configs/ /configs/
COPY static/ /static/

# Use distroless nonroot user
USER nonroot:nonroot

# Expose port
EXPOSE 8080

# Health check (note: distroless doesn't support HEALTHCHECK)
# Use external health check monitoring

# Run the agent
ENTRYPOINT ["/golanggraph-agent"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8080"]
`

func main() {
	if err := rootCmd.Execute(); err != nil {
		_, _ = fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}
