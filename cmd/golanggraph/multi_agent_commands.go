// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package main

import (
	"context"
	"encoding/json"
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
	"github.com/spf13/cobra"
	yaml "gopkg.in/yaml.v3"

	"github.com/UnicoLab/GoLangGraph/pkg/agent"
	"github.com/UnicoLab/GoLangGraph/pkg/llm"
	"github.com/UnicoLab/GoLangGraph/pkg/server"
	"github.com/UnicoLab/GoLangGraph/pkg/tools"
)

// multiAgentCmd represents the multi-agent command group
var multiAgentCmd = &cobra.Command{
	Use:   "multi-agent",
	Short: "Multi-agent deployment and management commands",
	Long: `Multi-agent commands provide functionality to manage multiple AI agents
with different configurations, routing, and deployment options.

This includes:
- Initializing multi-agent projects
- Deploying multiple agents simultaneously
- Managing agent routing and load balancing
- Monitoring multi-agent deployments
- Schema validation for individual agents`,
}

// multiAgentInitCmd represents the multi-agent init command
var multiAgentInitCmd = &cobra.Command{
	Use:   "init [project-name]",
	Short: "Initialize a new multi-agent project",
	Long: `Initialize a new multi-agent project with example configurations and templates.
Creates a directory structure optimized for managing multiple agents with different
configurations, routing rules, and deployment settings.`,
	Args: cobra.MaximumNArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		template, err := cmd.Flags().GetString("template")
		if err != nil {
			return err
		}
		agentCount, err := cmd.Flags().GetInt("agents")
		if err != nil {
			return err
		}
		outputFormat, err := cmd.Flags().GetString("format")
		if err != nil {
			return err
		}
		routingType, err := cmd.Flags().GetString("routing")
		if err != nil {
			return err
		}
		return runMultiAgentInit(cmd.OutOrStdout(), args, template, agentCount, outputFormat, routingType)
	},
}

// multiAgentValidateCmd represents the multi-agent validate command
var multiAgentValidateCmd = &cobra.Command{
	Use:   "validate [config-file]",
	Short: "Validate multi-agent configuration",
	Long: `Validate multi-agent configuration files including agent definitions,
routing rules, deployment settings, and schema validation.`,
	Args: cobra.MaximumNArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		strict, err := cmd.Flags().GetBool("strict")
		if err != nil {
			return err
		}
		checkSchemas, err := cmd.Flags().GetBool("check-schemas")
		if err != nil {
			return err
		}
		return runMultiAgentValidate(cmd.OutOrStdout(), args, strict, checkSchemas)
	},
}

// multiAgentDeployCmd represents the multi-agent deploy command
var multiAgentDeployCmd = &cobra.Command{
	Use:   "deploy [config-file]",
	Short: "Deploy multiple agents",
	Long: `Deploy multiple agents according to the multi-agent configuration.

Docker deployments build an image and start a named container running the
multi-agent server. Kubernetes and serverless targets require provider-specific
artifacts and are deliberately rejected instead of reporting a false success.`,
	Args: cobra.MaximumNArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		deploymentType, err := cmd.Flags().GetString("type")
		if err != nil {
			return err
		}
		environment, err := cmd.Flags().GetString("environment")
		if err != nil {
			return err
		}
		dryRun, err := cmd.Flags().GetBool("dry-run")
		if err != nil {
			return err
		}
		opts := multiAgentDeployOptions{DryRun: dryRun}
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
		return runMultiAgentDeployWithOptions(cmd.Context(), cmd.OutOrStdout(), args, deploymentType, environment, opts)
	},
}

// multiAgentServeCmd represents the multi-agent serve command
var multiAgentServeCmd = &cobra.Command{
	Use:   "serve [config-file]",
	Short: "Start multi-agent server",
	Long: `Start a server that hosts multiple agents with routing and load balancing.
Provides HTTP endpoints for agent execution, management, and monitoring.`,
	Args: cobra.MaximumNArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		host, err := cmd.Flags().GetString("host")
		if err != nil {
			return err
		}
		port, err := cmd.Flags().GetInt("port")
		if err != nil {
			return err
		}
		ctx, stop := signal.NotifyContext(cmd.Context(), syscall.SIGINT, syscall.SIGTERM)
		defer stop()
		return runMultiAgentServe(ctx, cmd.OutOrStdout(), args, host, port)
	},
}

// multiAgentStatusCmd represents the multi-agent status command
var multiAgentStatusCmd = &cobra.Command{
	Use:   "status [config-file]",
	Short: "Show configured multi-agent definitions",
	Long: `Show the agents declared in a multi-agent configuration. This command
does not claim to contact or health-check a deployment. With --watch it prints
an updated status whenever the configuration file changes.`,
	Args: cobra.MaximumNArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		outputFormat, err := cmd.Flags().GetString("format")
		if err != nil {
			return err
		}
		watch, err := cmd.Flags().GetBool("watch")
		if err != nil {
			return err
		}
		return runMultiAgentStatusWithContext(cmd.Context(), cmd.OutOrStdout(), args, outputFormat, watch)
	},
}

// multiAgentGenerateCmd represents the multi-agent generate command
var multiAgentGenerateCmd = &cobra.Command{
	Use:   "generate",
	Short: "Generate deployment artifacts",
	Long:  `Generate deployment artifacts such as Docker files, Kubernetes manifests, and configuration files.`,
}

// multiAgentGenerateDockerCmd represents the generate docker command
var multiAgentGenerateDockerCmd = &cobra.Command{
	Use:   "docker [config-file]",
	Short: "Generate Docker deployment files",
	Long:  `Generate Docker Compose files and Dockerfiles for multi-agent deployment.`,
	Args:  cobra.MaximumNArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		outputDir, err := cmd.Flags().GetString("output")
		if err != nil {
			return err
		}
		multiService, err := cmd.Flags().GetBool("multi-service")
		if err != nil {
			return err
		}
		return runGenerateDocker(cmd.OutOrStdout(), args, outputDir, multiService)
	},
}

// multiAgentGenerateK8sCmd represents the generate k8s command
var multiAgentGenerateK8sCmd = &cobra.Command{
	Use:   "k8s [config-file]",
	Short: "Generate Kubernetes deployment manifests",
	Long:  `Generate Kubernetes deployment, service, and ingress manifests for multi-agent deployment.`,
	Args:  cobra.MaximumNArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		outputDir, err := cmd.Flags().GetString("output")
		if err != nil {
			return err
		}
		namespace, err := cmd.Flags().GetString("namespace")
		if err != nil {
			return err
		}
		return runGenerateK8s(cmd.OutOrStdout(), args, outputDir, namespace)
	},
}

func init() {
	// Add multi-agent command group to root
	rootCmd.AddCommand(multiAgentCmd)

	// Add subcommands
	// Load command
	multiAgentLoadCmd := &cobra.Command{
		Use:   "load [plugin-path or directory]",
		Short: "Load agent definitions from config files or plugins",
		Long: `Load agent definitions from declarative config files or Go plugins.

Directories contain YAML or JSON agent configuration files. Go plugins remain
supported for programmatic agent definitions; arbitrary Go source is never
compiled or executed during a directory scan.

Examples:
  # Load agents from a plugin file
  golanggraph multi-agent load ./agents.so

  # Load agents from configuration files in a directory
  golanggraph multi-agent load ./agents/

  # Load agents from current directory
  golanggraph multi-agent load .`,
		Args: cobra.MaximumNArgs(1),
		RunE: runMultiAgentLoad,
	}

	multiAgentLoadCmd.Flags().BoolP("recursive", "r", false, "Recursively scan directories for agent config files")
	multiAgentLoadCmd.Flags().StringSliceP("include", "i", []string{"*.yaml", "*.yml", "*.json"}, "Agent config file patterns to include")
	multiAgentLoadCmd.Flags().StringSliceP("exclude", "e", []string{"*_test.yaml", "*_test.yml", "*_test.json"}, "Agent config file patterns to exclude")
	multiAgentLoadCmd.Flags().BoolP("validate", "v", true, "Validate loaded agent definitions")
	multiAgentLoadCmd.Flags().BoolP("verbose", "", false, "Verbose output")

	// List command
	multiAgentListCmd := &cobra.Command{
		Use:   "list",
		Short: "List all registered agent definitions",
		Long: `List all registered agent definitions including those from:
- Configuration files
- Go-based definitions
- Factories
- Plugins

This shows the source, type, and metadata for each registered agent.`,
		RunE: runMultiAgentList,
	}

	multiAgentListCmd.Flags().StringP("format", "f", "table", "Output format (table, json, yaml)")
	multiAgentListCmd.Flags().StringP("filter", "", "", "Filter agents by name pattern")
	multiAgentListCmd.Flags().BoolP("show-metadata", "m", false, "Show agent metadata")
	multiAgentListCmd.Flags().BoolP("show-config", "c", false, "Show agent configuration")

	// Add subcommands after all commands are declared
	multiAgentCmd.AddCommand(multiAgentInitCmd)
	multiAgentCmd.AddCommand(multiAgentValidateCmd)
	multiAgentCmd.AddCommand(multiAgentDeployCmd)
	multiAgentCmd.AddCommand(multiAgentServeCmd)
	multiAgentCmd.AddCommand(multiAgentStatusCmd)
	multiAgentCmd.AddCommand(multiAgentGenerateCmd)
	multiAgentCmd.AddCommand(multiAgentLoadCmd)
	multiAgentCmd.AddCommand(multiAgentListCmd)

	// Add generation subcommands
	multiAgentGenerateCmd.AddCommand(multiAgentGenerateDockerCmd)
	multiAgentGenerateCmd.AddCommand(multiAgentGenerateK8sCmd)

	// Multi-agent init flags
	multiAgentInitCmd.Flags().StringP("template", "t", "basic", "Project template (basic, microservices, rag, workflow)")
	multiAgentInitCmd.Flags().IntP("agents", "a", 3, "Number of agents to create")
	multiAgentInitCmd.Flags().StringP("format", "f", "yaml", "Configuration format (yaml, json)")
	multiAgentInitCmd.Flags().StringP("routing", "r", "path", "Routing type (path, host, header, query)")

	// Multi-agent validate flags
	multiAgentValidateCmd.Flags().BoolP("strict", "s", false, "Enable strict validation")
	multiAgentValidateCmd.Flags().Bool("check-schemas", true, "Validate input/output schemas")

	// Multi-agent deploy flags
	multiAgentDeployCmd.Flags().StringP("type", "t", "docker", "Deployment type (docker, kubernetes, serverless)")
	multiAgentDeployCmd.Flags().StringP("environment", "e", "", "Deployment environment override (defaults to the configuration value)")
	multiAgentDeployCmd.Flags().Bool("dry-run", false, "Show what would be deployed without actually deploying")
	multiAgentDeployCmd.Flags().String("tag", "", "Docker image tag")
	multiAgentDeployCmd.Flags().String("name", "golanggraph-multi-agent", "Name for the Docker container")
	multiAgentDeployCmd.Flags().IntP("port", "p", 8080, "Host port to publish for the multi-agent server")
	multiAgentDeployCmd.Flags().String("context", ".", "Docker build context directory")

	// Multi-agent serve flags
	multiAgentServeCmd.Flags().StringP("host", "H", "0.0.0.0", "Host to bind to")
	multiAgentServeCmd.Flags().IntP("port", "p", 8080, "Port to bind to")

	// Multi-agent status flags
	multiAgentStatusCmd.Flags().StringP("format", "f", "table", "Output format (table, json, yaml)")
	multiAgentStatusCmd.Flags().BoolP("watch", "w", false, "Watch for status changes")

	// Generate docker flags
	multiAgentGenerateDockerCmd.Flags().StringP("output", "o", "./deploy", "Output directory")
	multiAgentGenerateDockerCmd.Flags().Bool("multi-service", true, "Generate multi-service Docker Compose")

	// Generate k8s flags
	multiAgentGenerateK8sCmd.Flags().StringP("output", "o", "./k8s", "Output directory")
	multiAgentGenerateK8sCmd.Flags().StringP("namespace", "n", "golanggraph", "Kubernetes namespace")
}

// runMultiAgentInit initializes a new multi-agent project
func runMultiAgentInit(out io.Writer, args []string, template string, agentCount int, outputFormat, routingType string) error {
	projectName := "golanggraph-multi-agent"
	if len(args) > 0 {
		projectName = args[0]
	}

	// "multi-agent init ../../somewhere" used to scaffold outside the working
	// directory; keep the project below it.
	dir, err := safeProjectDir(projectName)
	if err != nil {
		return err
	}

	switch outputFormat {
	case "yaml", "yml", "json":
	default:
		return fmt.Errorf("unsupported output format %q (want yaml or json)", outputFormat)
	}
	switch template {
	case "basic", "microservices", "rag", "workflow":
	default:
		return fmt.Errorf("unknown template %q (want basic, microservices, rag or workflow)", template)
	}
	switch routingType {
	case "path", "host", "header", "query":
	default:
		return fmt.Errorf("unknown routing type %q (want path, host, header or query)", routingType)
	}
	if agentCount <= 0 {
		return fmt.Errorf("--agents must be at least 1, got %d", agentCount)
	}

	_, _ = fmt.Fprintf(out, "Initializing multi-agent project: %s\n", dir)
	_, _ = fmt.Fprintf(out, "Template: %s, Agents: %d, Format: %s, Routing: %s\n", template, agentCount, outputFormat, routingType)

	for _, sub := range []string{"", "agents", "configs", "deploy", "k8s", "scripts", "static", "tests"} {
		if mkErr := os.MkdirAll(filepath.Join(dir, sub), 0750); mkErr != nil {
			return fmt.Errorf("failed to create directory %s: %w", filepath.Join(dir, sub), mkErr)
		}
	}

	config := createMultiAgentConfig(template, agentCount, routingType)

	for agentID := range config.Agents {
		if mkErr := os.MkdirAll(filepath.Join(dir, "agents", agentID), 0750); mkErr != nil {
			return fmt.Errorf("failed to create agent directory: %w", mkErr)
		}
	}

	configFile := fmt.Sprintf("multi-agent.%s", outputFormat)
	configData, err := marshalConfig(config, outputFormat)
	if err != nil {
		return fmt.Errorf("failed to encode configuration: %w", err)
	}
	if err := writeFileChecked(filepath.Join(dir, "configs", configFile), string(configData)); err != nil {
		return err
	}

	if err := createIndividualAgentConfigs(dir, config, outputFormat); err != nil {
		return err
	}
	if err := createDockerComposeFile(dir, config); err != nil {
		return err
	}
	if err := createK8sManifests(dir, config); err != nil {
		return err
	}
	if err := createProjectREADME(dir, config); err != nil {
		return err
	}

	_, _ = fmt.Fprintf(out, "\nMulti-agent project '%s' initialized.\n", dir)
	_, _ = fmt.Fprintf(out, "\nNext steps:\n")
	_, _ = fmt.Fprintf(out, "  cd %s\n", dir)
	_, _ = fmt.Fprintf(out, "  golanggraph multi-agent validate configs/%s\n", configFile)
	_, _ = fmt.Fprintf(out, "  golanggraph multi-agent serve configs/%s\n", configFile)
	return nil
}

// marshalConfig encodes a configuration in the requested format.
func marshalConfig(config interface{}, format string) ([]byte, error) {
	switch format {
	case "yaml", "yml":
		return yaml.Marshal(config)
	case "json":
		return json.MarshalIndent(config, "", "  ")
	default:
		return nil, fmt.Errorf("unsupported format %q", format)
	}
}

// runMultiAgentValidate validates multi-agent configuration
func runMultiAgentValidate(out io.Writer, args []string, strict, checkSchemas bool) error {
	configFile := "configs/multi-agent.yaml"
	if len(args) > 0 {
		configFile = args[0]
	}

	_, _ = fmt.Fprintf(out, "Validating multi-agent configuration: %s\n", configFile)
	_, _ = fmt.Fprintf(out, "Strict mode: %t, Check schemas: %t\n", strict, checkSchemas)

	config, err := agent.LoadMultiAgentConfigFromFile(configFile)
	if err != nil {
		return err
	}

	if err := config.Validate(); err != nil {
		return fmt.Errorf("configuration validation failed: %w", err)
	}

	report := &validationReport{}
	if checkSchemas {
		schemaReport := validateAgentSchemas(config)
		report.Errors = append(report.Errors, schemaReport.Errors...)
		report.Warnings = append(report.Warnings, schemaReport.Warnings...)
	}
	if strict {
		report.Errors = append(report.Errors, validateStrictMode(config)...)
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

	_, _ = fmt.Fprintf(out, "✅ Configuration validation passed!\n")
	_, _ = fmt.Fprintf(out, "- Agents: %d\n", len(config.Agents))
	// Routing and Deployment are optional pointers, and printing through them
	// unconditionally panicked on any configuration that omitted them -- after
	// the success message had already been printed.
	if config.Routing != nil {
		_, _ = fmt.Fprintf(out, "- Routing rules: %d\n", len(config.Routing.Rules))
	} else {
		_, _ = fmt.Fprintf(out, "- Routing rules: none configured\n")
	}
	if config.Deployment != nil && config.Deployment.Type != "" {
		_, _ = fmt.Fprintf(out, "- Deployment type: %s\n", config.Deployment.Type)
	} else {
		_, _ = fmt.Fprintf(out, "- Deployment type: none configured\n")
	}
	return nil
}

// runMultiAgentDeploy deploys multiple agents
func runMultiAgentDeploy(out io.Writer, args []string, deploymentType, environment string, dryRun bool) error {
	return runMultiAgentDeployWithOptions(context.Background(), out, args, deploymentType, environment, multiAgentDeployOptions{DryRun: dryRun})
}

type multiAgentDeployOptions struct {
	Tag           string
	ContainerName string
	Port          int
	ContextDir    string
	DryRun        bool
}

func runMultiAgentDeployWithOptions(ctx context.Context, out io.Writer, args []string, deploymentType, environment string, opts multiAgentDeployOptions) error {
	if ctx == nil {
		ctx = context.Background()
	}
	configFile := "configs/multi-agent.yaml"
	if len(args) > 0 {
		configFile = args[0]
	}

	_, _ = fmt.Fprintf(out, "Deploying multi-agent system: %s\n", configFile)
	_, _ = fmt.Fprintf(out, "Type: %s, Environment: %s, Dry-run: %t\n", deploymentType, environment, opts.DryRun)

	config, err := agent.LoadMultiAgentConfigFromFile(configFile)
	if err != nil {
		return err
	}
	if err := config.Validate(); err != nil {
		return fmt.Errorf("configuration validation failed: %w", err)
	}

	// config.Deployment is optional; assigning through it panicked.
	if config.Deployment == nil {
		config.Deployment = &agent.DeploymentConfig{}
	}
	if deploymentType != "" {
		config.Deployment.Type = deploymentType
	}
	if environment != "" {
		config.Deployment.Environment = environment
	}

	switch config.Deployment.Type {
	case "docker", "kubernetes", "serverless":
	default:
		return fmt.Errorf("unsupported deployment type %q (want docker, kubernetes or serverless)", config.Deployment.Type)
	}

	if opts.DryRun {
		_, _ = fmt.Fprintf(out, "DRY RUN - would deploy the following agents:\n")
		for _, agentID := range sortedAgentIDs(config) {
			agentConfig := config.Agents[agentID]
			_, _ = fmt.Fprintf(out, "  - %s: %s (%s on %s)\n", agentID, agentConfig.Name, agentConfig.Type, agentConfig.Provider)
		}
		return nil
	}

	switch config.Deployment.Type {
	case "docker":
		return deployMultiAgentDocker(ctx, out, configFile, opts)
	case "kubernetes", "serverless":
		return fmt.Errorf("deploying to %s is %w: generate the artifacts with 'golanggraph multi-agent generate %s' and apply them with your own tooling",
			config.Deployment.Type, errNotImplemented, generateSubcommandFor(config.Deployment.Type))
	default:
		panic("validated deployment type reached unreachable branch")
	}
}

// deployMultiAgentDocker builds the generic GoLangGraph image then runs the
// multi-agent server with the validated host configuration mounted read-only.
func deployMultiAgentDocker(ctx context.Context, out io.Writer, configFile string, opts multiAgentDeployOptions) error {
	configPath, err := filepath.Abs(configFile)
	if err != nil {
		return fmt.Errorf("resolve multi-agent config %s: %w", configFile, err)
	}
	tag := opts.Tag
	if tag == "" {
		tag = "golanggraph-multi-agent:latest"
	}
	name := opts.ContainerName
	if name == "" {
		name = "golanggraph-multi-agent"
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

	if err := runDockerBuild(ctx, out, []string{configFile}, dockerBuildOptions{
		Tag: tag, ContextDir: opts.ContextDir, DryRun: opts.DryRun,
	}); err != nil {
		return err
	}
	runArgs := []string{
		"run", "--detach", "--name", name,
		"--publish", fmt.Sprintf("%d:8080", port),
		"--volume", configPath + ":/app/configs/multi-agent.yaml:ro",
		tag, "multi-agent", "serve", "/app/configs/multi-agent.yaml",
		"--host", "0.0.0.0", "--port", "8080",
	}
	_, _ = fmt.Fprintf(out, "docker %s\n", strings.Join(runArgs, " "))
	if opts.DryRun {
		_, _ = fmt.Fprintln(out, "Dry run: the container was not started.")
		return nil
	}
	if err := runCommand(ctx, out, "docker", runArgs...); err != nil {
		return fmt.Errorf("docker run failed: %w", err)
	}
	_, _ = fmt.Fprintf(out, "Deployed multi-agent container %s at http://127.0.0.1:%d\n", name, port)
	return nil
}

// generateSubcommandFor names the generator that produces artifacts for a
// deployment target.
func generateSubcommandFor(deploymentType string) string {
	if deploymentType == "kubernetes" {
		return "k8s"
	}
	return "docker"
}

// sortedAgentIDs returns the agent IDs in a stable order; Go map iteration is
// randomized, so output ordering was different on every run.
func sortedAgentIDs(config *agent.MultiAgentConfig) []string {
	ids := make([]string, 0, len(config.Agents))
	for id := range config.Agents {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return ids
}

// runMultiAgentServe starts multi-agent server
func runMultiAgentServe(ctx context.Context, out io.Writer, args []string, host string, port int) error {
	configFile := "configs/multi-agent.yaml"
	if len(args) > 0 {
		configFile = args[0]
	}

	if port <= 0 || port > 65535 {
		return fmt.Errorf("invalid port %d", port)
	}

	_, _ = fmt.Fprintf(out, "Starting multi-agent server: %s\n", configFile)

	config, err := agent.LoadMultiAgentConfigFromFile(configFile)
	if err != nil {
		return err
	}

	// The previous implementation built a MultiAgentManager, started it, and
	// then served an unrelated, empty server.NewServer -- while printing
	// "Agent endpoints: http://host:port/agents". Nothing connected the two, so
	// none of the advertised agent endpoints existed. Serve the agents through
	// the auto-server, which generates an endpoint per registered agent.
	// A multi-agent server must only expose the definitions in its own config.
	// Sharing the process-wide registry lets a different embedded server leak
	// agents into this deployment.
	autoServer := server.NewAutoServerWithRegistry(&server.AutoServerConfig{
		Host:             host,
		Port:             port,
		BasePath:         "/api",
		EnableWebUI:      true,
		EnablePlayground: true,
		EnableSchemaAPI:  true,
		EnableMetricsAPI: true,
		EnableCORS:       true,
		SchemaValidation: true,
		ServerTimeout:    30 * time.Second,
		MaxRequestSize:   10 * 1024 * 1024,
		Middleware:       []string{"cors", "logging", "recovery"},
	}, agent.NewAgentRegistry())

	if err := autoServer.LoadAgentsFromConfig(configFile); err != nil {
		return fmt.Errorf("failed to register agents: %w", err)
	}
	_, _ = fmt.Fprintf(out, "Registered %d agent(s): %s\n", len(config.Agents), strings.Join(sortedAgentIDs(config), ", "))

	if err := checkAddressAvailable(host, port); err != nil {
		return err
	}

	_, _ = fmt.Fprintf(out, "Multi-agent server listening on %s:%d\n", host, port)
	_, _ = fmt.Fprintf(out, "Health check: http://%s:%d/health\n", host, port)
	_, _ = fmt.Fprintf(out, "Agent endpoints: http://%s:%d/agents\n", host, port)

	if err := autoServer.Start(ctx); err != nil {
		return fmt.Errorf("server failed: %w", err)
	}
	return nil
}

// runMultiAgentStatus reports the configured agents.
func runMultiAgentStatus(out io.Writer, args []string, outputFormat string, watch bool) error {
	return runMultiAgentStatusWithContext(context.Background(), out, args, outputFormat, watch)
}

func runMultiAgentStatusWithContext(ctx context.Context, out io.Writer, args []string, outputFormat string, watch bool) error {
	configFile := "configs/multi-agent.yaml"
	if len(args) > 0 {
		configFile = args[0]
	}

	if err := writeMultiAgentStatus(out, configFile, outputFormat); err != nil {
		return err
	}
	if !watch {
		return nil
	}
	return watchMultiAgentStatus(ctx, out, configFile, outputFormat)
}

func writeMultiAgentStatus(out io.Writer, configFile, outputFormat string) error {
	config, err := agent.LoadMultiAgentConfigFromFile(configFile)
	if err != nil {
		return err
	}

	_, _ = fmt.Fprintf(out, "Configured agents in %s (this reads the configuration; it does not contact a running deployment):\n", configFile)

	agents := make(map[string]interface{}, len(config.Agents))
	for agentID, agentConfig := range config.Agents {
		agents[agentID] = map[string]interface{}{
			"name":     agentConfig.Name,
			"type":     agentConfig.Type,
			"provider": agentConfig.Provider,
			"model":    agentConfig.Model,
		}
	}
	status := map[string]interface{}{
		"timestamp": time.Now().UTC().Format(time.RFC3339),
		"config":    configFile,
		"agents":    agents,
	}

	switch outputFormat {
	case "json", "yaml":
		encoded, err := marshalConfig(status, outputFormat)
		if err != nil {
			return err
		}
		_, _ = fmt.Fprintln(out, string(encoded))
	case "table", "":
		_, _ = fmt.Fprintf(out, "\n%-20s %-15s %-12s %-20s\n", "Agent ID", "Type", "Provider", "Model")
		_, _ = fmt.Fprintf(out, "%-20s %-15s %-12s %-20s\n", "--------", "----", "--------", "-----")
		for _, agentID := range sortedAgentIDs(config) {
			agentConfig := config.Agents[agentID]
			_, _ = fmt.Fprintf(out, "%-20s %-15s %-12s %-20s\n", agentID, agentConfig.Type, agentConfig.Provider, agentConfig.Model)
		}
	default:
		return fmt.Errorf("unsupported output format %q (want table, json or yaml)", outputFormat)
	}
	return nil
}

// watchMultiAgentStatus watches the directory rather than only the file so
// atomic-save editors (which rename a temporary file into place) are handled.
func watchMultiAgentStatus(ctx context.Context, out io.Writer, configFile, outputFormat string) error {
	absPath, err := filepath.Abs(configFile)
	if err != nil {
		return fmt.Errorf("resolve status config: %w", err)
	}
	absPath = filepath.Clean(absPath)

	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		return fmt.Errorf("create status watcher: %w", err)
	}
	defer func() { _ = watcher.Close() }()
	if err := watcher.Add(filepath.Dir(absPath)); err != nil {
		return fmt.Errorf("watch status config directory: %w", err)
	}
	_, _ = fmt.Fprintf(out, "Watching configured agents in %s\n", absPath)

	var debounce <-chan time.Time
	var timer *time.Timer
	for {
		select {
		case <-ctx.Done():
			if timer != nil {
				timer.Stop()
			}
			return nil
		case event, ok := <-watcher.Events:
			if !ok {
				return nil
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
			if err := writeMultiAgentStatus(out, absPath, outputFormat); err != nil {
				_, _ = fmt.Fprintf(out, "Status update failed; keeping watch active: %v\n", err)
			}
		case err, ok := <-watcher.Errors:
			if !ok {
				return nil
			}
			_, _ = fmt.Fprintf(out, "Status watcher error: %v\n", err)
		}
	}
}

// Helper functions for generating project artifacts

func createMultiAgentConfig(template string, agentCount int, routingType string) *agent.MultiAgentConfig {
	config := agent.DefaultMultiAgentConfig()
	config.Name = "example-multi-agent"
	config.Description = "Example multi-agent configuration"
	config.Routing.Type = routingType

	// Create agents based on template
	switch template {
	case "microservices":
		createMicroservicesAgents(config, agentCount)
	case "rag":
		createRAGAgents(config, agentCount)
	case "workflow":
		createWorkflowAgents(config, agentCount)
	default:
		createBasicAgents(config, agentCount)
	}

	// Setup routing rules
	setupRoutingRules(config, routingType)

	return config
}

func createBasicAgents(config *agent.MultiAgentConfig, count int) {
	agentTypes := []agent.AgentType{agent.AgentTypeChat, agent.AgentTypeReAct, agent.AgentTypeTool}

	for i := 1; i <= count; i++ {
		agentID := fmt.Sprintf("agent-%d", i)
		agentType := agentTypes[(i-1)%len(agentTypes)]

		agentConfig := agent.DefaultAgentConfig()
		agentConfig.ID = agentID
		agentConfig.Name = fmt.Sprintf("Agent %d", i)
		agentConfig.Type = agentType
		agentConfig.Model = "gpt-3.5-turbo"
		agentConfig.Provider = "openai"
		agentConfig.SystemPrompt = fmt.Sprintf("You are Agent %d, a helpful AI assistant specialized in %s tasks.", i, agentType)
		agentConfig.Tools = []string{"calculator", "web_search"}

		config.Agents[agentID] = agentConfig
	}
}

func createMicroservicesAgents(config *agent.MultiAgentConfig, count int) {
	services := []struct {
		name        string
		agentType   agent.AgentType
		description string
		tools       []string
	}{
		{"user-service", agent.AgentTypeChat, "Handles user interactions and authentication", []string{"user_db", "auth"}},
		{"order-service", agent.AgentTypeReAct, "Processes orders and payments", []string{"payment", "inventory"}},
		{"inventory-service", agent.AgentTypeTool, "Manages product inventory", []string{"database", "calculator"}},
		{"notification-service", agent.AgentTypeChat, "Sends notifications and alerts", []string{"email", "sms"}},
		{"analytics-service", agent.AgentTypeReAct, "Provides analytics and insights", []string{"database", "calculator", "chart"}},
	}

	for i := 0; i < count && i < len(services); i++ {
		service := services[i]
		agentID := fmt.Sprintf("agent-%d", i+1)

		agentConfig := agent.DefaultAgentConfig()
		agentConfig.ID = agentID
		agentConfig.Name = service.name
		agentConfig.Type = service.agentType
		agentConfig.Model = "gpt-4"
		agentConfig.Provider = "openai"
		agentConfig.SystemPrompt = fmt.Sprintf("You are the %s agent. %s", service.name, service.description)
		agentConfig.Tools = service.tools

		config.Agents[agentID] = agentConfig
	}
}

func createRAGAgents(config *agent.MultiAgentConfig, count int) {
	ragAgents := []struct {
		name        string
		description string
		domain      string
	}{
		{"document-processor", "Processes and indexes documents", "document-processing"},
		{"knowledge-retriever", "Retrieves relevant knowledge from vector store", "information-retrieval"},
		{"answer-generator", "Generates answers based on retrieved context", "question-answering"},
	}

	for i := 0; i < count && i < len(ragAgents); i++ {
		ragAgent := ragAgents[i]
		agentID := fmt.Sprintf("agent-%d", i+1)

		agentConfig := agent.DefaultAgentConfig()
		agentConfig.ID = agentID
		agentConfig.Name = ragAgent.name
		agentConfig.Type = agent.AgentTypeReAct
		agentConfig.Model = "gpt-4"
		agentConfig.Provider = "openai"
		agentConfig.SystemPrompt = fmt.Sprintf("You are the %s agent specialized in %s. %s",
			ragAgent.name, ragAgent.domain, ragAgent.description)
		agentConfig.Tools = []string{"vector_search", "document_loader", "summarizer"}

		config.Agents[agentID] = agentConfig
	}
}

func createWorkflowAgents(config *agent.MultiAgentConfig, count int) {
	workflowSteps := []struct {
		name        string
		agentType   agent.AgentType
		description string
	}{
		{"input-validator", agent.AgentTypeTool, "Validates and preprocesses input data"},
		{"task-planner", agent.AgentTypeReAct, "Plans the execution workflow"},
		{"executor", agent.AgentTypeReAct, "Executes the planned tasks"},
		{"result-aggregator", agent.AgentTypeTool, "Aggregates and formats results"},
		{"output-formatter", agent.AgentTypeChat, "Formats final output for users"},
	}

	for i := 0; i < count && i < len(workflowSteps); i++ {
		step := workflowSteps[i]
		agentID := fmt.Sprintf("agent-%d", i+1)

		agentConfig := agent.DefaultAgentConfig()
		agentConfig.ID = agentID
		agentConfig.Name = step.name
		agentConfig.Type = step.agentType
		agentConfig.Model = "gpt-4"
		agentConfig.Provider = "openai"
		agentConfig.SystemPrompt = fmt.Sprintf("You are the %s agent in the workflow. %s", step.name, step.description)
		agentConfig.Tools = []string{"validator", "planner", "executor"}

		config.Agents[agentID] = agentConfig
	}
}

func setupRoutingRules(config *agent.MultiAgentConfig, routingType string) {
	i := 1
	for agentID := range config.Agents {
		rule := agent.RoutingRule{
			ID:       fmt.Sprintf("rule-%d", i),
			AgentID:  agentID,
			Method:   "POST",
			Priority: 100 - i,
		}

		switch routingType {
		case "path":
			rule.Pattern = fmt.Sprintf("/%s", agentID)
		case "host":
			rule.Pattern = fmt.Sprintf("%s.example.com", agentID)
		case "header":
			rule.Pattern = fmt.Sprintf("X-Agent-ID:%s", agentID)
		case "query":
			rule.Pattern = fmt.Sprintf("agent=%s", agentID)
		default:
			rule.Pattern = fmt.Sprintf("/%s", agentID)
		}

		config.Routing.Rules = append(config.Routing.Rules, rule)
		i++
	}

	// Set default agent to the first one
	if len(config.Agents) > 0 {
		for agentID := range config.Agents {
			config.Routing.DefaultAgent = agentID
			break
		}
	}
}

func createIndividualAgentConfigs(projectName string, config *agent.MultiAgentConfig, format string) error {
	for agentID, agentConfig := range config.Agents {
		configData, err := marshalConfig(agentConfig, format)
		if err != nil {
			return fmt.Errorf("agent %s: %w", agentID, err)
		}
		path := filepath.Join(projectName, "agents", agentID, fmt.Sprintf("config.%s", format))
		if err := writeFileChecked(path, string(configData)); err != nil {
			return err
		}
	}
	return nil
}

func createDockerComposeFile(projectName string, config *agent.MultiAgentConfig) error {
	dockerCompose := `version: '3.8'
services:
  multi-agent:
    build: .
    ports:
      - "8080:8080"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - OLLAMA_URL=${OLLAMA_URL}
    volumes:
      - ./configs:/app/configs:ro
    depends_on:
      - postgres
      - redis

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

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

volumes:
  postgres_data:
  ollama_data:
`

	return writeFileChecked(filepath.Join(projectName, "docker-compose.yml"), dockerCompose)
}

func createK8sManifests(projectName string, config *agent.MultiAgentConfig) error {
	k8sDir := filepath.Join(projectName, "k8s")

	// Deployment manifest
	deployment := `apiVersion: apps/v1
kind: Deployment
metadata:
  name: golanggraph-multi-agent
  labels:
    app: golanggraph-multi-agent
spec:
  replicas: 1
  selector:
    matchLabels:
      app: golanggraph-multi-agent
  template:
    metadata:
      labels:
        app: golanggraph-multi-agent
    spec:
      containers:
      - name: multi-agent
        image: golanggraph-multi-agent:latest
        ports:
        - containerPort: 8080
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: golanggraph-secrets
              key: openai-api-key
`

	// Service manifest
	service := `apiVersion: v1
kind: Service
metadata:
  name: golanggraph-multi-agent-service
spec:
  selector:
    app: golanggraph-multi-agent
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8080
  type: LoadBalancer
`

	if err := writeFileChecked(filepath.Join(k8sDir, "deployment.yaml"), deployment); err != nil {
		return err
	}
	return writeFileChecked(filepath.Join(k8sDir, "service.yaml"), service)
}

func createProjectREADME(projectName string, config *agent.MultiAgentConfig) error {
	readme := fmt.Sprintf("# %s\n\nMulti-agent GoLangGraph project with %d agents.\n\n", projectName, len(config.Agents))

	readme += "## Quick Start\n\n"
	readme += "1. **Validate configuration:**\n"
	readme += "   ```bash\n"
	readme += "   golanggraph multi-agent validate configs/multi-agent.yaml\n"
	readme += "   ```\n\n"
	readme += "2. **Start the multi-agent server:**\n"
	readme += "   ```bash\n"
	readme += "   golanggraph multi-agent serve configs/multi-agent.yaml\n"
	readme += "   ```\n\n"
	readme += "3. **Deploy with Docker:**\n"
	readme += "   ```bash\n"
	readme += "   docker-compose up -d\n"
	readme += "   ```\n\n"
	readme += "4. **Deploy to Kubernetes:**\n"
	readme += "   ```bash\n"
	readme += "   kubectl apply -f k8s/\n"
	readme += "   ```\n\n"

	readme += "## Configuration\n\n"
	readme += "The multi-agent configuration is defined in `configs/multi-agent.yaml`.\n\n"
	readme += "### Agents\n\n"

	for _, agentID := range sortedAgentIDs(config) {
		agentConfig := config.Agents[agentID]
		readme += fmt.Sprintf("- **%s**: %s (%s)\n", agentID, agentConfig.Name, agentConfig.Type)
	}

	readme += "\n### Routing\n\n"
	readme += "Requests are routed to different agents based on the configured routing rules.\n\n"
	readme += "### API Endpoints\n\n"
	readme += "- `POST /agent-1` - Route to Agent 1\n"
	readme += "- `POST /agent-2` - Route to Agent 2\n"
	readme += "- `GET /health` - Health check\n"
	readme += "- `GET /metrics` - Metrics\n"
	readme += "- `GET /agents` - List all agents\n\n"

	readme += "## Development\n\n"
	readme += "1. **Add a new agent:**\n"
	readme += "   - Edit `configs/multi-agent.yaml`\n"
	readme += "   - Add agent configuration\n"
	readme += "   - Update routing rules\n"
	readme += "   - Validate configuration\n\n"
	readme += "2. **Test changes:**\n"
	readme += "   ```bash\n"
	readme += "   golanggraph multi-agent validate\n"
	readme += "   golanggraph multi-agent serve\n"
	readme += "   ```\n\n"

	readme += "## Deployment\n\n"
	readme += "### Docker\n\n"
	readme += "```bash\n"
	readme += "docker-compose up -d\n"
	readme += "```\n\n"
	readme += "### Kubernetes\n\n"
	readme += "```bash\n"
	readme += "kubectl apply -f k8s/\n"
	readme += "```\n\n"

	readme += "## Monitoring\n\n"
	readme += "- Health: `http://localhost:8080/health`\n"
	readme += "- Metrics: `http://localhost:8080/metrics`\n"
	readme += "- Agent Status: `http://localhost:8080/agents`\n"

	return writeFileChecked(filepath.Join(projectName, "README.md"), readme)
}

// validateStrictMode applies the checks that MultiAgentConfig.Validate does not.
//
// It used to be `return nil` with a "// Add strict validation logic here"
// comment, so "multi-agent validate --strict" reported that strict validation
// had passed without performing any.
func validateStrictMode(config *agent.MultiAgentConfig) []string {
	var problems []string

	for _, agentID := range sortedAgentIDs(config) {
		agentConfig := config.Agents[agentID]
		if agentConfig.SystemPrompt == "" {
			problems = append(problems, fmt.Sprintf("agent %s: no system prompt", agentID))
		}
		if agentConfig.ID != "" && agentConfig.ID != agentID {
			problems = append(problems, fmt.Sprintf("agent %s: id field is %q, which does not match its key", agentID, agentConfig.ID))
		}
		if !knownProviders[strings.ToLower(agentConfig.Provider)] {
			problems = append(problems, fmt.Sprintf("agent %s: provider %q is not one of the built-in providers", agentID, agentConfig.Provider))
		}
	}

	if config.Routing != nil {
		if config.Routing.DefaultAgent == "" {
			problems = append(problems, "routing: no default agent")
		}
		patterns := map[string]string{}
		for _, rule := range config.Routing.Rules {
			if rule.Pattern == "" {
				problems = append(problems, fmt.Sprintf("routing: rule %s has no pattern", rule.ID))
				continue
			}
			if previous, clash := patterns[rule.Pattern]; clash {
				problems = append(problems, fmt.Sprintf("routing: pattern %q is claimed by both %s and %s", rule.Pattern, previous, rule.AgentID))
			}
			patterns[rule.Pattern] = rule.AgentID
		}
		for _, agentID := range sortedAgentIDs(config) {
			routed := false
			for _, rule := range config.Routing.Rules {
				if rule.AgentID == agentID {
					routed = true
					break
				}
			}
			if !routed && config.Routing.DefaultAgent != agentID {
				problems = append(problems, fmt.Sprintf("agent %s: no routing rule reaches it", agentID))
			}
		}
	}

	return problems
}

// validateAgentSchemas checks each agent definition against what the runtime
// requires: a known type, resolvable tools and a graph that builds.
//
// This was `return nil` too, while --check-schemas defaults to true -- so every
// "multi-agent validate" run reported schema validation it never did.
func validateAgentSchemas(config *agent.MultiAgentConfig) *validationReport {
	report := &validationReport{}

	toolRegistry := tools.NewToolRegistry()
	known := map[string]bool{}
	for _, name := range toolRegistry.ListTools() {
		known[name] = true
	}
	llmManager := llm.NewProviderManager()

	for _, agentID := range sortedAgentIDs(config) {
		// Validate a copy with the framework defaults filled in for keys the
		// file omits, so a configuration that merely leaves max_tokens unset is
		// not reported as broken (the runtime defaults it too).
		agentConfig := *config.Agents[agentID]
		defaults := agent.DefaultAgentConfig()
		if agentConfig.MaxTokens == 0 {
			agentConfig.MaxTokens = defaults.MaxTokens
		}
		if agentConfig.MaxIterations == 0 {
			agentConfig.MaxIterations = defaults.MaxIterations
		}

		if err := agentConfig.Validate(); err != nil {
			report.errorf("agent %s: %v", agentID, err)
			continue
		}

		switch agentConfig.Type {
		case agent.AgentTypeChat, agent.AgentTypeReAct, agent.AgentTypeTool:
		default:
			// The runtime silently falls back to a chat graph for an unknown
			// type, so an operator would never learn of the typo.
			report.errorf("agent %s: unknown type %q (want chat, react or tool)", agentID, agentConfig.Type)
			continue
		}

		for _, tool := range agentConfig.Tools {
			if !known[tool] {
				// A warning, not an error: tools can also be registered by the
				// operator's own Go code at start-up.
				report.warnf("agent %s: tool %q is not registered", agentID, tool)
			}
		}

		built := agent.NewAgent(&agentConfig, llmManager, toolRegistry)
		graph := built.GetGraph()
		if graph == nil {
			report.errorf("agent %s: no execution graph was built", agentID)
			continue
		}
		if err := graph.Validate(); err != nil {
			report.errorf("agent %s: execution graph is invalid: %v", agentID, err)
		}
	}

	return report
}

// runGenerateDocker writes the Docker deployment artifacts.
//
// It used to print "Generating Docker deployment files..." and generate
// nothing, ignoring --output and --multi-service entirely.
func runGenerateDocker(out io.Writer, args []string, outputDir string, multiService bool) error {
	configFile := "configs/multi-agent.yaml"
	if len(args) > 0 {
		configFile = args[0]
	}
	if outputDir == "" {
		outputDir = "./deploy"
	}

	config, err := agent.LoadMultiAgentConfigFromFile(configFile)
	if err != nil {
		return err
	}
	if err = os.MkdirAll(outputDir, 0750); err != nil {
		return fmt.Errorf("failed to create %s: %w", outputDir, err)
	}

	buildContext, dockerfile, err := dockerBuildPaths(outputDir)
	if err != nil {
		return err
	}
	configFiles, err := writeDockerAgentConfigs(outputDir, config, multiService)
	if err != nil {
		return err
	}

	compose := dockerComposeFor(config, multiService, buildContext, dockerfile, configFiles)
	composePath := filepath.Join(outputDir, "docker-compose.yml")
	if err := writeFileChecked(composePath, compose); err != nil {
		return err
	}

	dockerfilePath := filepath.Join(outputDir, "Dockerfile")
	if err := writeFileChecked(dockerfilePath, multiAgentDockerfile); err != nil {
		return err
	}

	_, _ = fmt.Fprintf(out, "Generated %s\n", composePath)
	_, _ = fmt.Fprintf(out, "Generated %s\n", dockerfilePath)
	return nil
}

// dockerBuildPaths derives paths understood by Docker Compose, which resolves
// build.context relative to the compose file and dockerfile relative to that
// context. The generated artifact may live outside the source checkout.
func dockerBuildPaths(outputDir string) (buildContext, dockerfile string, err error) {
	outputDir, err = filepath.Abs(outputDir)
	if err != nil {
		return "", "", fmt.Errorf("resolve Docker output directory: %w", err)
	}
	workdir, err := os.Getwd()
	if err != nil {
		return "", "", fmt.Errorf("get working directory: %w", err)
	}
	workdir, err = filepath.Abs(workdir)
	if err != nil {
		return "", "", fmt.Errorf("resolve working directory: %w", err)
	}

	buildContext, err = filepath.Rel(outputDir, workdir)
	if err != nil {
		return "", "", fmt.Errorf("derive Docker build context: %w", err)
	}
	dockerfile, err = filepath.Rel(workdir, filepath.Join(outputDir, "Dockerfile"))
	if err != nil {
		return "", "", fmt.Errorf("derive Dockerfile path: %w", err)
	}
	if buildContext == "" {
		buildContext = "."
	}
	return filepath.ToSlash(buildContext), filepath.ToSlash(dockerfile), nil
}

// writeDockerAgentConfigs makes the generated compose file self-contained as
// far as runtime configuration is concerned. In multi-service mode, each
// service gets a one-agent configuration; mounting the same fleet file in
// every service would start every agent everywhere and make AGENT_ID a no-op.
func writeDockerAgentConfigs(outputDir string, config *agent.MultiAgentConfig, multiService bool) (map[string]string, error) {
	configDir := filepath.Join(outputDir, "configs")
	if err := os.MkdirAll(configDir, 0750); err != nil {
		return nil, fmt.Errorf("create generated config directory: %w", err)
	}

	files := make(map[string]string, len(config.Agents))
	if !multiService {
		encoded, err := yaml.Marshal(config)
		if err != nil {
			return nil, fmt.Errorf("encode multi-agent config: %w", err)
		}
		const name = "multi-agent.yaml"
		if err := writeFileChecked(filepath.Join(configDir, name), string(encoded)); err != nil {
			return nil, err
		}
		files["multi-agent"] = name
		return files, nil
	}

	for _, agentID := range sortedAgentIDs(config) {
		if agentID == "." || agentID == ".." || strings.ContainsAny(agentID, `/\\`) {
			return nil, fmt.Errorf("agent id %q cannot be used as a generated configuration filename", agentID)
		}
		name := agentID + ".yaml"
		oneAgent := &agent.MultiAgentConfig{
			Name:        config.Name + "-" + agentID,
			Version:     config.Version,
			Description: config.Description,
			Agents:      map[string]*agent.AgentConfig{agentID: config.Agents[agentID]},
			Deployment:  deploymentForAgent(config.Deployment, agentID),
			Shared:      config.Shared,
			Metadata:    config.Metadata,
		}
		encoded, err := yaml.Marshal(oneAgent)
		if err != nil {
			return nil, fmt.Errorf("encode generated config for agent %s: %w", agentID, err)
		}
		if err := writeFileChecked(filepath.Join(configDir, name), string(encoded)); err != nil {
			return nil, err
		}
		files[agentID] = name
	}
	return files, nil
}

// deploymentForAgent clones the parts of a deployment configuration that
// refer to individual agents. A service generated for one agent cannot retain
// health checks for its former peers: validation correctly rejects those IDs,
// and the container would otherwise restart forever.
func deploymentForAgent(deployment *agent.DeploymentConfig, agentID string) *agent.DeploymentConfig {
	if deployment == nil {
		return nil
	}

	copy := *deployment
	if deployment.HealthCheck == nil {
		return &copy
	}

	healthCheck := *deployment.HealthCheck
	if specific, exists := deployment.HealthCheck.AgentSpecific[agentID]; exists {
		healthCheck.AgentSpecific = map[string]*agent.HealthCheckConfig{agentID: specific}
	} else {
		healthCheck.AgentSpecific = nil
	}
	copy.HealthCheck = &healthCheck
	return &copy
}

// dockerComposeFor renders a compose file for the configured agents. With
// --multi-service each agent gets its own service and port.
func dockerComposeFor(config *agent.MultiAgentConfig, multiService bool, buildContext, dockerfile string, configFiles map[string]string) string {
	var b strings.Builder
	b.WriteString("# Generated by golanggraph multi-agent generate docker\n")
	b.WriteString("services:\n")

	if multiService {
		port := 8080
		for _, agentID := range sortedAgentIDs(config) {
			_, _ = fmt.Fprintf(&b, "  %s:\n", agentID)
			writeComposeBuild(&b, buildContext, dockerfile)
			_, _ = fmt.Fprintf(&b, "    command: [\"multi-agent\", \"serve\", \"/app/configs/%s\", \"--host\", \"0.0.0.0\", \"--port\", \"8080\"]\n", configFiles[agentID])
			_, _ = fmt.Fprintf(&b, "    ports:\n      - \"%d:8080\"\n", port)
			_, _ = fmt.Fprintf(&b, "    volumes:\n      - ./configs/%s:/app/configs/%s:ro\n", configFiles[agentID], configFiles[agentID])
			b.WriteString("    restart: unless-stopped\n")
			port++
		}
	} else {
		b.WriteString("  multi-agent:\n")
		writeComposeBuild(&b, buildContext, dockerfile)
		_, _ = fmt.Fprintf(&b, "    command: [\"multi-agent\", \"serve\", \"/app/configs/%s\", \"--host\", \"0.0.0.0\", \"--port\", \"8080\"]\n", configFiles["multi-agent"])
		b.WriteString("    ports:\n      - \"8080:8080\"\n")
		_, _ = fmt.Fprintf(&b, "    volumes:\n      - ./configs/%s:/app/configs/%s:ro\n", configFiles["multi-agent"], configFiles["multi-agent"])
		b.WriteString("    restart: unless-stopped\n")
	}

	return b.String()
}

func writeComposeBuild(b *strings.Builder, buildContext, dockerfile string) {
	b.WriteString("    build:\n")
	_, _ = fmt.Fprintf(b, "      context: %q\n", buildContext)
	_, _ = fmt.Fprintf(b, "      dockerfile: %q\n", dockerfile)
}

// runGenerateK8s writes the Kubernetes manifests.
//
// It used to print "Generating Kubernetes manifests..." and generate nothing,
// ignoring --output and --namespace entirely.
func runGenerateK8s(out io.Writer, args []string, outputDir, namespace string) error {
	configFile := "configs/multi-agent.yaml"
	if len(args) > 0 {
		configFile = args[0]
	}
	if outputDir == "" {
		outputDir = "./k8s"
	}
	if namespace == "" {
		namespace = "golanggraph"
	}

	config, err := agent.LoadMultiAgentConfigFromFile(configFile)
	if err != nil {
		return err
	}
	if err = os.MkdirAll(outputDir, 0750); err != nil {
		return fmt.Errorf("failed to create %s: %w", outputDir, err)
	}

	replicas := 1
	if config.Deployment != nil && config.Deployment.Replicas > 0 {
		replicas = config.Deployment.Replicas
	}

	configYAML, err := yaml.Marshal(config)
	if err != nil {
		return fmt.Errorf("encode Kubernetes agent configuration: %w", err)
	}
	manifests := map[string]string{
		"configmap.yaml":  k8sConfigMapManifest(namespace, string(configYAML)),
		"namespace.yaml":  k8sNamespaceManifest(namespace),
		"deployment.yaml": k8sDeploymentManifest(namespace, replicas),
		"service.yaml":    k8sServiceManifest(namespace),
	}

	names := make([]string, 0, len(manifests))
	for name := range manifests {
		names = append(names, name)
	}
	sort.Strings(names)

	for _, name := range names {
		path := filepath.Join(outputDir, name)
		if err := writeFileChecked(path, manifests[name]); err != nil {
			return err
		}
		_, _ = fmt.Fprintf(out, "Generated %s\n", path)
	}
	return nil
}

func k8sNamespaceManifest(namespace string) string {
	return fmt.Sprintf(`apiVersion: v1
kind: Namespace
metadata:
  name: %s
`, namespace)
}

func k8sConfigMapManifest(namespace, configYAML string) string {
	var data strings.Builder
	for _, line := range strings.Split(strings.TrimSuffix(configYAML, "\n"), "\n") {
		data.WriteString("    ")
		data.WriteString(line)
		data.WriteByte('\n')
	}
	return fmt.Sprintf(`apiVersion: v1
kind: ConfigMap
metadata:
  name: golanggraph-multi-agent-config
  namespace: %s
data:
  multi-agent.yaml: |
%s`, namespace, data.String())
}

func k8sDeploymentManifest(namespace string, replicas int) string {
	return fmt.Sprintf(`apiVersion: apps/v1
kind: Deployment
metadata:
  name: golanggraph-multi-agent
  namespace: %s
  labels:
    app: golanggraph-multi-agent
spec:
  replicas: %d
  selector:
    matchLabels:
      app: golanggraph-multi-agent
  template:
    metadata:
      labels:
        app: golanggraph-multi-agent
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1001
      containers:
      - name: multi-agent
        image: golanggraph-multi-agent:latest
        imagePullPolicy: IfNotPresent
        command: ["./golanggraph", "multi-agent", "serve", "/app/configs/multi-agent.yaml", "--host", "0.0.0.0", "--port", "8080"]
        ports:
        - containerPort: 8080
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop: ["ALL"]
        volumeMounts:
        - name: agent-config
          mountPath: /app/configs
          readOnly: true
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 30
      volumes:
      - name: agent-config
        configMap:
          name: golanggraph-multi-agent-config
`, namespace, replicas)
}

func k8sServiceManifest(namespace string) string {
	return fmt.Sprintf(`apiVersion: v1
kind: Service
metadata:
  name: golanggraph-multi-agent
  namespace: %s
spec:
  selector:
    app: golanggraph-multi-agent
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8080
  type: ClusterIP
`, namespace)
}

// runMultiAgentLoad loads agent definitions from Go files or plugins
func runMultiAgentLoad(cmd *cobra.Command, args []string) error {
	path := "."
	if len(args) > 0 {
		path = args[0]
	}

	validate, err := cmd.Flags().GetBool("validate")
	if err != nil {
		return err
	}
	verbose, err := cmd.Flags().GetBool("verbose")
	if err != nil {
		return err
	}

	out := cmd.OutOrStdout()
	_, _ = fmt.Fprintf(out, "Loading agent definitions from: %s\n", path)

	registry := agent.GetGlobalRegistry()

	if strings.HasSuffix(strings.ToLower(path), ".so") {
		if verbose {
			_, _ = fmt.Fprintf(out, "Loading plugin: %s\n", path)
		}
		if err := registry.LoadFromPlugin(path); err != nil {
			return fmt.Errorf("failed to load plugin: %w", err)
		}
		_, _ = fmt.Fprintf(out, "Successfully loaded plugin: %s\n", path)
	} else {
		recursive, err := cmd.Flags().GetBool("recursive")
		if err != nil {
			return err
		}
		include, err := cmd.Flags().GetStringSlice("include")
		if err != nil {
			return err
		}
		exclude, err := cmd.Flags().GetStringSlice("exclude")
		if err != nil {
			return err
		}
		files, err := agentConfigFiles(path, recursive, include, exclude)
		if err != nil {
			return err
		}
		for _, file := range files {
			configs, err := loadAgentConfigs(file)
			if err != nil {
				return fmt.Errorf("load agent config %s: %w", file, err)
			}
			report := validateAgentConfigs(configs)
			if len(report.Errors) > 0 {
				return fmt.Errorf("agent config %s is invalid: %s", file, strings.Join(report.Errors, "; "))
			}
			for _, cfg := range configs {
				id := cfg.ID
				if id == "" {
					id = cfg.Key
				}
				if id == "" {
					id = cfg.Name
				}
				if id == "" {
					return fmt.Errorf("agent config %s has an agent without an id or name", file)
				}
				if err := registry.RegisterDefinition(id, agent.NewBaseAgentDefinition(cfg.toAgentConfig())); err != nil {
					return fmt.Errorf("register agent %q from %s: %w", id, file, err)
				}
			}
			if verbose {
				_, _ = fmt.Fprintf(out, "Loaded agent config: %s\n", file)
			}
		}
		_, _ = fmt.Fprintf(out, "Successfully loaded %d agent config file(s)\n", len(files))
	}

	// List loaded agents
	definitions := registry.ListDefinitions()
	factories := registry.ListFactories()
	sort.Strings(definitions)
	sort.Strings(factories)

	_, _ = fmt.Fprintf(out, "\nLoaded agents:\n")
	_, _ = fmt.Fprintf(out, "  Definitions: %d\n", len(definitions))
	_, _ = fmt.Fprintf(out, "  Factories: %d\n", len(factories))

	if !validate {
		return nil
	}

	_, _ = fmt.Fprintf(out, "\nValidating loaded agent definitions...\n")

	var invalid []string
	for _, defID := range definitions {
		def, exists := registry.GetDefinition(defID)
		if !exists {
			continue
		}
		if err := def.Validate(); err != nil {
			_, _ = fmt.Fprintf(out, "  ❌ %s: %v\n", defID, err)
			invalid = append(invalid, defID)
			continue
		}
		_, _ = fmt.Fprintf(out, "  ✅ %s: valid\n", defID)
	}

	for _, factoryID := range factories {
		// This used to print "✅ Factory X: valid" whenever the registry held
		// any factory at all, without ever looking at the factory in question.
		// Build the definition it produces and validate that.
		definition, err := registry.CreateAgentFromFactory(factoryID, llm.NewProviderManager(), tools.NewToolRegistry())
		if err != nil || definition == nil {
			_, _ = fmt.Fprintf(out, "  ❌ Factory %s: %v\n", factoryID, err)
			invalid = append(invalid, factoryID)
			continue
		}
		_, _ = fmt.Fprintf(out, "  ✅ Factory %s: valid\n", factoryID)
	}

	if len(invalid) > 0 {
		return fmt.Errorf("%d agent definition(s) are invalid: %s", len(invalid), strings.Join(invalid, ", "))
	}
	return nil
}

// agentConfigFiles discovers declarative agent configuration files without
// following symlinks. Go source is intentionally not compiled here: loading
// arbitrary source as a plugin would execute code from the scanned directory.
func agentConfigFiles(root string, recursive bool, include, exclude []string) ([]string, error) {
	info, err := os.Stat(root)
	if err != nil {
		return nil, fmt.Errorf("agent source %s: %w", root, err)
	}
	if !info.IsDir() {
		return nil, fmt.Errorf("agent source %s is not a directory or .so plugin", root)
	}

	var files []string
	err = filepath.WalkDir(root, func(path string, entry os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if entry.IsDir() {
			if path != root && !recursive {
				return filepath.SkipDir
			}
			return nil
		}
		if entry.Type()&os.ModeSymlink != 0 {
			return nil
		}
		ext := strings.ToLower(filepath.Ext(path))
		if ext != ".yaml" && ext != ".yml" && ext != ".json" {
			return nil
		}
		base := filepath.Base(path)
		if !matchesFilePattern(base, include) || matchesFilePattern(base, exclude) {
			return nil
		}
		files = append(files, path)
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("scan agent directory %s: %w", root, err)
	}
	if len(files) == 0 {
		return nil, fmt.Errorf("no agent config files found in %s", root)
	}
	sort.Strings(files)
	return files, nil
}

func matchesFilePattern(name string, patterns []string) bool {
	for _, pattern := range patterns {
		if matched, err := filepath.Match(pattern, name); err == nil && matched {
			return true
		}
	}
	return false
}

// runMultiAgentList lists all registered agent definitions
func runMultiAgentList(cmd *cobra.Command, args []string) error {
	format, err := cmd.Flags().GetString("format")
	if err != nil {
		return err
	}
	filter, err := cmd.Flags().GetString("filter")
	if err != nil {
		return err
	}
	showMetadata, err := cmd.Flags().GetBool("show-metadata")
	if err != nil {
		return err
	}
	showConfig, err := cmd.Flags().GetBool("show-config")
	if err != nil {
		return err
	}

	out := cmd.OutOrStdout()
	registry := agent.GetGlobalRegistry()
	infos := registry.GetAgentInfo()
	sort.Slice(infos, func(i, j int) bool { return infos[i].ID < infos[j].ID })

	// Apply filter if specified
	if filter != "" {
		var filteredInfos []agent.AgentInfo
		for _, info := range infos {
			if strings.Contains(info.ID, filter) {
				filteredInfos = append(filteredInfos, info)
			}
		}
		infos = filteredInfos
	}

	switch format {
	case "json":
		output, err := json.MarshalIndent(infos, "", "  ")
		if err != nil {
			return fmt.Errorf("failed to marshal JSON: %w", err)
		}
		_, _ = fmt.Fprintln(out, string(output))

	case "yaml":
		output, err := yaml.Marshal(infos)
		if err != nil {
			return fmt.Errorf("failed to marshal YAML: %w", err)
		}
		_, _ = fmt.Fprintln(out, string(output))

	case "table", "":
		_, _ = fmt.Fprintf(out, "Agent Definitions (%d total):\n\n", len(infos))
		_, _ = fmt.Fprintf(out, "%-20s %-12s %-15s %-10s\n", "ID", "Source", "Type", "Model")
		_, _ = fmt.Fprintf(out, "%-20s %-12s %-15s %-10s\n", "--", "------", "----", "-----")

		for _, info := range infos {
			model := "N/A"
			agentType := "N/A"

			if info.Config != nil {
				model = info.Config.Model
				agentType = string(info.Config.Type)
			}

			_, _ = fmt.Fprintf(out, "%-20s %-12s %-15s %-10s\n",
				info.ID, info.Source, agentType, model)

			if showConfig && info.Config != nil {
				_, _ = fmt.Fprintf(out, "  Config: Name=%s, Provider=%s, Tools=%v\n",
					info.Config.Name, info.Config.Provider, info.Config.Tools)
			}

			if showMetadata && len(info.Metadata) > 0 {
				keys := make([]string, 0, len(info.Metadata))
				for k := range info.Metadata {
					keys = append(keys, k)
				}
				sort.Strings(keys)
				_, _ = fmt.Fprintf(out, "  Metadata: ")
				for _, k := range keys {
					_, _ = fmt.Fprintf(out, "%s=%v ", k, info.Metadata[k])
				}
				_, _ = fmt.Fprintf(out, "\n")
			}
		}

	default:
		return fmt.Errorf("unsupported output format %q (want table, json or yaml)", format)
	}

	return nil
}
