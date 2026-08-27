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

	"github.com/spf13/cobra"
	yaml "gopkg.in/yaml.v3"

	"github.com/piotrlaczkowski/GoLangGraph/pkg/agent"
	"github.com/piotrlaczkowski/GoLangGraph/pkg/llm"
	"github.com/piotrlaczkowski/GoLangGraph/pkg/server"
	"github.com/piotrlaczkowski/GoLangGraph/pkg/tools"
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
Supports various deployment targets including Docker, Kubernetes, and serverless platforms.`,
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
		return runMultiAgentDeploy(cmd.OutOrStdout(), args, deploymentType, environment, dryRun)
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
	Short: "Check status of deployed agents",
	Long:  `Check the status of deployed agents including health, metrics, and runtime information.`,
	Args:  cobra.MaximumNArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		outputFormat, err := cmd.Flags().GetString("format")
		if err != nil {
			return err
		}
		watch, err := cmd.Flags().GetBool("watch")
		if err != nil {
			return err
		}
		return runMultiAgentStatus(cmd.OutOrStdout(), args, outputFormat, watch)
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
		Short: "Load agent definitions from Go files or plugins",
		Long: `Load agent definitions from Go files or plugins.

This command can load agents defined programmatically in Go files,
either as plugins or by analyzing Go source files in a directory.

Examples:
  # Load agents from a plugin file
  golanggraph multi-agent load ./agents.so

  # Load agents from Go files in a directory
  golanggraph multi-agent load ./agents/

  # Load agents from current directory
  golanggraph multi-agent load .`,
		Args: cobra.MaximumNArgs(1),
		RunE: runMultiAgentLoad,
	}

	// These three configure directory scanning, which is not implemented; the
	// command reports that rather than pretending to have loaded anything.
	multiAgentLoadCmd.Flags().BoolP("recursive", "r", false, "Recursively scan directories (directory loading is not implemented)")
	multiAgentLoadCmd.Flags().StringSliceP("include", "i", []string{"*.go"}, "File patterns to include (directory loading is not implemented)")
	multiAgentLoadCmd.Flags().StringSliceP("exclude", "e", []string{"*_test.go"}, "File patterns to exclude (directory loading is not implemented)")
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
	multiAgentDeployCmd.Flags().StringP("environment", "e", "development", "Deployment environment")
	multiAgentDeployCmd.Flags().Bool("dry-run", false, "Show what would be deployed without actually deploying")

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
		if err := os.MkdirAll(filepath.Join(dir, sub), 0750); err != nil {
			return fmt.Errorf("failed to create directory %s: %w", filepath.Join(dir, sub), err)
		}
	}

	config := createMultiAgentConfig(template, agentCount, routingType)

	for agentID := range config.Agents {
		if err := os.MkdirAll(filepath.Join(dir, "agents", agentID), 0750); err != nil {
			return fmt.Errorf("failed to create agent directory: %w", err)
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
	configFile := "configs/multi-agent.yaml"
	if len(args) > 0 {
		configFile = args[0]
	}

	_, _ = fmt.Fprintf(out, "Deploying multi-agent system: %s\n", configFile)
	_, _ = fmt.Fprintf(out, "Type: %s, Environment: %s, Dry-run: %t\n", deploymentType, environment, dryRun)

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

	if dryRun {
		_, _ = fmt.Fprintf(out, "DRY RUN - would deploy the following agents:\n")
		for _, agentID := range sortedAgentIDs(config) {
			agentConfig := config.Agents[agentID]
			_, _ = fmt.Fprintf(out, "  - %s: %s (%s on %s)\n", agentID, agentConfig.Name, agentConfig.Type, agentConfig.Provider)
		}
		return nil
	}

	switch config.Deployment.Type {
	case "docker", "kubernetes", "serverless":
		// deployDocker/deployKubernetes/deployServerless printed "Deploying to
		// Docker..." and returned without deploying anything, and the command
		// exited 0. Generate the artifacts and say plainly that applying them
		// is the operator's step.
		return fmt.Errorf("deploying to %s is %w: generate the artifacts with 'golanggraph multi-agent generate %s' and apply them with your own tooling",
			config.Deployment.Type, errNotImplemented, generateSubcommandFor(config.Deployment.Type))
	default:
		return fmt.Errorf("unsupported deployment type %q (want docker, kubernetes or serverless)", config.Deployment.Type)
	}
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
// randomised, so output ordering was different on every run.
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
	autoServer := server.NewAutoServer(&server.AutoServerConfig{
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
	})

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
	configFile := "configs/multi-agent.yaml"
	if len(args) > 0 {
		configFile = args[0]
	}

	if watch {
		// --watch used to fall into `select {}`, blocking forever after saying
		// it was watching for changes. Nothing was ever polled or printed.
		return fmt.Errorf("--watch is %w for multi-agent status", errNotImplemented)
	}

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
	if err := os.MkdirAll(outputDir, 0750); err != nil {
		return fmt.Errorf("failed to create %s: %w", outputDir, err)
	}

	compose := dockerComposeFor(config, multiService)
	composePath := filepath.Join(outputDir, "docker-compose.yml")
	if err := writeFileChecked(composePath, compose); err != nil {
		return err
	}

	dockerfilePath := filepath.Join(outputDir, "Dockerfile")
	if err := writeFileChecked(dockerfilePath, agentDockerfile); err != nil {
		return err
	}

	_, _ = fmt.Fprintf(out, "Generated %s\n", composePath)
	_, _ = fmt.Fprintf(out, "Generated %s\n", dockerfilePath)
	return nil
}

// dockerComposeFor renders a compose file for the configured agents. With
// --multi-service each agent gets its own service and port.
func dockerComposeFor(config *agent.MultiAgentConfig, multiService bool) string {
	var b strings.Builder
	b.WriteString("# Generated by golanggraph multi-agent generate docker\n")
	b.WriteString("services:\n")

	if multiService {
		port := 8080
		for _, agentID := range sortedAgentIDs(config) {
			_, _ = fmt.Fprintf(&b, "  %s:\n", agentID)
			b.WriteString("    build: .\n")
			_, _ = fmt.Fprintf(&b, "    command: [\"serve\", \"--host\", \"0.0.0.0\", \"--port\", \"8080\"]\n")
			_, _ = fmt.Fprintf(&b, "    ports:\n      - \"%d:8080\"\n", port)
			_, _ = fmt.Fprintf(&b, "    environment:\n      - GOLANGGRAPH_AGENT_ID=%s\n", agentID)
			b.WriteString("    restart: unless-stopped\n")
			port++
		}
	} else {
		b.WriteString("  multi-agent:\n")
		b.WriteString("    build: .\n")
		b.WriteString("    ports:\n      - \"8080:8080\"\n")
		b.WriteString("    volumes:\n      - ./configs:/app/configs:ro\n")
		b.WriteString("    restart: unless-stopped\n")
	}

	return b.String()
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
	if err := os.MkdirAll(outputDir, 0750); err != nil {
		return fmt.Errorf("failed to create %s: %w", outputDir, err)
	}

	replicas := 1
	if config.Deployment != nil && config.Deployment.Replicas > 0 {
		replicas = config.Deployment.Replicas
	}

	manifests := map[string]string{
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
      containers:
      - name: multi-agent
        image: golanggraph-multi-agent:latest
        ports:
        - containerPort: 8080
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

	// Only plugin loading exists. Directory loading used to print "not yet
	// implemented" and then return nil, so a script that checked the exit
	// status was told the load had succeeded.
	if !strings.HasSuffix(path, ".so") {
		return fmt.Errorf("loading agent definitions from a directory is %w; build your agents as a Go plugin and pass the .so file", errNotImplemented)
	}

	if verbose {
		_, _ = fmt.Fprintf(out, "Loading plugin: %s\n", path)
	}
	if err := registry.LoadFromPlugin(path); err != nil {
		return fmt.Errorf("failed to load plugin: %w", err)
	}
	_, _ = fmt.Fprintf(out, "Successfully loaded plugin: %s\n", path)

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
