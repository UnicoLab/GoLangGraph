// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/sirupsen/logrus"

	"github.com/piotrlaczkowski/GoLangGraph/pkg/llm"
)

// Tool represents a tool that can be executed by agents
type Tool interface {
	// GetName returns the tool name
	GetName() string

	// GetDescription returns the tool description
	GetDescription() string

	// GetDefinition returns the LLM tool definition
	GetDefinition() llm.ToolDefinition

	// Execute executes the tool with the given arguments
	Execute(ctx context.Context, args string) (string, error)

	// Validate validates the tool arguments
	Validate(args string) error

	// GetConfig returns the tool configuration
	GetConfig() map[string]interface{}

	// SetConfig updates the tool configuration
	SetConfig(config map[string]interface{}) error
}

// ToolRegistry manages a collection of tools
type ToolRegistry struct {
	tools  map[string]Tool
	logger *logrus.Logger
	mu     sync.RWMutex
}

// NewToolRegistry creates a new tool registry
func NewToolRegistry() *ToolRegistry {
	registry := &ToolRegistry{
		tools:  make(map[string]Tool),
		logger: logrus.New(),
	}

	// Register default tools
	registry.registerDefaultTools()

	return registry
}

// RegisterTool registers a tool
func (tr *ToolRegistry) RegisterTool(tool Tool) error {
	tr.mu.Lock()
	defer tr.mu.Unlock()

	name := tool.GetName()
	if _, exists := tr.tools[name]; exists {
		return fmt.Errorf("tool %s already registered", name)
	}

	tr.tools[name] = tool
	tr.logger.WithField("tool", name).Info("Tool registered")
	return nil
}

// UnregisterTool unregisters a tool
func (tr *ToolRegistry) UnregisterTool(name string) error {
	tr.mu.Lock()
	defer tr.mu.Unlock()

	if _, exists := tr.tools[name]; !exists {
		return fmt.Errorf("tool %s not found", name)
	}

	delete(tr.tools, name)
	tr.logger.WithField("tool", name).Info("Tool unregistered")
	return nil
}

// GetTool returns a tool by name
func (tr *ToolRegistry) GetTool(name string) (Tool, bool) {
	tr.mu.RLock()
	defer tr.mu.RUnlock()

	tool, exists := tr.tools[name]
	return tool, exists
}

// ListTools returns all registered tool names
func (tr *ToolRegistry) ListTools() []string {
	tr.mu.RLock()
	defer tr.mu.RUnlock()

	names := make([]string, 0, len(tr.tools))
	for name := range tr.tools {
		names = append(names, name)
	}
	return names
}

// GetAllDefinitions returns all tool definitions for LLM
func (tr *ToolRegistry) GetAllDefinitions() []llm.ToolDefinition {
	tr.mu.RLock()
	defer tr.mu.RUnlock()

	definitions := make([]llm.ToolDefinition, 0, len(tr.tools))
	for _, tool := range tr.tools {
		definitions = append(definitions, tool.GetDefinition())
	}
	return definitions
}

// GetDefinitions returns tool definitions for specific tools
func (tr *ToolRegistry) GetDefinitions(toolNames []string) []llm.ToolDefinition {
	tr.mu.RLock()
	defer tr.mu.RUnlock()

	definitions := make([]llm.ToolDefinition, 0, len(toolNames))
	for _, name := range toolNames {
		if tool, exists := tr.tools[name]; exists {
			definitions = append(definitions, tool.GetDefinition())
		}
	}
	return definitions
}

// registerDefaultTools registers default tools
func (tr *ToolRegistry) registerDefaultTools() {
	defaults := []Tool{
		NewWebSearchTool(),
		NewFileReadTool(), NewFileWriteTool(), NewFileListTool(),
		NewShellTool(),
		NewHTTPTool(),
		NewCalculatorTool(),
		NewTimeTool(),
	}

	for _, tool := range defaults {
		// Registration of the built-in set cannot legitimately fail; if it ever
		// does, the registry would silently be missing a tool callers expect.
		if err := tr.RegisterTool(tool); err != nil {
			tr.logger.WithError(err).Errorf("failed to register default tool %s", tool.GetName())
		}
	}
}

// WebSearchTool implements web search functionality
type WebSearchTool struct {
	apiKey string
	engine string
}

// NewWebSearchTool creates a new web search tool
func NewWebSearchTool() *WebSearchTool {
	return &WebSearchTool{
		apiKey: os.Getenv("SEARCH_API_KEY"),
		engine: "google", // Default engine
	}
}

func (t *WebSearchTool) GetName() string {
	return "web_search"
}

func (t *WebSearchTool) GetDescription() string {
	return "Search the web for information"
}

func (t *WebSearchTool) GetDefinition() llm.ToolDefinition {
	return llm.ToolDefinition{
		Type: "function",
		Function: llm.Function{
			Name:        t.GetName(),
			Description: t.GetDescription(),
			Parameters: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"query": map[string]interface{}{
						"type":        "string",
						"description": "The search query",
					},
					"num_results": map[string]interface{}{
						"type":        "integer",
						"description": "Number of results to return (default: 5)",
						"default":     5,
					},
				},
				"required": []string{"query"},
			},
		},
	}
}

func (t *WebSearchTool) Execute(ctx context.Context, args string) (string, error) {
	var params struct {
		Query      string `json:"query"`
		NumResults int    `json:"num_results"`
	}

	if err := json.Unmarshal([]byte(args), &params); err != nil {
		return "", fmt.Errorf("invalid arguments: %w", err)
	}

	if params.NumResults == 0 {
		params.NumResults = 5
	}

	// Simulate web search (in real implementation, use actual search API)
	results := fmt.Sprintf("Search results for '%s':\n", params.Query)
	for i := 1; i <= params.NumResults; i++ {
		results += fmt.Sprintf("%d. Sample result %d for query '%s'\n", i, i, params.Query)
	}

	return results, nil
}

func (t *WebSearchTool) Validate(args string) error {
	var params struct {
		Query string `json:"query"`
	}

	if err := json.Unmarshal([]byte(args), &params); err != nil {
		return fmt.Errorf("invalid JSON: %w", err)
	}

	if params.Query == "" {
		return fmt.Errorf("query is required")
	}

	return nil
}

func (t *WebSearchTool) GetConfig() map[string]interface{} {
	return map[string]interface{}{
		"api_key": t.apiKey,
		"engine":  t.engine,
	}
}

func (t *WebSearchTool) SetConfig(config map[string]interface{}) error {
	if apiKey, ok := config["api_key"].(string); ok {
		t.apiKey = apiKey
	}
	if engine, ok := config["engine"].(string); ok {
		t.engine = engine
	}
	return nil
}

// FileReadTool implements file reading functionality
type FileReadTool struct {
	maxFileSize int64
	allowedExts []string
	policy      *SecurityPolicy
}

// NewFileReadTool creates a new file read tool.
//
// Reads are confined to the policy's allowed roots. An extension allowlist on
// its own is not a boundary: ".json" and ".yaml" files outside the working
// directory include kubeconfigs, container registry credentials and cloud
// credential files.
func NewFileReadTool() *FileReadTool {
	return &FileReadTool{
		maxFileSize: 10 * 1024 * 1024, // 10MB
		allowedExts: []string{".txt", ".md", ".json", ".yaml", ".yml", ".csv"},
		policy:      DefaultSecurityPolicy(),
	}
}

// SetSecurityPolicy replaces the tool's security policy.
func (t *FileReadTool) SetSecurityPolicy(p *SecurityPolicy) {
	if p != nil {
		t.policy = p
	}
}

func (t *FileReadTool) GetName() string {
	return "file_read"
}

func (t *FileReadTool) GetDescription() string {
	return "Read the contents of a file"
}

func (t *FileReadTool) GetDefinition() llm.ToolDefinition {
	return llm.ToolDefinition{
		Type: "function",
		Function: llm.Function{
			Name:        t.GetName(),
			Description: t.GetDescription(),
			Parameters: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"file_path": map[string]interface{}{
						"type":        "string",
						"description": "Path to the file to read",
					},
				},
				"required": []string{"file_path"},
			},
		},
	}
}

func (t *FileReadTool) Execute(ctx context.Context, args string) (string, error) {
	var params struct {
		FilePath string `json:"file_path"`
	}

	if err := json.Unmarshal([]byte(args), &params); err != nil {
		return "", fmt.Errorf("invalid arguments: %w", err)
	}

	// Confine the path to the allowed roots before touching the filesystem.
	resolved, err := t.policy.ResolvePath(params.FilePath)
	if err != nil {
		return "", err
	}

	// Security check: ensure file extension is allowed
	ext := filepath.Ext(resolved)
	allowed := false
	for _, allowedExt := range t.allowedExts {
		if ext == allowedExt {
			allowed = true
			break
		}
	}
	if !allowed {
		return "", fmt.Errorf("file extension %s not allowed", ext)
	}

	// Check file size
	info, err := os.Stat(resolved)
	if err != nil {
		return "", fmt.Errorf("file not found: %w", err)
	}
	if !info.Mode().IsRegular() {
		return "", fmt.Errorf("path %q is not a regular file", params.FilePath)
	}

	if info.Size() > t.maxFileSize {
		return "", fmt.Errorf("file too large: %d bytes (max: %d)", info.Size(), t.maxFileSize)
	}

	content, err := os.ReadFile(resolved) // #nosec G304 -- resolved is confined to the policy roots
	if err != nil {
		return "", fmt.Errorf("failed to read file: %w", err)
	}

	out, truncated := t.policy.truncateOutput(content)
	if truncated {
		out += "\n[output truncated]"
	}
	return out, nil
}

func (t *FileReadTool) Validate(args string) error {
	var params struct {
		FilePath string `json:"file_path"`
	}

	if err := json.Unmarshal([]byte(args), &params); err != nil {
		return fmt.Errorf("invalid JSON: %w", err)
	}

	if params.FilePath == "" {
		return fmt.Errorf("file_path is required")
	}

	return nil
}

func (t *FileReadTool) GetConfig() map[string]interface{} {
	return map[string]interface{}{
		"max_file_size": t.maxFileSize,
		"allowed_exts":  t.allowedExts,
	}
}

func (t *FileReadTool) SetConfig(config map[string]interface{}) error {
	if maxSize, ok := config["max_file_size"].(int64); ok {
		t.maxFileSize = maxSize
	}
	if exts, ok := config["allowed_exts"].([]string); ok {
		t.allowedExts = exts
	}
	return nil
}

// FileWriteTool implements file writing functionality
type FileWriteTool struct {
	maxFileSize int64
	allowedExts []string
	policy      *SecurityPolicy
}

// NewFileWriteTool creates a new file write tool. Writes are confined to the
// policy's allowed roots.
func NewFileWriteTool() *FileWriteTool {
	return &FileWriteTool{
		maxFileSize: 10 * 1024 * 1024, // 10MB
		allowedExts: []string{".txt", ".md", ".json", ".yaml", ".yml", ".csv"},
		policy:      DefaultSecurityPolicy(),
	}
}

// SetSecurityPolicy replaces the tool's security policy.
func (t *FileWriteTool) SetSecurityPolicy(p *SecurityPolicy) {
	if p != nil {
		t.policy = p
	}
}

func (t *FileWriteTool) GetName() string {
	return "file_write"
}

func (t *FileWriteTool) GetDescription() string {
	return "Write content to a file"
}

func (t *FileWriteTool) GetDefinition() llm.ToolDefinition {
	return llm.ToolDefinition{
		Type: "function",
		Function: llm.Function{
			Name:        t.GetName(),
			Description: t.GetDescription(),
			Parameters: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"file_path": map[string]interface{}{
						"type":        "string",
						"description": "Path to the file to write",
					},
					"content": map[string]interface{}{
						"type":        "string",
						"description": "Content to write to the file",
					},
					"append": map[string]interface{}{
						"type":        "boolean",
						"description": "Whether to append to the file (default: false)",
						"default":     false,
					},
				},
				"required": []string{"file_path", "content"},
			},
		},
	}
}

func (t *FileWriteTool) Execute(ctx context.Context, args string) (string, error) {
	var params struct {
		FilePath string `json:"file_path"`
		Content  string `json:"content"`
		Append   bool   `json:"append"`
	}

	if err := json.Unmarshal([]byte(args), &params); err != nil {
		return "", fmt.Errorf("invalid arguments: %w", err)
	}

	// Confine the path to the allowed roots before touching the filesystem.
	resolved, err := t.policy.ResolvePath(params.FilePath)
	if err != nil {
		return "", err
	}
	params.FilePath = resolved

	// Security check: ensure file extension is allowed
	ext := filepath.Ext(resolved)
	allowed := false
	for _, allowedExt := range t.allowedExts {
		if ext == allowedExt {
			allowed = true
			break
		}
	}
	if !allowed {
		return "", fmt.Errorf("file extension %s not allowed", ext)
	}

	// Check content size
	if int64(len(params.Content)) > t.maxFileSize {
		return "", fmt.Errorf("content too large: %d bytes (max: %d)", len(params.Content), t.maxFileSize)
	}

	// Create directory if it doesn't exist
	dir := filepath.Dir(params.FilePath)
	if err := os.MkdirAll(dir, 0750); err != nil {
		return "", fmt.Errorf("failed to create directory: %w", err)
	}

	if params.Append {
		err = appendToFile(params.FilePath, params.Content)
	} else {
		err = os.WriteFile(params.FilePath, []byte(params.Content), 0600)
	}

	if err != nil {
		return "", fmt.Errorf("failed to write file: %w", err)
	}

	return fmt.Sprintf("Successfully wrote %d bytes to %s", len(params.Content), params.FilePath), nil
}

func (t *FileWriteTool) Validate(args string) error {
	var params struct {
		FilePath string `json:"file_path"`
		Content  string `json:"content"`
	}

	if err := json.Unmarshal([]byte(args), &params); err != nil {
		return fmt.Errorf("invalid JSON: %w", err)
	}

	if params.FilePath == "" {
		return fmt.Errorf("file_path is required")
	}

	return nil
}

func (t *FileWriteTool) GetConfig() map[string]interface{} {
	return map[string]interface{}{
		"max_file_size": t.maxFileSize,
		"allowed_exts":  t.allowedExts,
	}
}

func (t *FileWriteTool) SetConfig(config map[string]interface{}) error {
	if maxSize, ok := config["max_file_size"].(int64); ok {
		t.maxFileSize = maxSize
	}
	if exts, ok := config["allowed_exts"].([]string); ok {
		t.allowedExts = exts
	}
	return nil
}

// Helper function for appending to file
func appendToFile(filePath, content string) (err error) {
	file, err := os.OpenFile(filePath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0600) // #nosec G304 -- path is confined by the security policy
	if err != nil {
		return err
	}
	defer func() {
		// A buffered write can fail at Close; discarding that error would
		// report a successful append that never reached the file.
		if cerr := file.Close(); cerr != nil && err == nil {
			err = cerr
		}
	}()

	_, err = file.WriteString(content)
	return err
}

// FileListTool implements directory listing functionality
type FileListTool struct {
	maxItems int
	policy   *SecurityPolicy
}

// NewFileListTool creates a new file list tool. Listing is confined to the
// policy's allowed roots.
func NewFileListTool() *FileListTool {
	return &FileListTool{
		maxItems: 100,
		policy:   DefaultSecurityPolicy(),
	}
}

// SetSecurityPolicy replaces the tool's security policy.
func (t *FileListTool) SetSecurityPolicy(p *SecurityPolicy) {
	if p != nil {
		t.policy = p
	}
}

func (t *FileListTool) GetName() string {
	return "file_list"
}

func (t *FileListTool) GetDescription() string {
	return "List files and directories in a given path"
}

func (t *FileListTool) GetDefinition() llm.ToolDefinition {
	return llm.ToolDefinition{
		Type: "function",
		Function: llm.Function{
			Name:        t.GetName(),
			Description: t.GetDescription(),
			Parameters: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"path": map[string]interface{}{
						"type":        "string",
						"description": "Path to list (default: current directory)",
						"default":     ".",
					},
					"recursive": map[string]interface{}{
						"type":        "boolean",
						"description": "Whether to list recursively (default: false)",
						"default":     false,
					},
				},
			},
		},
	}
}

func (t *FileListTool) Execute(ctx context.Context, args string) (string, error) {
	var params struct {
		Path      string `json:"path"`
		Recursive bool   `json:"recursive"`
	}

	if err := json.Unmarshal([]byte(args), &params); err != nil {
		return "", fmt.Errorf("invalid arguments: %w", err)
	}

	if params.Path == "" {
		params.Path = "."
	}

	resolved, err := t.policy.ResolvePath(params.Path)
	if err != nil {
		return "", err
	}
	params.Path = resolved

	var result strings.Builder
	count := 0

	if params.Recursive {
		err := filepath.Walk(params.Path, func(path string, info os.FileInfo, err error) error {
			if err != nil {
				return err
			}

			if count >= t.maxItems {
				return filepath.SkipDir
			}

			fileType := "file"
			if info.IsDir() {
				fileType = "directory"
			}

			result.WriteString(fmt.Sprintf("%s (%s, %d bytes, %s)\n",
				path, fileType, info.Size(), info.ModTime().Format("2006-01-02 15:04:05")))
			count++

			return nil
		})

		if err != nil {
			return "", fmt.Errorf("failed to walk directory: %w", err)
		}
	} else {
		entries, err := os.ReadDir(params.Path)
		if err != nil {
			return "", fmt.Errorf("failed to read directory: %w", err)
		}

		for i, entry := range entries {
			if i >= t.maxItems {
				break
			}

			info, err := entry.Info()
			if err != nil {
				continue
			}

			fileType := "file"
			if entry.IsDir() {
				fileType = "directory"
			}

			result.WriteString(fmt.Sprintf("%s (%s, %d bytes, %s)\n",
				entry.Name(), fileType, info.Size(), info.ModTime().Format("2006-01-02 15:04:05")))
		}
	}

	return result.String(), nil
}

func (t *FileListTool) Validate(args string) error {
	// Path is optional, so no validation needed
	return nil
}

func (t *FileListTool) GetConfig() map[string]interface{} {
	return map[string]interface{}{
		"max_items": t.maxItems,
	}
}

func (t *FileListTool) SetConfig(config map[string]interface{}) error {
	if maxItems, ok := config["max_items"].(int); ok {
		t.maxItems = maxItems
	}
	return nil
}

// ShellTool implements shell command execution
type ShellTool struct {
	allowedCommands []string
	timeout         time.Duration
	policy          *SecurityPolicy
}

// NewShellTool creates a new shell tool.
//
// The previous default allowlist included "find", "cat" and "grep". A command
// allowlist cannot contain those: "find -exec" runs arbitrary programs, and
// "cat"/"grep" read any file the process can reach, so the allowlist provided
// no real boundary. The default set is now restricted, arguments that can
// execute other programs are rejected, and output is capped.
func NewShellTool() *ShellTool {
	policy := DefaultSecurityPolicy()
	return &ShellTool{
		allowedCommands: policy.AllowedCommands,
		timeout:         30 * time.Second,
		policy:          policy,
	}
}

// SetSecurityPolicy replaces the tool's security policy.
func (t *ShellTool) SetSecurityPolicy(p *SecurityPolicy) {
	if p != nil {
		t.policy = p
		t.allowedCommands = p.AllowedCommands
	}
}

func (t *ShellTool) GetName() string {
	return "shell"
}

func (t *ShellTool) GetDescription() string {
	return "Execute shell commands (limited for security)"
}

func (t *ShellTool) GetDefinition() llm.ToolDefinition {
	return llm.ToolDefinition{
		Type: "function",
		Function: llm.Function{
			Name:        t.GetName(),
			Description: t.GetDescription(),
			Parameters: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"command": map[string]interface{}{
						"type":        "string",
						"description": "Shell command to execute",
					},
				},
				"required": []string{"command"},
			},
		},
	}
}

func (t *ShellTool) Execute(ctx context.Context, args string) (string, error) {
	var params struct {
		Command string `json:"command"`
	}

	if err := json.Unmarshal([]byte(args), &params); err != nil {
		return "", fmt.Errorf("invalid arguments: %w", err)
	}

	// Security check: allowlisted command, with arguments that cannot be used
	// to execute something else.
	commandParts := strings.Fields(params.Command)
	if err := t.policy.CheckCommand(commandParts); err != nil {
		return "", err
	}

	// Create context with timeout
	ctx, cancel := context.WithTimeout(ctx, t.timeout)
	defer cancel()

	// Execute command. The command is run directly rather than through a shell,
	// so shell metacharacters are inert.
	cmd := exec.CommandContext(ctx, commandParts[0], commandParts[1:]...) // #nosec G204 -- command is allowlisted and arguments are validated
	output, err := cmd.CombinedOutput()
	text, truncated := t.policy.truncateOutput(output)
	if truncated {
		text += "\n[output truncated]"
	}
	if err != nil {
		return "", fmt.Errorf("command failed: %w\nOutput: %s", err, text)
	}

	return text, nil
}

func (t *ShellTool) Validate(args string) error {
	var params struct {
		Command string `json:"command"`
	}

	if err := json.Unmarshal([]byte(args), &params); err != nil {
		return fmt.Errorf("invalid JSON: %w", err)
	}

	if params.Command == "" {
		return fmt.Errorf("command is required")
	}

	return nil
}

func (t *ShellTool) GetConfig() map[string]interface{} {
	return map[string]interface{}{
		"allowed_commands": t.allowedCommands,
		"timeout":          t.timeout,
	}
}

func (t *ShellTool) SetConfig(config map[string]interface{}) error {
	if commands, ok := config["allowed_commands"].([]string); ok {
		t.allowedCommands = commands
	}
	if timeout, ok := config["timeout"].(time.Duration); ok {
		t.timeout = timeout
	}
	return nil
}

// HTTPTool implements HTTP request functionality
type HTTPTool struct {
	timeout time.Duration
	client  *http.Client
	policy  *SecurityPolicy
}

// NewHTTPTool creates a new HTTP tool.
//
// The request URL comes from a language model, so it is treated as untrusted:
// requests to loopback, private and link-local addresses are refused by
// default, which blocks server-side request forgery against internal services
// and the cloud instance metadata endpoint. The check is applied at dial time
// and on every redirect hop, so a hostname that resolves to an internal address
// is caught even if it resolved elsewhere a moment earlier.
func NewHTTPTool() *HTTPTool {
	timeout := 30 * time.Second
	policy := DefaultSecurityPolicy()
	return &HTTPTool{
		timeout: timeout,
		client:  policy.HTTPClient(timeout),
		policy:  policy,
	}
}

// SetSecurityPolicy replaces the tool's security policy and rebuilds its client.
func (t *HTTPTool) SetSecurityPolicy(p *SecurityPolicy) {
	if p != nil {
		t.policy = p
		t.client = p.HTTPClient(t.timeout)
	}
}

func (t *HTTPTool) GetName() string {
	return "http_request"
}

func (t *HTTPTool) GetDescription() string {
	return "Make HTTP requests"
}

func (t *HTTPTool) GetDefinition() llm.ToolDefinition {
	return llm.ToolDefinition{
		Type: "function",
		Function: llm.Function{
			Name:        t.GetName(),
			Description: t.GetDescription(),
			Parameters: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"url": map[string]interface{}{
						"type":        "string",
						"description": "URL to request",
					},
					"method": map[string]interface{}{
						"type":        "string",
						"description": "HTTP method (default: GET)",
						"default":     "GET",
					},
					"headers": map[string]interface{}{
						"type":        "object",
						"description": "HTTP headers",
					},
					"body": map[string]interface{}{
						"type":        "string",
						"description": "Request body",
					},
				},
				"required": []string{"url"},
			},
		},
	}
}

func (t *HTTPTool) Execute(ctx context.Context, args string) (string, error) {
	var params struct {
		URL     string            `json:"url"`
		Method  string            `json:"method"`
		Headers map[string]string `json:"headers"`
		Body    string            `json:"body"`
	}

	if err := json.Unmarshal([]byte(args), &params); err != nil {
		return "", fmt.Errorf("invalid arguments: %w", err)
	}

	if params.Method == "" {
		params.Method = "GET"
	}
	switch strings.ToUpper(params.Method) {
	case http.MethodGet, http.MethodPost, http.MethodPut, http.MethodPatch,
		http.MethodDelete, http.MethodHead, http.MethodOptions:
		params.Method = strings.ToUpper(params.Method)
	default:
		return "", fmt.Errorf("HTTP method %q is not allowed", params.Method)
	}

	if _, err := t.policy.CheckURL(params.URL); err != nil {
		return "", err
	}

	var bodyReader io.Reader
	if params.Body != "" {
		bodyReader = strings.NewReader(params.Body)
	}

	req, err := http.NewRequestWithContext(ctx, params.Method, params.URL, bodyReader)
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}

	// Add headers
	for key, value := range params.Headers {
		req.Header.Set(key, value)
	}

	resp, err := t.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	// Cap the response so a large or endless body cannot exhaust memory.
	limit := t.policy.maxOutput()
	body, err := io.ReadAll(io.LimitReader(resp.Body, limit+1))
	if err != nil {
		return "", fmt.Errorf("failed to read response: %w", err)
	}
	text, truncated := t.policy.truncateOutput(body)
	if truncated {
		text += "\n[response truncated]"
	}

	result := fmt.Sprintf("Status: %d %s\nHeaders: %v\nBody: %s",
		resp.StatusCode, resp.Status, resp.Header, text)

	return result, nil
}

func (t *HTTPTool) Validate(args string) error {
	var params struct {
		URL string `json:"url"`
	}

	if err := json.Unmarshal([]byte(args), &params); err != nil {
		return fmt.Errorf("invalid JSON: %w", err)
	}

	if params.URL == "" {
		return fmt.Errorf("url is required")
	}

	return nil
}

func (t *HTTPTool) GetConfig() map[string]interface{} {
	return map[string]interface{}{
		"timeout": t.timeout,
	}
}

func (t *HTTPTool) SetConfig(config map[string]interface{}) error {
	if timeout, ok := config["timeout"].(time.Duration); ok {
		t.timeout = timeout
		t.client = t.policy.HTTPClient(timeout)
	}
	return nil
}

// CalculatorTool implements basic mathematical calculations
type CalculatorTool struct{}

// NewCalculatorTool creates a new calculator tool
func NewCalculatorTool() *CalculatorTool {
	return &CalculatorTool{}
}

func (t *CalculatorTool) GetName() string {
	return "calculator"
}

func (t *CalculatorTool) GetDescription() string {
	return "Perform basic mathematical calculations"
}

func (t *CalculatorTool) GetDefinition() llm.ToolDefinition {
	return llm.ToolDefinition{
		Type: "function",
		Function: llm.Function{
			Name:        t.GetName(),
			Description: t.GetDescription(),
			Parameters: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"expression": map[string]interface{}{
						"type":        "string",
						"description": "Mathematical expression to evaluate",
					},
				},
				"required": []string{"expression"},
			},
		},
	}
}

func (t *CalculatorTool) Execute(ctx context.Context, args string) (string, error) {
	var params struct {
		Expression string `json:"expression"`
	}

	if err := json.Unmarshal([]byte(args), &params); err != nil {
		return "", fmt.Errorf("invalid arguments: %w", err)
	}

	// Simple calculator - only supports basic operations for security
	result, err := t.evaluateExpression(params.Expression)
	if err != nil {
		return "", fmt.Errorf("calculation failed: %w", err)
	}

	return fmt.Sprintf("Result: %v", result), nil
}

func (t *CalculatorTool) evaluateExpression(expr string) (float64, error) {
	// Remove spaces
	expr = strings.ReplaceAll(expr, " ", "")

	// Simple evaluation for basic operations
	// This is a simplified implementation - in production, use a proper expression parser

	// Handle addition
	if strings.Contains(expr, "+") {
		parts := strings.Split(expr, "+")
		if len(parts) != 2 {
			return 0, fmt.Errorf("invalid expression")
		}
		a, err := strconv.ParseFloat(parts[0], 64)
		if err != nil {
			return 0, err
		}
		b, err := strconv.ParseFloat(parts[1], 64)
		if err != nil {
			return 0, err
		}
		return a + b, nil
	}

	// Handle subtraction
	if strings.Contains(expr, "-") && !strings.HasPrefix(expr, "-") {
		parts := strings.Split(expr, "-")
		if len(parts) != 2 {
			return 0, fmt.Errorf("invalid expression")
		}
		a, err := strconv.ParseFloat(parts[0], 64)
		if err != nil {
			return 0, err
		}
		b, err := strconv.ParseFloat(parts[1], 64)
		if err != nil {
			return 0, err
		}
		return a - b, nil
	}

	// Handle multiplication
	if strings.Contains(expr, "*") {
		parts := strings.Split(expr, "*")
		if len(parts) != 2 {
			return 0, fmt.Errorf("invalid expression")
		}
		a, err := strconv.ParseFloat(parts[0], 64)
		if err != nil {
			return 0, err
		}
		b, err := strconv.ParseFloat(parts[1], 64)
		if err != nil {
			return 0, err
		}
		return a * b, nil
	}

	// Handle division
	if strings.Contains(expr, "/") {
		parts := strings.Split(expr, "/")
		if len(parts) != 2 {
			return 0, fmt.Errorf("invalid expression")
		}
		a, err := strconv.ParseFloat(parts[0], 64)
		if err != nil {
			return 0, err
		}
		b, err := strconv.ParseFloat(parts[1], 64)
		if err != nil {
			return 0, err
		}
		if b == 0 {
			return 0, fmt.Errorf("division by zero")
		}
		return a / b, nil
	}

	// Single number
	return strconv.ParseFloat(expr, 64)
}

func (t *CalculatorTool) Validate(args string) error {
	var params struct {
		Expression string `json:"expression"`
	}

	if err := json.Unmarshal([]byte(args), &params); err != nil {
		return fmt.Errorf("invalid JSON: %w", err)
	}

	if params.Expression == "" {
		return fmt.Errorf("expression is required")
	}

	return nil
}

func (t *CalculatorTool) GetConfig() map[string]interface{} {
	return map[string]interface{}{}
}

func (t *CalculatorTool) SetConfig(config map[string]interface{}) error {
	return nil
}

// TimeTool implements time-related functionality
type TimeTool struct{}

// NewTimeTool creates a new time tool
func NewTimeTool() *TimeTool {
	return &TimeTool{}
}

func (t *TimeTool) GetName() string {
	return "time"
}

func (t *TimeTool) GetDescription() string {
	return "Get current time and date information"
}

func (t *TimeTool) GetDefinition() llm.ToolDefinition {
	return llm.ToolDefinition{
		Type: "function",
		Function: llm.Function{
			Name:        t.GetName(),
			Description: t.GetDescription(),
			Parameters: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"format": map[string]interface{}{
						"type":        "string",
						"description": "Time format (default: RFC3339)",
						"default":     "RFC3339",
					},
					"timezone": map[string]interface{}{
						"type":        "string",
						"description": "Timezone (default: UTC)",
						"default":     "UTC",
					},
				},
			},
		},
	}
}

func (t *TimeTool) Execute(ctx context.Context, args string) (string, error) {
	var params struct {
		Format   string `json:"format"`
		Timezone string `json:"timezone"`
	}

	if err := json.Unmarshal([]byte(args), &params); err != nil {
		return "", fmt.Errorf("invalid arguments: %w", err)
	}

	if params.Format == "" {
		params.Format = "RFC3339"
	}
	if params.Timezone == "" {
		params.Timezone = "UTC"
	}

	// Load timezone
	loc, err := time.LoadLocation(params.Timezone)
	if err != nil {
		return "", fmt.Errorf("invalid timezone: %w", err)
	}

	now := time.Now().In(loc)

	var formatted string
	switch params.Format {
	case "RFC3339":
		formatted = now.Format(time.RFC3339)
	case "RFC822":
		formatted = now.Format(time.RFC822)
	case "Kitchen":
		formatted = now.Format(time.Kitchen)
	case "Stamp":
		formatted = now.Format(time.Stamp)
	case "Unix":
		formatted = fmt.Sprintf("%d", now.Unix())
	default:
		// Try as custom format
		formatted = now.Format(params.Format)
	}

	return fmt.Sprintf("Current time: %s (timezone: %s)", formatted, params.Timezone), nil
}

func (t *TimeTool) Validate(args string) error {
	// All parameters are optional
	return nil
}

func (t *TimeTool) GetConfig() map[string]interface{} {
	return map[string]interface{}{}
}

func (t *TimeTool) SetConfig(config map[string]interface{}) error {
	return nil
}
