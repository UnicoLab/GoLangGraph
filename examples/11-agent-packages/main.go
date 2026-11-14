// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/UnicoLab/GoLangGraph/agents/designer"
	"github.com/UnicoLab/GoLangGraph/agents/habitat"
	"github.com/UnicoLab/GoLangGraph/agents/storymaker"
	"github.com/UnicoLab/GoLangGraph/pkg/agentruntime"
	"github.com/UnicoLab/GoLangGraph/pkg/agentspec"
	"github.com/UnicoLab/GoLangGraph/pkg/core"
	"github.com/UnicoLab/GoLangGraph/pkg/llm"
	"github.com/UnicoLab/GoLangGraph/pkg/persistence"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	live := flag.Bool("live", false, "Use a running Ollama server instead of the offline model double")
	endpoint := flag.String("endpoint", "http://localhost:11434", "Ollama endpoint for -live")
	model := flag.String("model", "", "Override the packages' default model for new threads")
	brief := flag.String("brief", "A small solar-powered habitat beside a community garden", "Design brief")
	stateDir := flag.String("state-dir", "", "Directory for durable threads; omitted uses memory")
	resume := flag.String("resume-designer", "", "Continue a designer thread from -state-dir")
	workflow := flag.String("workflow", "package", "Composition: package (persistent children) or graph (application-owned nodes)")
	resumeHabitat := flag.String("resume-habitat", "", "Continue a habitat thread from -state-dir")
	requestID := flag.String("request-id", "", "Stable request ID for deduplication in package mode")
	catalog := flag.Bool("catalog", false, "Print package manifests without executing agents")
	flag.Parse()
	if (*resume != "" || *resumeHabitat != "") && *stateDir == "" {
		return fmt.Errorf("resume requires -state-dir")
	}
	if *workflow != "package" && *workflow != "graph" {
		return fmt.Errorf("-workflow must be package or graph")
	}
	if (*workflow == "package" && *resume != "") || (*workflow == "graph" && (*resumeHabitat != "" || *requestID != "")) {
		return fmt.Errorf("resume/request flags must match the selected workflow")
	}
	var providers agentruntime.Completer = demoProvider{}
	mode := "offline model double; real package tools, graph and persistence"
	if *live {
		manager := llm.NewProviderManager()
		defer manager.Close()
		provider, err := llm.NewOllamaProvider(&llm.ProviderConfig{Endpoint: *endpoint, Timeout: 90 * time.Second})
		if err != nil {
			return err
		}
		if err := manager.RegisterProvider("ollama", provider); err != nil {
			return err
		}
		providers, mode = manager, "live Ollama"
	}
	var checkpointer persistence.Checkpointer = persistence.NewMemoryCheckpointer()
	if *stateDir != "" {
		checkpointer = persistence.NewFileCheckpointer(*stateDir)
	}
	defer checkpointer.Close()
	config := agentruntime.Config{Providers: providers, Checkpointer: checkpointer}
	if *model != "" {
		config.Models = map[string]agentspec.Model{
			designer.AgentID:   {Provider: "ollama", Name: *model, Temperature: 0.7, MaxTokens: 1500},
			storymaker.AgentID: {Provider: "ollama", Name: *model, Temperature: 0.7, MaxTokens: 1800},
		}
	}
	runtime, err := newRuntime(config)
	if err != nil {
		return err
	}
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetIndent("", "  ")
	if *catalog {
		return encoder.Encode(runtime.Catalog())
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	var report *Report
	if *workflow == "package" {
		report, err = composePackage(ctx, runtime, *brief, *resumeHabitat, *requestID)
	} else {
		report, err = compose(ctx, runtime, *brief, *resume)
	}
	if err != nil {
		return err
	}
	report.Mode = mode
	return encoder.Encode(report)
}

func newRuntime(config agentruntime.Config) (*agentruntime.Runtime, error) {
	designPackage, err := designer.Package()
	if err != nil {
		return nil, err
	}
	storyPackage, err := storymaker.Package()
	if err != nil {
		return nil, err
	}
	habitatPackage, err := habitat.Package()
	if err != nil {
		return nil, err
	}
	runtime, err := agentruntime.New(config)
	if err != nil {
		return nil, err
	}
	if err := runtime.Register(habitatPackage, designPackage, storyPackage); err != nil {
		return nil, err
	}
	return runtime, nil
}

type Report struct {
	Mode             string               `json:"mode"`
	HabitatThread    string               `json:"habitat_thread,omitempty"`
	RequestID        string               `json:"request_id,omitempty"`
	Events           []agentruntime.Event `json:"events,omitempty"`
	DesignerThread   string               `json:"designer_thread"`
	StorymakerThread string               `json:"storymaker_thread"`
	DesignerToolUses int                  `json:"designer_tool_uses"`
	Design           designer.Output      `json:"design"`
	Story            storymaker.Output    `json:"story"`
}

func composePackage(ctx context.Context, runtime *agentruntime.Runtime, brief, resume, requestID string) (*Report, error) {
	var thread *agentruntime.Thread
	var err error
	if resume == "" {
		thread, err = runtime.CreateThread(ctx, habitat.AgentID)
	} else {
		thread, err = runtime.ReadThread(ctx, resume)
	}
	if err != nil {
		return nil, err
	}
	if thread.AgentID != habitat.AgentID {
		return nil, fmt.Errorf("resume thread must belong to habitat")
	}
	result, err := runtime.RunWithOptions(ctx, thread.ID, habitat.Input{Message: brief}, agentruntime.RunOptions{RequestID: requestID})
	if err != nil {
		return nil, err
	}
	var output habitat.Output
	if err := json.Unmarshal(result.Output, &output); err != nil {
		return nil, err
	}
	report := &Report{HabitatThread: thread.ID, RequestID: result.RequestID, DesignerThread: thread.Children["designer"], StorymakerThread: thread.Children["storymaker"], Design: output.Design, Story: output.Story}
	parent, err := runtime.ReadThread(ctx, thread.ID)
	if err != nil {
		return nil, err
	}
	report.Events = parent.Runs[result.RequestID].Events
	design, err := runtime.ReadThread(ctx, report.DesignerThread)
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(design.Values["palette_uses"], &report.DesignerToolUses); err != nil {
		return nil, err
	}
	return report, nil
}

// The consumer owns the workflow. Neither agent module imports the other, and
// neither module imports core.Graph. Data crosses the boundary through types.
func compose(ctx context.Context, runtime *agentruntime.Runtime, brief, resume string) (*Report, error) {
	var designThread *agentruntime.Thread
	var err error
	if resume != "" {
		designThread, err = runtime.ReadThread(ctx, resume)
		if err == nil && designThread.AgentID != designer.AgentID {
			return nil, fmt.Errorf("resume thread must belong to the designer")
		}
	} else {
		designThread, err = runtime.CreateThread(ctx, designer.AgentID)
	}
	if err != nil {
		return nil, err
	}
	storyThread, err := runtime.CreateThread(ctx, storymaker.AgentID)
	if err != nil {
		return nil, err
	}
	report := &Report{DesignerThread: designThread.ID, StorymakerThread: storyThread.ID}
	graph := core.NewGraph("packaged-habitat-workflow")
	defer graph.Close()
	graph.Config.Timeout = 0
	graph.AddNode("design", "Design habitat", func(ctx context.Context, state *core.BaseState) (*core.BaseState, error) {
		output, err := agentruntime.RunTyped[designer.Input, designer.Output](ctx, runtime, designThread.ID, designer.Input{Message: brief})
		if err != nil {
			return nil, err
		}
		state.Set("design", output)
		report.Design = output
		return state, nil
	})
	graph.AddNode("story", "Tell its story", func(ctx context.Context, state *core.BaseState) (*core.BaseState, error) {
		value, _ := state.Get("design")
		design, ok := value.(designer.Output)
		if !ok {
			return nil, fmt.Errorf("design node did not produce a typed design")
		}
		output, err := agentruntime.RunTyped[storymaker.Input, storymaker.Output](ctx, runtime, storyThread.ID, storymaker.Input{
			StoryPrompt: "A resident's first morning in the habitat", DesignDescription: design.Description,
		})
		if err != nil {
			return nil, err
		}
		report.Story = output
		return state, nil
	})
	graph.AddEdge("design", "story", nil)
	if err := graph.SetStartNode("design"); err != nil {
		return nil, err
	}
	if err := graph.AddEndNode("story"); err != nil {
		return nil, err
	}
	if _, err := graph.Execute(ctx, core.NewBaseState()); err != nil {
		return nil, err
	}
	snapshot, err := runtime.ReadThread(ctx, designThread.ID)
	if err != nil {
		return nil, err
	}
	if value, exists := snapshot.Values["palette_uses"]; exists {
		if err := json.Unmarshal(value, &report.DesignerToolUses); err != nil {
			return nil, err
		}
	}
	return report, nil
}
