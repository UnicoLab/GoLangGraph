// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package main

import (
	"context"
	"encoding/json"
	"strings"
	"sync"
	"testing"

	"github.com/UnicoLab/GoLangGraph/agents/designer"
	"github.com/UnicoLab/GoLangGraph/agents/storymaker"
	"github.com/UnicoLab/GoLangGraph/pkg/agentruntime"
	"github.com/UnicoLab/GoLangGraph/pkg/persistence"
)

func TestIndependentlyPackagedAgentsComposeAndResume(t *testing.T) {
	dir := t.TempDir()
	config := func() agentruntime.Config {
		return agentruntime.Config{Providers: demoProvider{}, Checkpointer: persistence.NewFileCheckpointer(dir)}
	}
	runtime, err := newRuntime(config())
	if err != nil {
		t.Fatal(err)
	}
	report, err := compose(context.Background(), runtime, "solar energy garden", "")
	if err != nil {
		t.Fatal(err)
	}
	if report.DesignerToolUses != 1 || !strings.Contains(report.Story.Story, report.Design.Description) || report.Design.Style != "futuristic" {
		t.Fatalf("package composition did not use design/tool results: %+v", report)
	}
	catalog := runtime.Catalog()
	if len(catalog) != 3 || catalog[0].Module != designer.Module || catalog[2].Module != storymaker.Module {
		t.Fatalf("incorrect catalog: %+v", catalog)
	}
	if _, err := json.Marshal(catalog); err != nil {
		t.Fatal(err)
	}
	// Recreate the host from persisted records, without carrying live instances.
	restarted, err := newRuntime(config())
	if err != nil {
		t.Fatal(err)
	}
	next, err := compose(context.Background(), restarted, "rainwater garden", report.DesignerThread)
	if err != nil {
		t.Fatal(err)
	}
	if next.DesignerThread != report.DesignerThread || next.DesignerToolUses != 2 || next.Design.Style != "modern" {
		t.Fatalf("failed continuation: %+v", next)
	}
	story, err := restarted.ReadThread(context.Background(), next.StorymakerThread)
	if err != nil || len(story.Values) != 0 || story.CompletedRuns != 1 {
		t.Fatalf("designer state leaked into storyteller: %+v, %v", story, err)
	}
}

func TestOneHostRunsIndependentCompositions(t *testing.T) {
	runtime, err := newRuntime(agentruntime.Config{Providers: demoProvider{}})
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	results := make(chan *Report, 6)
	errors := make(chan error, 6)
	for i := 0; i < 6; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			report, err := compose(context.Background(), runtime, "nature habitat", "")
			if err != nil {
				errors <- err
				return
			}
			results <- report
		}()
	}
	wg.Wait()
	close(results)
	close(errors)
	for err := range errors {
		t.Fatal(err)
	}
	seen := make(map[string]bool)
	for report := range results {
		if report.DesignerToolUses != 1 || seen[report.DesignerThread] {
			t.Fatalf("reused mutable agent instance: %+v", report)
		}
		seen[report.DesignerThread] = true
	}
	if len(seen) != 6 {
		t.Fatalf("only %d compositions completed", len(seen))
	}
}

func TestReusableHabitatPackageResumesAndDeduplicatesAfterRestart(t *testing.T) {
	dir := t.TempDir()
	open := func() *agentruntime.Runtime {
		r, err := newRuntime(agentruntime.Config{Providers: demoProvider{}, Checkpointer: persistence.NewFileCheckpointer(dir)})
		if err != nil {
			t.Fatal(err)
		}
		return r
	}
	ctx := context.Background()
	r := open()
	first, err := composePackage(ctx, r, "solar garden", "", "request-1")
	if err != nil {
		t.Fatal(err)
	}
	r = open()
	repeated, err := composePackage(ctx, r, "solar garden", first.HabitatThread, "request-1")
	if err != nil || repeated.DesignerToolUses != 1 || repeated.StorymakerThread != first.StorymakerThread {
		t.Fatalf("redelivery: %+v %v", repeated, err)
	}
	next, err := composePackage(ctx, r, "rainwater garden", first.HabitatThread, "request-2")
	if err != nil || next.DesignerToolUses != 2 || next.Design.Style != "modern" || next.DesignerThread != first.DesignerThread {
		t.Fatalf("continuation: %+v %v", next, err)
	}
	if !strings.Contains(next.Story.Story, next.Design.Description) || len(next.Events) != 6 {
		t.Fatalf("missing composition output or trace: %+v", next)
	}
	story, err := r.ReadThread(ctx, next.StorymakerThread)
	if err != nil || story.CompletedRuns != 2 || story.ParentID != next.HabitatThread {
		t.Fatalf("child history: %+v %v", story, err)
	}
}
