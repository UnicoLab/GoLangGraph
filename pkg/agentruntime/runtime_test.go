// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package agentruntime_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/UnicoLab/GoLangGraph/pkg/agentruntime"
	"github.com/UnicoLab/GoLangGraph/pkg/agentspec"
	"github.com/UnicoLab/GoLangGraph/pkg/llm"
	"github.com/UnicoLab/GoLangGraph/pkg/persistence"
)

type completionFunc func(context.Context, string, llm.CompletionRequest) (*llm.CompletionResponse, error)

func (fn completionFunc) Complete(ctx context.Context, provider string, req llm.CompletionRequest) (*llm.CompletionResponse, error) {
	return fn(ctx, provider, req)
}

type input struct {
	Text string `json:"text"`
}
type output struct {
	Count int `json:"count"`
}

func counterPackage(t *testing.T, module, version string, limit int) agentspec.Package {
	t.Helper()
	tool, err := agentspec.NewTool("increment", "Increment this thread's counter", func(_ context.Context, state agentspec.State, _ struct{}) (output, error) {
		count := 0
		if data, ok := state.GetValue("count"); ok {
			if err := json.Unmarshal(data, &count); err != nil {
				return output{}, err
			}
			data[0] = '9' // GetValue must not expose the stored byte slice.
		}
		count++
		data, err := json.Marshal(count)
		if err != nil {
			return output{}, err
		}
		if err := state.SetValue("count", data); err != nil {
			return output{}, err
		}
		data[0] = '9' // SetValue must also take ownership of a copy.
		if err := state.SetValue("large", json.RawMessage(`18446744073709551615`)); err != nil {
			return output{}, err
		}
		return output{count}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	definition, err := agentspec.NewAgent[input, output](agentspec.AgentConfig{
		Name: "counter", Description: "Per-thread counter", MaxModelCalls: limit,
		Prompt: agentspec.Prompt{System: "Use the counter", Model: agentspec.Model{Provider: "test", Name: "test-model", MaxTokens: 100}},
		Tools:  []agentspec.Tool{tool},
	})
	if err != nil {
		t.Fatal(err)
	}
	pkg, err := agentspec.NewPackage(module, version, "Counter package", definition)
	if err != nil {
		t.Fatal(err)
	}
	return pkg
}

func reply(message llm.Message) *llm.CompletionResponse {
	return &llm.CompletionResponse{Choices: []llm.Choice{{Message: message}}}
}

func counterProvider(_ context.Context, _ string, req llm.CompletionRequest) (*llm.CompletionResponse, error) {
	last := req.Messages[len(req.Messages)-1]
	if last.Role == "tool" {
		return reply(llm.Message{Content: last.Content}), nil
	}
	return reply(llm.Message{ToolCalls: []llm.ToolCall{{ID: fmt.Sprint(len(req.Messages)), Type: "function", Function: llm.FunctionCall{Name: "increment", Arguments: `{}`}}}}), nil
}

func newRuntime(t *testing.T, config agentruntime.Config, packages ...agentspec.Package) *agentruntime.Runtime {
	t.Helper()
	if config.Providers == nil {
		config.Providers = completionFunc(counterProvider)
	}
	runtime, err := agentruntime.New(config)
	if err != nil {
		t.Fatal(err)
	}
	if err := runtime.Register(packages...); err != nil {
		t.Fatal(err)
	}
	return runtime
}

func createThread(t *testing.T, runtime *agentruntime.Runtime, id string) *agentruntime.Thread {
	t.Helper()
	thread, err := runtime.CreateThread(context.Background(), id)
	if err != nil {
		t.Fatal(err)
	}
	return thread
}

func TestIndependentThreadsContinueAfterReopeningFileStore(t *testing.T) {
	ctx := context.Background()
	pkg := counterPackage(t, "example.com/counter", "v0.1.0", 4)
	dir := t.TempDir()
	runtime := newRuntime(t, agentruntime.Config{Checkpointer: persistence.NewFileCheckpointer(dir)}, pkg)
	first := createThread(t, runtime, pkg.AgentID("counter"))
	second := createThread(t, runtime, pkg.AgentID("counter"))
	for _, thread := range []*agentruntime.Thread{first, second, first} {
		if _, err := runtime.Run(ctx, thread.ID, input{"hello"}); err != nil {
			t.Fatal(err)
		}
	}
	// Drop the runtime and checkpoint backend; restore only from files.
	restarted := newRuntime(t, agentruntime.Config{Checkpointer: persistence.NewFileCheckpointer(dir)}, pkg)
	answer, err := agentruntime.RunTyped[input, output](ctx, restarted, first.ID, input{"continue"})
	if err != nil || answer.Count != 3 {
		t.Fatalf("restored count=%d, err=%v", answer.Count, err)
	}
	snapshot, err := restarted.ReadThread(ctx, first.ID)
	if err != nil {
		t.Fatal(err)
	}
	if snapshot.CompletedRuns != 3 || len(snapshot.Messages) != 12 || string(snapshot.Values["large"]) != "18446744073709551615" {
		t.Fatalf("history or state did not round-trip: %+v", snapshot)
	}
	snapshot.Values["count"][0] = '9'
	snapshot.Messages[0].Content = "corrupted"
	other, err := restarted.ReadThread(ctx, second.ID)
	if err != nil || string(other.Values["count"]) != "1" || other.CompletedRuns != 1 {
		t.Fatalf("thread state leaked: %+v, %v", other, err)
	}
	answer, err = agentruntime.RunTyped[input, output](ctx, restarted, first.ID, input{"again"})
	if err != nil || answer.Count != 4 {
		t.Fatalf("snapshot mutation affected store: %+v, %v", answer, err)
	}
}

func TestConcurrentTurnsAreSerializedPerThread(t *testing.T) {
	pkg := counterPackage(t, "example.com/counter", "v0.1.0", 4)
	runtime := newRuntime(t, agentruntime.Config{}, pkg)
	thread := createThread(t, runtime, pkg.AgentID("counter"))
	var wg sync.WaitGroup
	counts := make(chan int, 12)
	errors := make(chan error, 12)
	for i := 0; i < 12; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			answer, err := agentruntime.RunTyped[input, output](context.Background(), runtime, thread.ID, input{"hello"})
			if err != nil {
				errors <- err
				return
			}
			counts <- answer.Count
		}()
	}
	wg.Wait()
	close(counts)
	close(errors)
	for err := range errors {
		t.Fatal(err)
	}
	var got []int
	for count := range counts {
		got = append(got, count)
	}
	sort.Ints(got)
	for i, count := range got {
		if count != i+1 {
			t.Fatalf("lost a concurrent turn: %v", got)
		}
	}
	if len(got) != 12 {
		t.Fatalf("missing results: %v", got)
	}
}

func TestDifferentThreadsRunConcurrentlyAndWaitersCanCancel(t *testing.T) {
	pkg := counterPackage(t, "example.com/counter", "v0.1.0", 4)
	entered := make(chan struct{}, 2)
	release := make(chan struct{})
	provider := completionFunc(func(ctx context.Context, name string, req llm.CompletionRequest) (*llm.CompletionResponse, error) {
		if req.Messages[len(req.Messages)-1].Role == "user" {
			entered <- struct{}{}
			select {
			case <-release:
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}
		return counterProvider(ctx, name, req)
	})
	runtime := newRuntime(t, agentruntime.Config{Providers: provider}, pkg)
	first := createThread(t, runtime, pkg.AgentID("counter"))
	second := createThread(t, runtime, pkg.AgentID("counter"))
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	finished := make(chan error, 2)
	for _, id := range []string{first.ID, second.ID} {
		go func(id string) { _, err := runtime.Run(ctx, id, input{"hello"}); finished <- err }(id)
	}
	for i := 0; i < 2; i++ {
		select {
		case <-entered:
		case <-ctx.Done():
			t.Fatal("independent threads blocked one another")
		}
	}
	waitCtx, waitCancel := context.WithTimeout(ctx, 20*time.Millisecond)
	defer waitCancel()
	if _, err := runtime.Run(waitCtx, first.ID, input{"waiting"}); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("queued caller could not cancel: %v", err)
	}
	close(release)
	for i := 0; i < 2; i++ {
		if err := <-finished; err != nil {
			t.Fatal(err)
		}
	}
}

type failingSaver struct {
	persistence.Checkpointer
	fail bool
}

var errSave = errors.New("storage unavailable")

func (s *failingSaver) Save(ctx context.Context, cp *persistence.Checkpoint) error {
	if s.fail {
		return errSave
	}
	return s.Checkpointer.Save(ctx, cp)
}

func TestValidationAndStorageFailuresNeverReportCommittedTurns(t *testing.T) {
	pkg := counterPackage(t, "example.com/counter", "v0.1.0", 4)
	store := &failingSaver{Checkpointer: persistence.NewMemoryCheckpointer()}
	var calls atomic.Int32
	runtime := newRuntime(t, agentruntime.Config{Checkpointer: store, Providers: completionFunc(func(ctx context.Context, name string, req llm.CompletionRequest) (*llm.CompletionResponse, error) {
		calls.Add(1)
		return counterProvider(ctx, name, req)
	})}, pkg)
	thread := createThread(t, runtime, pkg.AgentID("counter"))
	if _, err := runtime.Run(context.Background(), thread.ID, map[string]any{"unknown": true}); err == nil || calls.Load() != 0 {
		t.Fatalf("invalid input invoked provider: %v", err)
	}
	store.fail = true
	if _, err := runtime.Run(context.Background(), thread.ID, input{"hello"}); !errors.Is(err, errSave) {
		t.Fatalf("checkpoint error swallowed: %v", err)
	}
	store.fail = false
	snapshot, err := runtime.ReadThread(context.Background(), thread.ID)
	if err != nil || snapshot.CompletedRuns != 0 || len(snapshot.Messages) != 0 || len(snapshot.Values) != 0 {
		t.Fatalf("failed commit changed history: %+v, %v", snapshot, err)
	}
	if _, err := runtime.Run(context.Background(), "missing", input{"hello"}); err == nil {
		t.Fatal("missing thread silently created")
	}
}

func TestModelLimitAndInvalidOutputDoNotCommit(t *testing.T) {
	pkg := counterPackage(t, "example.com/counter", "v0.1.0", 2)
	for _, test := range []struct {
		name     string
		provider completionFunc
		want     error
	}{
		{"limit", func(_ context.Context, _ string, _ llm.CompletionRequest) (*llm.CompletionResponse, error) {
			return reply(llm.Message{ToolCalls: []llm.ToolCall{{ID: "call", Function: llm.FunctionCall{Name: "increment", Arguments: `{}`}}}}), nil
		}, agentruntime.ErrModelCallLimit},
		{"output", func(_ context.Context, _ string, _ llm.CompletionRequest) (*llm.CompletionResponse, error) {
			return reply(llm.Message{Content: `{"wrong":"shape"}`}), nil
		}, nil},
	} {
		t.Run(test.name, func(t *testing.T) {
			runtime := newRuntime(t, agentruntime.Config{Providers: test.provider}, pkg)
			thread := createThread(t, runtime, pkg.AgentID("counter"))
			_, err := runtime.Run(context.Background(), thread.ID, input{"hello"})
			if err == nil || (test.want != nil && !errors.Is(err, test.want)) {
				t.Fatalf("unexpected error: %v", err)
			}
			snapshot, err := runtime.ReadThread(context.Background(), thread.ID)
			if err != nil || snapshot.CompletedRuns != 0 || len(snapshot.Values) != 0 {
				t.Fatalf("failed run persisted: %+v, %v", snapshot, err)
			}
		})
	}
}

func TestUndeclaredToolsAreObservationsAndCannotAccessHostTools(t *testing.T) {
	pkg := counterPackage(t, "example.com/counter", "v0.1.0", 4)
	provider := completionFunc(func(_ context.Context, _ string, req llm.CompletionRequest) (*llm.CompletionResponse, error) {
		last := req.Messages[len(req.Messages)-1]
		if last.Role == "tool" {
			if !strings.Contains(last.Content, "not declared") {
				return nil, fmt.Errorf("missing tool error observation")
			}
			return reply(llm.Message{Content: `{"count":0}`}), nil
		}
		return reply(llm.Message{ToolCalls: []llm.ToolCall{{ID: "unavailable", Function: llm.FunctionCall{Name: "shell", Arguments: `{}`}}}}), nil
	})
	runtime := newRuntime(t, agentruntime.Config{Providers: provider}, pkg)
	thread := createThread(t, runtime, pkg.AgentID("counter"))
	if _, err := runtime.Run(context.Background(), thread.ID, input{"hello"}); err != nil {
		t.Fatal(err)
	}
	snapshot, err := runtime.ReadThread(context.Background(), thread.ID)
	if err != nil || len(snapshot.Values) != 0 || snapshot.Messages[2].ToolCallID != "unavailable" {
		t.Fatalf("undeclared tool affected state or lost correlation: %+v, %v", snapshot, err)
	}
}

func TestRegistrationIsAtomicAndThreadsPinPackageVersionAndModel(t *testing.T) {
	pkg := counterPackage(t, "example.com/counter", "v0.1.0", 4)
	other := counterPackage(t, "example.com/other", "v0.1.0", 4)
	store := persistence.NewMemoryCheckpointer()
	overrides := map[string]agentspec.Model{pkg.AgentID("counter"): {Provider: "test", Name: "pinned-model", MaxTokens: 200}}
	runtime := newRuntime(t, agentruntime.Config{Checkpointer: store, Models: overrides}, pkg)
	delete(overrides, pkg.AgentID("counter"))
	if err := runtime.Register(other, pkg); err == nil || len(runtime.Catalog()) != 1 {
		t.Fatal("partially installed failed package batch")
	}
	thread := createThread(t, runtime, pkg.AgentID("counter"))
	if thread.Model.Name != "pinned-model" {
		t.Fatal("caller mutated runtime model configuration")
	}
	newVersion := counterPackage(t, "example.com/counter", "v0.2.0", 4)
	restarted := newRuntime(t, agentruntime.Config{Checkpointer: store}, newVersion)
	if _, err := restarted.Run(context.Background(), thread.ID, input{"hello"}); err == nil {
		t.Fatal("thread silently upgraded its package")
	}
	var observed string
	restored := newRuntime(t, agentruntime.Config{Checkpointer: store, Providers: completionFunc(func(ctx context.Context, provider string, req llm.CompletionRequest) (*llm.CompletionResponse, error) {
		observed = req.Model
		return counterProvider(ctx, provider, req)
	})}, pkg)
	if _, err := restored.Run(context.Background(), thread.ID, input{"hello"}); err != nil {
		t.Fatal(err)
	}
	if observed != "pinned-model" {
		t.Fatalf("restored thread switched model to %s", observed)
	}
}

type globalIDStore struct {
	persistence.Checkpointer
	owners map[string]string
}

func (s *globalIDStore) Save(ctx context.Context, checkpoint *persistence.Checkpoint) error {
	if owner, exists := s.owners[checkpoint.ID]; exists && owner != checkpoint.ThreadID {
		return fmt.Errorf("checkpoint ID collided across threads")
	}
	s.owners[checkpoint.ID] = checkpoint.ThreadID
	return s.Checkpointer.Save(ctx, checkpoint)
}

func TestCheckpointIDsSupportGloballyKeyedBackends(t *testing.T) {
	pkg := counterPackage(t, "example.com/counter", "v0.1.0", 4)
	store := &globalIDStore{Checkpointer: persistence.NewMemoryCheckpointer(), owners: make(map[string]string)}
	runtime := newRuntime(t, agentruntime.Config{Checkpointer: store}, pkg)
	first := createThread(t, runtime, pkg.AgentID("counter"))
	second := createThread(t, runtime, pkg.AgentID("counter"))
	for _, id := range []string{first.ID, second.ID, first.ID} {
		if _, err := runtime.Run(context.Background(), id, input{"hello"}); err != nil {
			t.Fatal(err)
		}
	}
	if len(store.owners) != 2 {
		t.Fatalf("expected one stable checkpoint per thread, got %v", store.owners)
	}
}

func TestMultipleToolsSeePriorWritesInDeclaredOrder(t *testing.T) {
	pkg := counterPackage(t, "example.com/counter", "v0.1.0", 4)
	provider := completionFunc(func(_ context.Context, _ string, req llm.CompletionRequest) (*llm.CompletionResponse, error) {
		last := req.Messages[len(req.Messages)-1]
		if last.Role == "tool" {
			previous := req.Messages[len(req.Messages)-2]
			if previous.Content != `{"count":1}` || last.Content != `{"count":2}` || previous.ToolCallID != "first" || last.ToolCallID != "second" {
				return nil, fmt.Errorf("tool ordering/state visibility violated: %+v, %+v", previous, last)
			}
			return reply(llm.Message{Content: last.Content}), nil
		}
		return reply(llm.Message{ToolCalls: []llm.ToolCall{
			{ID: "first", Function: llm.FunctionCall{Name: "increment", Arguments: `{}`}},
			{ID: "second", Function: llm.FunctionCall{Name: "increment", Arguments: `{}`}},
		}}), nil
	})
	runtime := newRuntime(t, agentruntime.Config{Providers: provider}, pkg)
	thread := createThread(t, runtime, pkg.AgentID("counter"))
	result, err := runtime.Run(context.Background(), thread.ID, input{"hello"})
	if err != nil || result.ToolCalls != 2 || string(result.Output) != `{"count":2}` {
		t.Fatalf("tool batch failed: %+v, %v", result, err)
	}
}
