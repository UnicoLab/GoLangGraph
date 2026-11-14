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
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/UnicoLab/GoLangGraph/pkg/agentruntime"
	"github.com/UnicoLab/GoLangGraph/pkg/agentspec"
	"github.com/UnicoLab/GoLangGraph/pkg/llm"
	"github.com/UnicoLab/GoLangGraph/pkg/persistence"
)

func handlerPackage(t *testing.T, module string, children []agentspec.Subagent, fn func(context.Context, agentspec.Execution, input) (output, error)) agentspec.Package {
	t.Helper()
	definition, err := agentspec.NewHandler[input, output](agentspec.HandlerConfig{Name: "run", Description: "Run handler", Subagents: children}, fn)
	if err != nil {
		t.Fatal(err)
	}
	pkg, err := agentspec.NewPackage(module, "v0.1.0", "Test handler", definition)
	if err != nil {
		t.Fatal(err)
	}
	return pkg
}

func TestHandlerCompositionPersistsIsolatedChildrenAndRejectsWrongTypes(t *testing.T) {
	ctx := context.Background()
	child := handlerPackage(t, "example.com/child", nil, func(_ context.Context, s agentspec.Execution, _ input) (output, error) {
		count := 0
		if data, ok := s.GetValue("count"); ok {
			if err := json.Unmarshal(data, &count); err != nil {
				return output{}, err
			}
		}
		count++
		data, _ := json.Marshal(count)
		return output{count}, s.SetValue("count", data)
	})
	ref, err := agentspec.NewSubagent[input, output]("worker", "Count independently", child.AgentID("run"), "v0.1.0")
	if err != nil {
		t.Fatal(err)
	}
	parent := handlerPackage(t, "example.com/parent", []agentspec.Subagent{ref}, func(ctx context.Context, s agentspec.Execution, i input) (output, error) {
		if _, err := s.InvokeAgent(ctx, "undeclared", json.RawMessage(`{"text":"hello"}`)); err == nil {
			return output{}, errors.New("undeclared child allowed")
		}
		return agentspec.CallAgent[input, output](ctx, s, "worker", i)
	})
	dir := t.TempDir()
	open := func() *agentruntime.Runtime {
		r, err := agentruntime.New(agentruntime.Config{Checkpointer: persistence.NewFileCheckpointer(dir)}) // no model
		if err != nil {
			t.Fatal(err)
		}
		if err := r.Register(parent, child); err != nil {
			t.Fatal(err)
		} // parent-first batch works
		return r
	}
	r := open()
	a, b := createThread(t, r, parent.AgentID("run")), createThread(t, r, parent.AgentID("run"))
	if a.Children["worker"] == b.Children["worker"] {
		t.Fatal("shared child instance")
	}
	for _, thread := range []*agentruntime.Thread{a, b} {
		out, err := agentruntime.RunTyped[input, output](ctx, r, thread.ID, input{"hi"})
		if err != nil || out.Count != 1 {
			t.Fatalf("first run: %+v %v", out, err)
		}
	}
	r = open()
	out, err := agentruntime.RunTyped[input, output](ctx, r, a.ID, input{"again"})
	if err != nil || out.Count != 2 {
		t.Fatalf("persistent child: %+v %v", out, err)
	}
	childState, err := r.ReadThread(ctx, a.Children["worker"])
	if err != nil || childState.ParentID != a.ID || childState.ChildName != "worker" || childState.CompletedRuns != 2 {
		t.Fatalf("lineage: %+v %v", childState, err)
	}
	before, _ := r.ReadThread(ctx, a.ID)
	if _, err := agentruntime.RunTyped[input, struct{ Wrong string }](ctx, r, a.ID, input{"bad type"}); err == nil {
		t.Fatal("wrong output accepted")
	}
	after, _ := r.ReadThread(ctx, a.ID)
	if after.CompletedRuns != before.CompletedRuns || len(after.Runs) != len(before.Runs) {
		t.Fatal("wrong type executed a turn")
	}
	if len(after.Values) != 0 {
		t.Fatal("child values leaked into parent")
	}
	for _, record := range after.Runs {
		if record.Result.ChildCalls != 1 || record.Result.ModelCalls != 0 {
			t.Fatalf("wrong counters: %+v", record.Result)
		}
		if len(record.Events) != 4 || record.Events[1].ChildThreadID != a.Children["worker"] {
			t.Fatalf("missing trace: %+v", record.Events)
		}
	}
}

func TestRegistrationRejectsUnresolvedVersionSchemaAndCyclesAtomically(t *testing.T) {
	noop := func(context.Context, agentspec.Execution, input) (output, error) { return output{}, nil }
	child := handlerPackage(t, "example.com/child", nil, noop)
	for _, test := range []struct {
		name, version string
		mismatch      bool
	}{
		{"missing", "v0.1.0", false}, {"version", "v0.2.0", false}, {"schema", "v0.1.0", true},
	} {
		t.Run(test.name, func(t *testing.T) {
			ref, err := agentspec.NewSubagent[input, output]("child", "Child", child.AgentID("run"), test.version)
			if test.mismatch {
				ref, err = agentspec.NewSubagent[struct{ Other string }, output]("child", "Child", child.AgentID("run"), test.version)
			}
			if err != nil {
				t.Fatal(err)
			}
			parent := handlerPackage(t, "example.com/parent", []agentspec.Subagent{ref}, noop)
			r, _ := agentruntime.New(agentruntime.Config{})
			packages := []agentspec.Package{parent}
			if test.name != "missing" {
				packages = append(packages, child)
			}
			if err := r.Register(packages...); err == nil || len(r.Catalog()) != 0 {
				t.Fatalf("registration was not rejected atomically: %v", err)
			}
		})
	}
	ref, _ := agentspec.NewSubagent[input, output]("self", "Self", "example.com/cycle#run", "v0.1.0")
	r, _ := agentruntime.New(agentruntime.Config{})
	if err := r.Register(handlerPackage(t, "example.com/cycle", []agentspec.Subagent{ref}, noop)); err == nil || !strings.Contains(err.Error(), "cyclic") {
		t.Fatalf("cycle: %v", err)
	}
}

func TestDurableRequestIDsDeduplicateAcrossRestartAndConcurrentRedelivery(t *testing.T) {
	var calls atomic.Int32
	pkg := handlerPackage(t, "example.com/handler", nil, func(context.Context, agentspec.Execution, input) (output, error) {
		return output{int(calls.Add(1))}, nil
	})
	dir := t.TempDir()
	open := func() *agentruntime.Runtime {
		return newRuntime(t, agentruntime.Config{Checkpointer: persistence.NewFileCheckpointer(dir)}, pkg)
	}
	r := open()
	thread := createThread(t, r, pkg.AgentID("run"))
	options := agentruntime.RunOptions{RequestID: "order-42"}
	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			out, err := r.RunWithOptions(context.Background(), thread.ID, input{"hello"}, options)
			if err != nil || string(out.Output) != `{"count":1}` {
				t.Errorf("redelivery: %+v %v", out, err)
			}
		}()
	}
	wg.Wait()
	r = open()
	result, err := r.RunWithOptions(context.Background(), thread.ID, map[string]string{"text": "hello"}, options)
	if err != nil || calls.Load() != 1 || result.RequestID != options.RequestID {
		t.Fatalf("restart dedupe: %+v %v calls=%d", result, err, calls.Load())
	}
	result.Output[0] = 'x'
	result, err = r.RunWithOptions(context.Background(), thread.ID, input{"hello"}, options)
	if err != nil || !json.Valid(result.Output) {
		t.Fatal("result alias corrupted journal")
	}
	if _, err := r.RunWithOptions(context.Background(), thread.ID, input{"different"}, options); !errors.Is(err, agentruntime.ErrRequestConflict) {
		t.Fatalf("conflict: %v", err)
	}
}

type failFinalStore struct {
	persistence.Checkpointer
	fail bool
}

func (s *failFinalStore) Save(ctx context.Context, cp *persistence.Checkpoint) error {
	if s.fail && cp.StepID > 0 {
		return errSave
	}
	return s.Checkpointer.Save(ctx, cp)
}

func TestUncertainCommitBlocksReplayUntilExplicitAbandonment(t *testing.T) {
	ctx := context.Background()
	var effects int
	pkg := handlerPackage(t, "example.com/effects", nil, func(_ context.Context, s agentspec.Execution, _ input) (output, error) {
		effects++
		return output{effects}, s.SetValue("value", json.RawMessage(`1`))
	})
	store := &failFinalStore{Checkpointer: persistence.NewMemoryCheckpointer(), fail: true}
	r := newRuntime(t, agentruntime.Config{Checkpointer: store}, pkg)
	thread := createThread(t, r, pkg.AgentID("run"))
	options := agentruntime.RunOptions{RequestID: "first"}
	if _, err := r.RunWithOptions(ctx, thread.ID, input{"go"}, options); !errors.Is(err, agentruntime.ErrRunIndeterminate) || !errors.Is(err, errSave) {
		t.Fatalf("commit failure: %v", err)
	}
	store.fail = false
	r = newRuntime(t, agentruntime.Config{Checkpointer: store}, pkg)
	for _, id := range []string{"first", "another"} {
		if _, err := r.RunWithOptions(ctx, thread.ID, input{"go"}, agentruntime.RunOptions{RequestID: id}); !errors.Is(err, agentruntime.ErrRunIndeterminate) {
			t.Fatalf("replayed uncertain run: %v", err)
		}
	}
	if effects != 1 {
		t.Fatalf("duplicated external effects: %d", effects)
	}
	snapshot, _ := r.ReadThread(ctx, thread.ID)
	if snapshot.CompletedRuns != 0 || len(snapshot.Values) != 0 || snapshot.Runs["first"].Status != "running" {
		t.Fatalf("bad checkpoint: %+v", snapshot)
	}
	if err := r.AbandonRun(ctx, thread.ID, "first", ""); err == nil {
		t.Fatal("empty abandonment reason accepted")
	}
	if err := r.AbandonRun(ctx, thread.ID, "first", "Operator inspected the external action"); err != nil {
		t.Fatal(err)
	}
	if _, err := r.RunWithOptions(ctx, thread.ID, input{"go"}, options); !errors.Is(err, agentruntime.ErrRunFailed) {
		t.Fatalf("abandoned key reused: %v", err)
	}
	if _, err := r.RunWithOptions(ctx, thread.ID, input{"deliberate retry"}, agentruntime.RunOptions{RequestID: "second"}); err != nil || effects != 2 {
		t.Fatalf("deliberate retry: %v", err)
	}
}

func TestHandlerFailurePanicAndCancellationPersistFailureWithoutValues(t *testing.T) {
	for _, failure := range []string{"error", "panic", "cancel"} {
		t.Run(failure, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			pkg := handlerPackage(t, "example.com/failure", nil, func(_ context.Context, s agentspec.Execution, _ input) (output, error) {
				if err := s.SetValue("uncommitted", json.RawMessage(`1`)); err != nil {
					return output{}, err
				}
				switch failure {
				case "panic":
					panic("handler failed")
				case "cancel":
					cancel()
					return output{}, ctx.Err()
				}
				return output{}, errors.New("handler failed")
			})
			r := newRuntime(t, agentruntime.Config{MaxRecordedRuns: 1}, pkg)
			thread := createThread(t, r, pkg.AgentID("run"))
			if _, err := r.RunWithOptions(ctx, thread.ID, input{"fail"}, agentruntime.RunOptions{RequestID: "failed"}); err == nil {
				t.Fatal("failure swallowed")
			}
			snapshot, err := r.ReadThread(context.Background(), thread.ID)
			if err != nil || len(snapshot.Values) != 0 || len(snapshot.Messages) != 0 || snapshot.Runs["failed"].Status != "failed" {
				t.Fatalf("failure not journaled: %+v %v", snapshot, err)
			}
			if _, err := r.Run(context.Background(), thread.ID, input{"again"}); !errors.Is(err, agentruntime.ErrJournalFull) {
				t.Fatalf("journal evicted key: %v", err)
			}
		})
	}
}

func TestManifestFingerprintPreventsSilentPromptChanges(t *testing.T) {
	pkg := counterPackage(t, "example.com/pinned", "v0.1.0", 4)
	store := persistence.NewMemoryCheckpointer()
	r := newRuntime(t, agentruntime.Config{Checkpointer: store}, pkg)
	thread := createThread(t, r, pkg.AgentID("counter"))
	changed := counterPackage(t, "example.com/pinned", "v0.1.0", 5)
	r = newRuntime(t, agentruntime.Config{Checkpointer: store}, changed)
	if _, err := r.Run(context.Background(), thread.ID, input{"hello"}); err == nil {
		t.Fatal("changed definition resumed old thread")
	}
}

func TestPromptCallsDeclaredChildAsTool(t *testing.T) {
	child := handlerPackage(t, "example.com/worker", nil, func(context.Context, agentspec.Execution, input) (output, error) { return output{7}, nil })
	ref, err := agentspec.NewSubagent[input, output]("worker", "Delegate to worker", child.AgentID("run"), "v0.1.0")
	if err != nil {
		t.Fatal(err)
	}
	definition, err := agentspec.NewAgent[input, output](agentspec.AgentConfig{
		Name: "parent", Description: "Delegate", Subagents: []agentspec.Subagent{ref},
		Prompt: agentspec.Prompt{System: "Use worker", Model: agentspec.Model{Provider: "test", Name: "model", MaxTokens: 100}},
	})
	if err != nil {
		t.Fatal(err)
	}
	parent, err := agentspec.NewPackage("example.com/parent", "v0.1.0", "Parent", definition)
	if err != nil {
		t.Fatal(err)
	}
	provider := completionFunc(func(_ context.Context, _ string, req llm.CompletionRequest) (*llm.CompletionResponse, error) {
		if len(req.Tools) != 1 || req.Tools[0].Function.Name != "worker" {
			t.Fatal("child not advertised to model")
		}
		last := req.Messages[len(req.Messages)-1]
		if last.Role == "tool" {
			return reply(llm.Message{Content: last.Content}), nil
		}
		return reply(llm.Message{ToolCalls: []llm.ToolCall{{ID: "delegate", Type: "function", Function: llm.FunctionCall{Name: "worker", Arguments: `{"text":"delegated"}`}}}}), nil
	})
	r := newRuntime(t, agentruntime.Config{Providers: provider}, parent, child)
	thread := createThread(t, r, parent.AgentID("parent"))
	result, err := r.Run(context.Background(), thread.ID, input{"start"})
	if err != nil || string(result.Output) != `{"count":7}` || result.ChildCalls != 1 || result.ToolCalls != 0 || result.ModelCalls != 2 {
		t.Fatalf("delegated result: %+v %v", result, err)
	}
}

func TestParentFailurePreservesCommittedChildAndDoesNotReplaySameRequest(t *testing.T) {
	var calls int
	child := handlerPackage(t, "example.com/child", nil, func(context.Context, agentspec.Execution, input) (output, error) { calls++; return output{calls}, nil })
	ref, _ := agentspec.NewSubagent[input, output]("child", "Child", child.AgentID("run"), "v0.1.0")
	parent := handlerPackage(t, "example.com/parent", []agentspec.Subagent{ref}, func(ctx context.Context, s agentspec.Execution, i input) (output, error) {
		if _, err := agentspec.CallAgent[input, struct{ Wrong string }](ctx, s, "child", i); err == nil {
			return output{}, errors.New("mismatched child type accepted")
		}
		out, err := agentspec.CallAgent[input, output](ctx, s, "child", i)
		if err != nil {
			return output{}, err
		}
		return out, errors.New("parent failed after child commit")
	})
	r := newRuntime(t, agentruntime.Config{}, parent, child)
	thread := createThread(t, r, parent.AgentID("run"))
	ctx := context.Background()
	options := agentruntime.RunOptions{RequestID: "failure"}
	if _, err := r.RunWithOptions(ctx, thread.ID, input{"start"}, options); err == nil {
		t.Fatal("parent failure swallowed")
	}
	if _, err := r.RunWithOptions(ctx, thread.ID, input{"start"}, options); !errors.Is(err, agentruntime.ErrRunFailed) {
		t.Fatalf("replayed parent: %v", err)
	}
	parentState, _ := r.ReadThread(ctx, thread.ID)
	childState, _ := r.ReadThread(ctx, thread.Children["child"])
	if calls != 1 || parentState.CompletedRuns != 0 || childState.CompletedRuns != 1 {
		t.Fatalf("bad partial commit semantics: calls=%d parent=%d child=%d", calls, parentState.CompletedRuns, childState.CompletedRuns)
	}
	if parentState.Runs[options.RequestID].Events[2].Type != "child_succeeded" {
		t.Fatal("failed parent lost child trace")
	}
}

func TestCreateThreadCleansUpPartialTreeOnStorageFailure(t *testing.T) {
	noop := func(context.Context, agentspec.Execution, input) (output, error) { return output{}, nil }
	child := handlerPackage(t, "example.com/child", nil, noop)
	ref, _ := agentspec.NewSubagent[input, output]("child", "Child", child.AgentID("run"), "v0.1.0")
	parent := handlerPackage(t, "example.com/parent", []agentspec.Subagent{ref}, noop)
	store := &failNthStore{Checkpointer: persistence.NewMemoryCheckpointer(), failAt: 2}
	r := newRuntime(t, agentruntime.Config{Checkpointer: store}, parent, child)
	if _, err := r.CreateThread(context.Background(), parent.AgentID("run")); !errors.Is(err, errSave) {
		t.Fatalf("tree save: %v", err)
	}
	for _, cp := range store.saved {
		if _, err := store.Load(context.Background(), cp.ThreadID, cp.ID); err == nil {
			t.Fatal("partial tree left saved records")
		}
	}
}

type failNthStore struct {
	persistence.Checkpointer
	failAt, calls int
	saved         []*persistence.Checkpoint
}

func (s *failNthStore) Save(ctx context.Context, cp *persistence.Checkpoint) error {
	s.calls++
	if s.calls == s.failAt {
		return errSave
	}
	s.saved = append(s.saved, cp)
	return s.Checkpointer.Save(ctx, cp)
}
