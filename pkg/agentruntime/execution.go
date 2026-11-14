// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package agentruntime

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"reflect"
	"sync"
	"time"

	"github.com/UnicoLab/GoLangGraph/pkg/agentspec"
)

// execution serializes handler accesses, including child invocations. Callbacks
// must finish all work before returning and must not retain this handle.
type execution struct {
	mu          sync.Mutex
	runtime     *Runtime
	thread      *Thread
	result      *Result
	record      *RunRecord
	tools       map[string]agentspec.Tool
	children    map[string]agentspec.SubagentManifest
	invocations int
}

func newExecution(r *Runtime, definition agentspec.Definition, thread *Thread, result *Result, record *RunRecord) *execution {
	s := &execution{runtime: r, thread: thread, result: result, record: record, tools: make(map[string]agentspec.Tool), children: make(map[string]agentspec.SubagentManifest)}
	for _, tool := range definition.Tools() {
		s.tools[tool.Manifest().Name] = tool
	}
	for _, child := range definition.Manifest().Subagents {
		s.children[child.Name] = child
	}
	return s
}

func (s *execution) ThreadID() string { return s.thread.ID }

func (s *execution) GetValue(key string) (json.RawMessage, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	value, exists := s.thread.Values[key]
	return append(json.RawMessage(nil), value...), exists
}

func (s *execution) SetValue(key string, value json.RawMessage) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return (&toolState{values: s.thread.Values}).SetValue(key, value)
}

func (s *execution) event(kind, name, childID, childRequestID string) {
	s.record.Events = append(s.record.Events, Event{Sequence: len(s.record.Events) + 1, Type: kind, Name: name, Time: time.Now().UTC(), ChildThreadID: childID, ChildRequestID: childRequestID})
}

func (s *execution) budget(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if s.invocations >= 256 {
		return ErrInvocationLimit
	}
	s.invocations++
	return nil
}

func (s *execution) InvokeTool(ctx context.Context, name string, input json.RawMessage) (json.RawMessage, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.budget(ctx); err != nil {
		return nil, err
	}
	s.result.ToolCalls++
	s.event("tool_started", name, "", "")
	output, err := s.callTool(ctx, name, input)
	if err != nil {
		s.event("tool_failed", name, "", "")
	} else {
		s.event("tool_succeeded", name, "", "")
	}
	return output, err
}

func (s *execution) callTool(ctx context.Context, name string, input json.RawMessage) (json.RawMessage, error) {
	tool, exists := s.tools[name]
	if !exists {
		return nil, fmt.Errorf("tool %q is not declared by this agent", name)
	}
	values := make(map[string]json.RawMessage, len(s.thread.Values))
	for key, value := range s.thread.Values {
		values[key] = append(json.RawMessage(nil), value...)
	}
	output, err := tool.Execute(ctx, &toolState{id: s.thread.ID, values: values}, input)
	if err == nil {
		s.thread.Values = values
	}
	return output, err
}

func (s *execution) CheckAgent(name string, input, output agentspec.Schema) error {
	child, exists := s.children[name]
	if !exists || !reflect.DeepEqual(child.InputSchema, input) || !reflect.DeepEqual(child.OutputSchema, output) {
		return fmt.Errorf("child %q is undeclared or has incompatible input/output types", name)
	}
	return nil
}

func (s *execution) InvokeAgent(ctx context.Context, name string, input json.RawMessage) (json.RawMessage, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.budget(ctx); err != nil {
		return nil, err
	}
	child, exists := s.children[name]
	if !exists {
		return nil, fmt.Errorf("child %q is not declared by this agent", name)
	}
	id := s.thread.Children[name]
	instance, err := s.runtime.ReadThread(ctx, id)
	if err != nil {
		return nil, err
	}
	if instance.ParentID != s.thread.ID || instance.ChildName != name || instance.AgentID != child.AgentID || instance.PackageVersion != child.PackageVersion {
		return nil, fmt.Errorf("child %s has invalid lineage or package identity", name)
	}
	s.result.ChildCalls++
	requestID := fmt.Sprintf("child-%x", sha256.Sum256([]byte(fmt.Sprintf("%s/%s/%d", s.result.RequestID, name, s.result.ChildCalls))))
	s.event("child_started", name, id, requestID)
	result, err := s.runtime.RunWithOptions(ctx, id, input, RunOptions{RequestID: requestID})
	if err != nil {
		s.event("child_failed", name, id, requestID)
		return nil, err
	}
	s.event("child_succeeded", name, id, requestID)
	return result.Output, nil
}

func (s *execution) invoke(ctx context.Context, name string, input json.RawMessage) (json.RawMessage, error) {
	if _, exists := s.children[name]; exists {
		return s.InvokeAgent(ctx, name, input)
	}
	return s.InvokeTool(ctx, name, input)
}
