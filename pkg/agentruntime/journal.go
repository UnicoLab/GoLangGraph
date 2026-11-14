// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package agentruntime

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"time"

	"github.com/UnicoLab/GoLangGraph/pkg/llm"
	"github.com/google/uuid"
)

var (
	ErrRequestConflict  = errors.New("request ID was already used with different input")
	ErrRunIndeterminate = errors.New("run outcome is indeterminate; inspect effects before abandoning it")
	ErrRunFailed        = errors.New("request previously failed; a new request ID is required for a deliberate retry")
	ErrJournalFull      = errors.New("thread run journal is full; create a new thread")
	ErrInvocationLimit  = errors.New("run tool/child invocation limit exceeded")
)

type RunOptions struct {
	// RequestID is unique within a thread. Empty generates a new identity.
	RequestID string
}

// Event contains execution metadata, never raw prompts, inputs or tool outputs.
// Events are saved with the final run record, not streamed or checkpointed per step.
type Event struct {
	Sequence       int       `json:"sequence"`
	Type           string    `json:"type"`
	Name           string    `json:"name,omitempty"`
	Time           time.Time `json:"time"`
	ChildThreadID  string    `json:"child_thread_id,omitempty"`
	ChildRequestID string    `json:"child_request_id,omitempty"`
}

type RunRecord struct {
	RequestID  string     `json:"request_id"`
	InputHash  string     `json:"input_hash"`
	Status     string     `json:"status"` // running, succeeded, failed or abandoned
	StartedAt  time.Time  `json:"started_at"`
	FinishedAt *time.Time `json:"finished_at,omitempty"`
	Result     *Result    `json:"result,omitempty"`
	Error      string     `json:"error,omitempty"`
	Events     []Event    `json:"events"`
}

var requestPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$`)

// RunWithOptions durably records intent before calling external code. Completed
// IDs return their saved result. An interrupted run blocks further work until
// explicitly abandoned; this is deduplication, not exactly-once external I/O.
func (r *Runtime) RunWithOptions(ctx context.Context, threadID string, input any, options RunOptions) (*Result, error) {
	if options.RequestID == "" {
		options.RequestID = uuid.NewString()
	}
	if !requestPattern.MatchString(options.RequestID) {
		return nil, fmt.Errorf("request ID must be 1-128 ASCII letters, digits, dots, colons, underscores or hyphens, starting with a letter or digit")
	}
	release, err := r.acquire(ctx, threadID)
	if err != nil {
		return nil, err
	}
	defer release()
	thread, err := r.load(ctx, threadID)
	if err != nil {
		return nil, err
	}
	r.mu.RLock()
	registered, exists := r.agents[thread.AgentID]
	r.mu.RUnlock()
	if !exists || registered.version != thread.PackageVersion || registered.definition.Fingerprint() != thread.DefinitionFingerprint {
		return nil, fmt.Errorf("thread requires unchanged agent %s at package version %s", thread.AgentID, thread.PackageVersion)
	}
	data, err := json.Marshal(input)
	if err != nil {
		return nil, fmt.Errorf("encode input: %w", err)
	}
	if validationErr := registered.definition.ValidateInput(data); validationErr != nil {
		return nil, fmt.Errorf("agent input: %w", validationErr)
	}
	// Normalize object key order while preserving numbers, including uint64.
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	var normalized any
	if decodeErr := decoder.Decode(&normalized); decodeErr != nil {
		return nil, decodeErr
	}
	canonical, err := json.Marshal(normalized)
	if err != nil {
		return nil, err
	}
	hash := fmt.Sprintf("sha256:%x", sha256.Sum256(canonical))
	if previous, ok := thread.Runs[options.RequestID]; ok {
		if previous.InputHash != hash {
			return nil, ErrRequestConflict
		}
		switch previous.Status {
		case "succeeded":
			if previous.Result == nil {
				return nil, fmt.Errorf("corrupt completed run %s", options.RequestID)
			}
			return previous.Result, nil
		case "running":
			return nil, fmt.Errorf("%w: %s", ErrRunIndeterminate, options.RequestID)
		case "failed", "abandoned":
			return nil, fmt.Errorf("%w: %s: %s", ErrRunFailed, options.RequestID, previous.Error)
		default:
			return nil, fmt.Errorf("unsupported run status %q", previous.Status)
		}
	}
	for id, previous := range thread.Runs {
		if previous.Status == "running" {
			return nil, fmt.Errorf("%w: %s", ErrRunIndeterminate, id)
		}
	}
	if len(thread.Runs) >= r.maxRecordedRuns {
		return nil, ErrJournalFull
	}
	record := RunRecord{RequestID: options.RequestID, InputHash: hash, Status: "running", StartedAt: time.Now().UTC()}
	record.Events = []Event{{Sequence: 1, Type: "run_started", Time: record.StartedAt}}
	thread.Runs[options.RequestID] = record
	thread.UpdatedAt = record.StartedAt
	if saveErr := r.save(ctx, thread); saveErr != nil {
		return nil, saveErr
	}
	// Keep the intent snapshot separate from mutable execution state so a failed
	// turn retains the journal but discards its messages and value writes.
	before, err := json.Marshal(thread)
	if err != nil {
		return nil, err
	}
	result := &Result{ThreadID: thread.ID, RequestID: options.RequestID}
	scope := newExecution(r, registered.definition, thread, result, &record)
	thread.Messages = append(thread.Messages, llm.Message{Role: "user", Content: string(data)})
	_, executionErr := r.execute(ctx, registered.definition, thread, data, scope)
	if executionErr == nil {
		executionErr = ctx.Err()
	}
	finished := time.Now().UTC()
	record.FinishedAt = &finished
	if executionErr != nil {
		var original Thread
		if err := json.Unmarshal(before, &original); err != nil {
			return nil, errors.Join(ErrRunIndeterminate, executionErr, err)
		}
		record.Status, record.Error = "failed", executionErr.Error()
		scope.event("run_failed", "", "", "")
		original.Runs[options.RequestID], original.UpdatedAt = record, finished
		cleanup, cancel := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
		defer cancel()
		if err := r.save(cleanup, &original); err != nil {
			return nil, errors.Join(ErrRunIndeterminate, executionErr, err)
		}
		return nil, executionErr
	}
	record.Status, record.Result = "succeeded", result
	scope.event("run_succeeded", "", "", "")
	thread.Runs[options.RequestID] = record
	thread.CompletedRuns++
	thread.UpdatedAt = finished
	if err := r.save(ctx, thread); err != nil {
		return nil, errors.Join(ErrRunIndeterminate, err)
	}
	return result, nil
}

// AbandonRun acknowledges an interrupted request after the host has inspected
// its external and child effects. It never replays work or rolls back effects.
// A retry must use a new request ID. In-process locking prevents abandoning an
// actively executing run in this Runtime; cross-process ownership is required.
func (r *Runtime) AbandonRun(ctx context.Context, threadID, requestID, reason string) error {
	if len(reason) == 0 {
		return fmt.Errorf("an abandonment reason is required")
	}
	release, err := r.acquire(ctx, threadID)
	if err != nil {
		return err
	}
	defer release()
	thread, err := r.load(ctx, threadID)
	if err != nil {
		return err
	}
	record, exists := thread.Runs[requestID]
	if !exists || record.Status != "running" {
		return fmt.Errorf("request %s is not indeterminate", requestID)
	}
	now := time.Now().UTC()
	record.Status, record.Error, record.FinishedAt = "abandoned", reason, &now
	record.Events = append(record.Events, Event{Sequence: len(record.Events) + 1, Type: "run_abandoned", Time: now})
	thread.Runs[requestID], thread.UpdatedAt = record, now
	return r.save(ctx, thread)
}
