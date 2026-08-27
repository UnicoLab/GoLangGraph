// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

// Package fakes provides deterministic test doubles for GoLangGraph's external
// dependencies.
//
// The doubles stand in for a language model only. Everything else in a test
// using them — the graph engine, the agent loop, tool execution, state
// handling, the HTTP server — is the real implementation, so the tests
// exercise the framework rather than a mock of it.
package fakes

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/UnicoLab/GoLangGraph/pkg/llm"
)

// Provider is a scriptable llm.Provider.
//
// Replies are returned in order and the last one repeats, so a test can script
// an agent's reasoning turn by turn without depending on a live model.
type Provider struct {
	name string

	mu       sync.Mutex
	replies  []string
	failures []error
	delay    time.Duration
	prompts  []string

	calls   atomic.Int32
	healthy atomic.Bool
}

// NewProvider creates a provider that always answers with reply.
func NewProvider(name, reply string) *Provider {
	p := &Provider{name: name, replies: []string{reply}}
	p.healthy.Store(true)
	return p
}

// Script sets the replies returned by successive calls. The final reply repeats
// once the script is exhausted.
func (p *Provider) Script(replies ...string) *Provider {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.replies = append([]string(nil), replies...)
	return p
}

// FailWith makes the next calls fail with the given errors, in order. A nil
// entry means that call succeeds.
func (p *Provider) FailWith(errs ...error) *Provider {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.failures = append([]error(nil), errs...)
	return p
}

// WithDelay makes every call take at least d, respecting cancellation.
func (p *Provider) WithDelay(d time.Duration) *Provider {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.delay = d
	return p
}

// SetHealthy controls what IsHealthy reports.
func (p *Provider) SetHealthy(healthy bool) *Provider {
	p.healthy.Store(healthy)
	return p
}

// Calls returns how many completions have been requested.
func (p *Provider) Calls() int { return int(p.calls.Load()) }

// Prompts returns the content of every user message the provider has received,
// so a test can assert what the agent actually asked.
func (p *Provider) Prompts() []string {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]string(nil), p.prompts...)
}

// next returns the scripted reply and failure for a call index.
func (p *Provider) next(index int, prompt string) (string, error, time.Duration) {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.prompts = append(p.prompts, prompt)

	var failure error
	if index < len(p.failures) {
		failure = p.failures[index]
	}

	reply := ""
	switch {
	case len(p.replies) == 0:
	case index < len(p.replies):
		reply = p.replies[index]
	default:
		reply = p.replies[len(p.replies)-1]
	}
	return reply, failure, p.delay
}

func (p *Provider) GetName() string { return p.name }

func (p *Provider) GetModels(ctx context.Context) ([]string, error) {
	return []string{"fake-model"}, nil
}

func (p *Provider) Complete(ctx context.Context, req llm.CompletionRequest) (*llm.CompletionResponse, error) {
	index := int(p.calls.Add(1)) - 1

	prompt := ""
	for _, m := range req.Messages {
		if m.Role == "user" {
			prompt = m.Content
		}
	}

	reply, failure, delay := p.next(index, prompt)

	if delay > 0 {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(delay):
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if failure != nil {
		return nil, failure
	}

	return &llm.CompletionResponse{
		ID:      fmt.Sprintf("%s-%d", p.name, index),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   req.Model,
		Choices: []llm.Choice{{
			Index:        0,
			Message:      llm.Message{Role: "assistant", Content: reply},
			FinishReason: "stop",
		}},
		Usage: llm.Usage{PromptTokens: 1, CompletionTokens: 1, TotalTokens: 2},
	}, nil
}

func (p *Provider) CompleteStream(ctx context.Context, req llm.CompletionRequest, callback llm.StreamCallback) error {
	resp, err := p.Complete(ctx, req)
	if err != nil {
		return err
	}
	return callback(*resp)
}

func (p *Provider) CompleteWithMode(ctx context.Context, req llm.CompletionRequest, mode llm.StreamMode) (*llm.CompletionResponse, error) {
	return p.Complete(ctx, req)
}

func (p *Provider) CompleteStreamWithMode(ctx context.Context, req llm.CompletionRequest, callback llm.StreamCallback, mode llm.StreamMode) error {
	return p.CompleteStream(ctx, req, callback)
}

func (p *Provider) IsHealthy(ctx context.Context) error {
	if !p.healthy.Load() {
		return fmt.Errorf("provider %s is unhealthy", p.name)
	}
	return nil
}

func (p *Provider) GetConfig() map[string]interface{} {
	return map[string]interface{}{
		"type":     "fake",
		"endpoint": "memory://fake",
		"model":    "fake-model",
		// Present on purpose: the API must never echo credentials.
		"api_key": "super-secret-key",
	}
}

func (p *Provider) SetConfig(config map[string]interface{}) error   { return nil }
func (p *Provider) SupportsStreaming() bool                         { return true }
func (p *Provider) GetStreamingConfig() *llm.StreamingConfig        { return llm.DefaultStreamingConfig() }
func (p *Provider) SetStreamingConfig(c *llm.StreamingConfig) error { return nil }
func (p *Provider) Close() error                                    { return nil }
