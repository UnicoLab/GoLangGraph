// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

package e2e

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/piotrlaczkowski/GoLangGraph/pkg/llm"
)

// fakeProvider is a deterministic llm.Provider for end-to-end tests. It lets a
// full server, agent and graph run without a live model, so the tests exercise
// real HTTP, real graph execution and real state handling rather than mocks of
// the framework itself.
type fakeProvider struct {
	name string

	mu       sync.Mutex
	reply    string
	failWith error
	delay    time.Duration
	calls    atomic.Int32
	healthy  atomic.Bool
}

func newFakeProvider(name string) *fakeProvider {
	p := &fakeProvider{name: name, reply: "fake response"}
	p.healthy.Store(true)
	return p
}

func (p *fakeProvider) setReply(reply string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.reply = reply
}

func (p *fakeProvider) setDelay(d time.Duration) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.delay = d
}

func (p *fakeProvider) setFailure(err error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.failWith = err
}

func (p *fakeProvider) current() (string, error, time.Duration) {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.reply, p.failWith, p.delay
}

func (p *fakeProvider) GetName() string { return p.name }

func (p *fakeProvider) GetModels(ctx context.Context) ([]string, error) {
	return []string{"fake-model"}, nil
}

func (p *fakeProvider) Complete(ctx context.Context, req llm.CompletionRequest) (*llm.CompletionResponse, error) {
	p.calls.Add(1)
	reply, failure, delay := p.current()

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
		ID:      fmt.Sprintf("fake-%d", p.calls.Load()),
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

func (p *fakeProvider) CompleteStream(ctx context.Context, req llm.CompletionRequest, callback llm.StreamCallback) error {
	resp, err := p.Complete(ctx, req)
	if err != nil {
		return err
	}
	return callback(*resp)
}

func (p *fakeProvider) CompleteWithMode(ctx context.Context, req llm.CompletionRequest, mode llm.StreamMode) (*llm.CompletionResponse, error) {
	return p.Complete(ctx, req)
}

func (p *fakeProvider) CompleteStreamWithMode(ctx context.Context, req llm.CompletionRequest, callback llm.StreamCallback, mode llm.StreamMode) error {
	return p.CompleteStream(ctx, req, callback)
}

func (p *fakeProvider) IsHealthy(ctx context.Context) error {
	if !p.healthy.Load() {
		return fmt.Errorf("provider %s is unhealthy", p.name)
	}
	return nil
}

func (p *fakeProvider) GetConfig() map[string]interface{} {
	return map[string]interface{}{
		"type":     "fake",
		"endpoint": "memory://fake",
		"model":    "fake-model",
		// Deliberately present to prove the API never echoes credentials.
		"api_key": "super-secret-key",
	}
}

func (p *fakeProvider) SetConfig(config map[string]interface{}) error { return nil }
func (p *fakeProvider) SupportsStreaming() bool                       { return true }
func (p *fakeProvider) GetStreamingConfig() *llm.StreamingConfig      { return llm.DefaultStreamingConfig() }
func (p *fakeProvider) SetStreamingConfig(c *llm.StreamingConfig) error {
	return nil
}
func (p *fakeProvider) Close() error { return nil }
