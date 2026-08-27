// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// geminiStreamingAPI is a fake Generative Language API that streams several
// chunks, so streaming tests exercise the real SSE parsing path.
func geminiStreamingAPI(t *testing.T, parts ...string) *geminiAPI {
	t.Helper()
	if len(parts) == 0 {
		parts = []string{"one ", "two ", "three"}
	}
	return newGeminiAPI(t, func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.Path, "streamGenerateContent") {
			w.Header().Set("Content-Type", "text/event-stream")
			flusher, _ := w.(http.Flusher)
			for _, part := range parts {
				frame, _ := json.Marshal(map[string]interface{}{
					"candidates": []map[string]interface{}{{
						"content": map[string]interface{}{"parts": []map[string]string{{"text": part}}},
					}},
				})
				_, _ = fmt.Fprintf(w, "data: %s\n\n", frame)
				if flusher != nil {
					flusher.Flush()
				}
			}
			return
		}
		geminiSuccess(strings.Join(parts, ""))(w, r)
	})
}

func TestStreamingConfig(t *testing.T) {
	config := DefaultStreamingConfig()
	assert.False(t, config.Enabled)
	assert.Equal(t, StreamModeAuto, config.Mode)
	assert.Equal(t, 1024, config.ChunkSize)
	assert.Equal(t, 4096, config.BufferSize)
	assert.Equal(t, 50, config.FlushDelay)
	assert.True(t, config.KeepAlive)
}

func TestStreamModes(t *testing.T) {
	tests := []struct {
		mode     StreamMode
		expected string
	}{
		{StreamModeNone, "none"},
		{StreamModeAuto, "auto"},
		{StreamModeForced, "forced"},
	}

	for _, test := range tests {
		assert.Equal(t, test.expected, string(test.mode))
	}
}

func TestOllamaProviderStreaming(t *testing.T) {
	config := DefaultProviderConfig()
	config.Type = "ollama"
	config.Endpoint = "http://localhost:11434"
	config.Model = "llama2"

	provider, err := NewOllamaProvider(config)
	require.NoError(t, err)

	// Test streaming support
	assert.True(t, provider.SupportsStreaming())

	// Test streaming config
	streamConfig := provider.GetStreamingConfig()
	assert.NotNil(t, streamConfig)
	assert.False(t, streamConfig.Enabled) // Default is disabled

	// Test setting streaming config
	newConfig := &StreamingConfig{
		Enabled:    true,
		Mode:       StreamModeForced,
		ChunkSize:  512,
		BufferSize: 2048,
		FlushDelay: 100,
		KeepAlive:  false,
	}

	err = provider.SetStreamingConfig(newConfig)
	assert.NoError(t, err)

	updatedConfig := provider.GetStreamingConfig()
	assert.True(t, updatedConfig.Enabled)
	assert.Equal(t, StreamModeForced, updatedConfig.Mode)
	assert.Equal(t, 512, updatedConfig.ChunkSize)
	assert.Equal(t, 2048, updatedConfig.BufferSize)
	assert.Equal(t, 100, updatedConfig.FlushDelay)
	assert.False(t, updatedConfig.KeepAlive)
}

func TestOpenAIProviderStreaming(t *testing.T) {
	config := DefaultProviderConfig()
	config.Type = "openai"
	config.APIKey = "test-key" // pragma: allowlist secret
	config.Model = "gpt-3.5-turbo"

	provider, err := NewOpenAIProvider(config)
	require.NoError(t, err)

	// Test streaming support
	assert.True(t, provider.SupportsStreaming())

	// Test streaming config
	streamConfig := provider.GetStreamingConfig()
	assert.NotNil(t, streamConfig)

	// Test setting streaming config
	newConfig := &StreamingConfig{
		Enabled: true,
		Mode:    StreamModeAuto,
	}

	err = provider.SetStreamingConfig(newConfig)
	assert.NoError(t, err)

	updatedConfig := provider.GetStreamingConfig()
	assert.True(t, updatedConfig.Enabled)
	assert.Equal(t, StreamModeAuto, updatedConfig.Mode)
}

func TestGeminiProviderStreaming(t *testing.T) {
	config := DefaultProviderConfig()
	config.Type = "gemini"
	config.APIKey = "test-key" // pragma: allowlist secret
	config.Model = "gemini-pro"

	provider, err := NewGeminiProvider(config)
	require.NoError(t, err)

	// Test streaming support
	assert.True(t, provider.SupportsStreaming())

	// Test streaming config
	streamConfig := provider.GetStreamingConfig()
	assert.NotNil(t, streamConfig)
}

func TestProviderManagerStreaming(t *testing.T) {
	pm := NewProviderManager()

	// Add test provider
	config := DefaultProviderConfig()
	config.Type = "gemini"
	config.APIKey = "test-key" // pragma: allowlist secret

	provider, err := NewGeminiProvider(config)
	require.NoError(t, err)

	err = pm.RegisterProvider("test-gemini", provider)
	require.NoError(t, err)

	// Test enabling streaming
	err = pm.EnableStreaming("test-gemini")
	assert.NoError(t, err)

	streamConfig := provider.GetStreamingConfig()
	assert.True(t, streamConfig.Enabled)
	assert.Equal(t, StreamModeAuto, streamConfig.Mode)

	// Test disabling streaming
	err = pm.DisableStreaming("test-gemini")
	assert.NoError(t, err)

	streamConfig = provider.GetStreamingConfig()
	assert.False(t, streamConfig.Enabled)
	assert.Equal(t, StreamModeNone, streamConfig.Mode)

	// Test setting streaming mode
	err = pm.SetStreamingMode("test-gemini", StreamModeForced)
	assert.NoError(t, err)

	streamConfig = provider.GetStreamingConfig()
	assert.True(t, streamConfig.Enabled)
	assert.Equal(t, StreamModeForced, streamConfig.Mode)
}

func TestStreamingModesWithGemini(t *testing.T) {
	// Backed by a fake Generative Language API: the provider builds real
	// requests and parses real responses. Before, this passed against a
	// hardcoded reply that never left the process.
	api := newGeminiAPI(t, func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.Path, "streamGenerateContent") {
			w.Header().Set("Content-Type", "text/event-stream")
			frame, _ := json.Marshal(map[string]interface{}{
				"candidates": []map[string]interface{}{{
					"content": map[string]interface{}{"parts": []map[string]string{{"text": "streamed"}}},
				}},
			})
			_, _ = fmt.Fprintf(w, "data: %s\n\n", frame)
			return
		}
		geminiSuccess("complete")(w, r)
	})
	provider := api.provider(t, func(c *ProviderConfig) { c.Model = "gemini-pro" })

	ctx := context.Background()
	req := CompletionRequest{
		Messages: []Message{
			{Role: "user", Content: "Hello, how are you?"},
		},
	}

	// Test non-streaming mode
	resp, err := provider.CompleteWithMode(ctx, req, StreamModeNone)
	assert.NoError(t, err)
	assert.NotNil(t, resp)
	assert.Equal(t, "chat.completion", resp.Object)

	// Test forced streaming mode (collected)
	resp, err = provider.CompleteWithMode(ctx, req, StreamModeForced)
	assert.NoError(t, err)
	assert.NotNil(t, resp)
	assert.Equal(t, "chat.completion", resp.Object)

	// Test auto mode (non-streaming)
	resp, err = provider.CompleteWithMode(ctx, req, StreamModeAuto)
	assert.NoError(t, err)
	assert.NotNil(t, resp)

	// Test auto mode (streaming)
	req.Stream = true
	resp, err = provider.CompleteWithMode(ctx, req, StreamModeAuto)
	assert.NoError(t, err)
	assert.NotNil(t, resp)
}

func TestStreamingCallbackWithGemini(t *testing.T) {
	config := DefaultProviderConfig()
	config.Type = "gemini"
	_ = config

	api := geminiStreamingAPI(t, "1 ", "2 ", "3 ", "4 ", "5")
	provider := api.provider(t, func(c *ProviderConfig) { c.Model = "gemini-pro" })

	ctx := context.Background()
	req := CompletionRequest{
		Messages: []Message{
			{Role: "user", Content: "Count from 1 to 5"},
		},
	}

	var chunks []string
	callback := func(chunk CompletionResponse) error {
		if len(chunk.Choices) > 0 {
			chunks = append(chunks, chunk.Choices[0].Delta.Content)
		}
		return nil
	}

	// Test streaming with callback
	err := provider.CompleteStreamWithMode(ctx, req, callback, StreamModeForced)
	assert.NoError(t, err)
	assert.Greater(t, len(chunks), 1) // Should receive multiple chunks

	// Verify chunks combine to form complete response
	combined := strings.Join(chunks, "")
	assert.NotEmpty(t, combined)

	// Test non-streaming mode with callback (should get single chunk)
	chunks = []string{}
	err = provider.CompleteStreamWithMode(ctx, req, callback, StreamModeNone)
	assert.NoError(t, err)
	assert.Equal(t, 1, len(chunks)) // Should receive single chunk
}

func TestProviderConfigWithStreaming(t *testing.T) {
	config := DefaultProviderConfig()
	assert.NotNil(t, config.Streaming)
	assert.False(t, config.Streaming.Enabled)
	assert.Equal(t, StreamModeAuto, config.Streaming.Mode)
}

func TestStreamingConfigValidation(t *testing.T) {
	config := DefaultProviderConfig()
	config.APIKey = "test-key" // pragma: allowlist secret
	provider, err := NewGeminiProvider(config)
	require.NoError(t, err)

	// Test setting nil config
	err = provider.SetStreamingConfig(nil)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "cannot be nil")

	// Test setting valid config
	validConfig := &StreamingConfig{
		Enabled: true,
		Mode:    StreamModeForced,
	}
	err = provider.SetStreamingConfig(validConfig)
	assert.NoError(t, err)
}

// Benchmark tests for streaming performance
func BenchmarkStreamingVsNonStreaming(b *testing.B) {
	config := DefaultProviderConfig()
	config.Type = "gemini"
	config.APIKey = "test-key" // pragma: allowlist secret

	provider, err := NewGeminiProvider(config)
	require.NoError(b, err)

	ctx := context.Background()
	req := CompletionRequest{
		Messages: []Message{
			{Role: "user", Content: "Generate a short response"},
		},
	}

	b.Run("NonStreaming", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, err := provider.CompleteWithMode(ctx, req, StreamModeNone)
			if err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("StreamingCollected", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, err := provider.CompleteWithMode(ctx, req, StreamModeForced)
			if err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("StreamingCallback", func(b *testing.B) {
		callback := func(chunk CompletionResponse) error {
			return nil
		}

		for i := 0; i < b.N; i++ {
			err := provider.CompleteStreamWithMode(ctx, req, callback, StreamModeForced)
			if err != nil {
				b.Fatal(err)
			}
		}
	})
}

// Integration test for realistic streaming scenario
func TestRealisticStreamingScenario(t *testing.T) {
	// Create provider manager
	pm := NewProviderManager()

	// Register a Gemini provider backed by a fake API, so the streaming path
	// under test is the real one.
	api := geminiStreamingAPI(t, "Once ", "upon ", "a ", "time")
	provider := api.provider(t, func(c *ProviderConfig) { c.Model = "gemini-pro" })

	err := pm.RegisterProvider("gemini", provider)
	require.NoError(t, err)

	// Enable streaming
	err = pm.EnableStreaming("gemini")
	require.NoError(t, err)

	ctx := context.Background()
	req := CompletionRequest{
		Messages: []Message{
			{Role: "user", Content: "Write a short story about AI"},
		},
		Stream: true,
	}

	// Test streaming with real-time processing
	var chunks []string
	var totalLatency time.Duration
	startTime := time.Now()

	callback := func(chunk CompletionResponse) error {
		chunkTime := time.Now()
		chunks = append(chunks, chunk.Choices[0].Delta.Content)
		totalLatency += time.Since(chunkTime)

		// Simulate real-time processing (e.g., UI updates)
		time.Sleep(1 * time.Millisecond)
		return nil
	}

	err = pm.CompleteStreamWithMode(ctx, "gemini", req, callback, StreamModeAuto)
	assert.NoError(t, err)

	endTime := time.Now()
	totalTime := endTime.Sub(startTime)

	// Verify streaming performance characteristics
	assert.Greater(t, len(chunks), 1, "Should receive multiple chunks")
	assert.Less(t, totalLatency, totalTime, "Processing time should be less than total time")

	// Verify content
	combined := strings.Join(chunks, "")
	assert.NotEmpty(t, combined, "Combined content should not be empty")

	t.Logf("Streaming test completed: %d chunks, %v total time, %v processing time",
		len(chunks), totalTime, totalLatency)
}
