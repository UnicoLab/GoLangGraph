// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/sirupsen/logrus"
)

// DefaultGeminiEndpoint is the Generative Language API base URL.
const DefaultGeminiEndpoint = "https://generativelanguage.googleapis.com/v1beta"

// GeminiProvider implements the Provider interface against Google's Generative
// Language API.
//
// This provider previously returned hardcoded strings — "Hello! I'm Gemini..."
// and a note that a real implementation would call the API — while presenting
// itself as a working provider. Configuring it with a valid API key produced
// canned text with nothing to indicate the model had never been contacted.
type GeminiProvider struct {
	config   *ProviderConfig
	logger   *logrus.Logger
	models   []string
	client   *http.Client
	endpoint string
}

// NewGeminiProvider creates a new Gemini provider
func NewGeminiProvider(config *ProviderConfig) (*GeminiProvider, error) {
	if config == nil {
		return nil, fmt.Errorf("gemini configuration is required")
	}
	if config.APIKey == "" {
		return nil, fmt.Errorf("gemini API key is required")
	}

	timeout := config.Timeout
	if timeout <= 0 {
		timeout = 60 * time.Second
	}

	endpoint := strings.TrimRight(config.Endpoint, "/")
	if endpoint == "" {
		endpoint = DefaultGeminiEndpoint
	}

	provider := &GeminiProvider{
		config:   config,
		logger:   logrus.New(),
		models:   []string{"gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"},
		client:   &http.Client{Timeout: timeout},
		endpoint: endpoint,
	}

	return provider, nil
}

// ---------------------------------------------------------------------------
// Generative Language API wire types
// ---------------------------------------------------------------------------

type geminiPart struct {
	Text string `json:"text,omitempty"`
}

type geminiContent struct {
	Role  string       `json:"role,omitempty"`
	Parts []geminiPart `json:"parts"`
}

type geminiGenerationConfig struct {
	Temperature     *float64 `json:"temperature,omitempty"`
	MaxOutputTokens *int     `json:"maxOutputTokens,omitempty"`
	TopP            *float64 `json:"topP,omitempty"`
	StopSequences   []string `json:"stopSequences,omitempty"`
}

type geminiRequest struct {
	Contents          []geminiContent         `json:"contents"`
	SystemInstruction *geminiContent          `json:"systemInstruction,omitempty"`
	GenerationConfig  *geminiGenerationConfig `json:"generationConfig,omitempty"`
}

type geminiCandidate struct {
	Content      geminiContent `json:"content"`
	FinishReason string        `json:"finishReason"`
	Index        int           `json:"index"`
}

type geminiUsage struct {
	PromptTokenCount     int `json:"promptTokenCount"`
	CandidatesTokenCount int `json:"candidatesTokenCount"`
	TotalTokenCount      int `json:"totalTokenCount"`
}

type geminiResponse struct {
	Candidates    []geminiCandidate `json:"candidates"`
	UsageMetadata geminiUsage       `json:"usageMetadata"`
	Error         *struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
		Status  string `json:"status"`
	} `json:"error,omitempty"`
}

// modelFor resolves the model for a request, falling back to configuration.
func (p *GeminiProvider) modelFor(req CompletionRequest) string {
	if req.Model != "" {
		return req.Model
	}
	if p.config.Model != "" {
		return p.config.Model
	}
	return "gemini-1.5-flash"
}

// buildRequest converts a CompletionRequest into the Gemini wire format.
//
// Gemini names the assistant role "model" and carries system prompts in a
// dedicated systemInstruction field rather than as a message.
func (p *GeminiProvider) buildRequest(req CompletionRequest) (*geminiRequest, error) {
	if len(req.Messages) == 0 {
		return nil, fmt.Errorf("no messages provided")
	}

	out := &geminiRequest{}
	var systemParts []geminiPart

	if req.SystemPrompt != "" {
		systemParts = append(systemParts, geminiPart{Text: req.SystemPrompt})
	}

	for _, msg := range req.Messages {
		switch msg.Role {
		case "system":
			systemParts = append(systemParts, geminiPart{Text: msg.Content})
		case "assistant":
			out.Contents = append(out.Contents, geminiContent{
				Role: "model", Parts: []geminiPart{{Text: msg.Content}},
			})
		default:
			out.Contents = append(out.Contents, geminiContent{
				Role: "user", Parts: []geminiPart{{Text: msg.Content}},
			})
		}
	}

	if len(out.Contents) == 0 {
		return nil, fmt.Errorf("no user or assistant messages provided")
	}
	if len(systemParts) > 0 {
		out.SystemInstruction = &geminiContent{Parts: systemParts}
	}

	cfg := &geminiGenerationConfig{}
	set := false
	if req.Temperature != 0 {
		t := req.Temperature
		cfg.Temperature = &t
		set = true
	} else if p.config.Temperature != 0 {
		t := p.config.Temperature
		cfg.Temperature = &t
		set = true
	}
	if req.MaxTokens > 0 {
		m := req.MaxTokens
		cfg.MaxOutputTokens = &m
		set = true
	} else if p.config.MaxTokens > 0 {
		m := p.config.MaxTokens
		cfg.MaxOutputTokens = &m
		set = true
	}
	if len(req.StopSequences) > 0 {
		cfg.StopSequences = req.StopSequences
		set = true
	}
	if set {
		out.GenerationConfig = cfg
	}

	return out, nil
}

// callURL builds an API URL with the key supplied as a query parameter, which
// is how the Generative Language API authenticates.
func (p *GeminiProvider) callURL(model, method string, extra url.Values) string {
	values := url.Values{}
	for k, v := range extra {
		values[k] = v
	}
	values.Set("key", p.config.APIKey)
	return fmt.Sprintf("%s/models/%s:%s?%s", p.endpoint, url.PathEscape(model), method, values.Encode())
}

// toCompletionResponse converts a Gemini response into the common format.
func toCompletionResponse(model string, resp *geminiResponse) *CompletionResponse {
	out := &CompletionResponse{
		ID:      fmt.Sprintf("gemini-%d", time.Now().UnixNano()),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   model,
		Usage: Usage{
			PromptTokens:     resp.UsageMetadata.PromptTokenCount,
			CompletionTokens: resp.UsageMetadata.CandidatesTokenCount,
			TotalTokens:      resp.UsageMetadata.TotalTokenCount,
		},
	}

	for _, candidate := range resp.Candidates {
		var text strings.Builder
		for _, part := range candidate.Content.Parts {
			text.WriteString(part.Text)
		}
		out.Choices = append(out.Choices, Choice{
			Index:        candidate.Index,
			Message:      Message{Role: "assistant", Content: text.String()},
			FinishReason: strings.ToLower(candidate.FinishReason),
		})
	}

	if len(out.Choices) == 0 {
		// A response with no candidates usually means the prompt was blocked.
		out.Choices = append(out.Choices, Choice{
			Index:        0,
			Message:      Message{Role: "assistant", Content: ""},
			FinishReason: "stop",
		})
	}
	return out
}

// GetName returns the provider name
func (p *GeminiProvider) GetName() string {
	return "gemini"
}

// GetModels returns available models
func (p *GeminiProvider) GetModels(ctx context.Context) ([]string, error) {
	return p.models, nil
}

// Complete generates a completion by calling the Generative Language API.
func (p *GeminiProvider) Complete(ctx context.Context, req CompletionRequest) (*CompletionResponse, error) {
	payload, err := p.buildRequest(req)
	if err != nil {
		return nil, err
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	model := p.modelFor(req)
	var result *CompletionResponse

	attempt := func(ctx context.Context) error {
		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost,
			p.callURL(model, "generateContent", nil), bytes.NewReader(body))
		if err != nil {
			return fmt.Errorf("failed to create request: %w", err)
		}
		httpReq.Header.Set("Content-Type", "application/json")

		resp, err := p.client.Do(httpReq)
		if err != nil {
			if ctxErr := ctx.Err(); ctxErr != nil {
				return ctxErr
			}
			return NewTransportError("Gemini", err)
		}
		defer func() { _ = resp.Body.Close() }()

		if resp.StatusCode != http.StatusOK {
			return NewProviderError("Gemini", resp)
		}

		raw, err := io.ReadAll(limitedBody(resp.Body))
		if err != nil {
			return fmt.Errorf("failed to read response: %w", err)
		}

		var decoded geminiResponse
		if err := json.Unmarshal(raw, &decoded); err != nil {
			return fmt.Errorf("failed to decode response: %w", err)
		}
		if decoded.Error != nil {
			return fmt.Errorf("%w: gemini API error: %s", ErrProviderRequest, decoded.Error.Message)
		}

		result = toCompletionResponse(model, &decoded)
		return nil
	}

	if err := WithRetry(ctx, p.config, attempt); err != nil {
		return nil, err
	}
	return result, nil
}

// CompleteStream streams a completion using the API's server-sent events
// endpoint, invoking the callback for each chunk as it arrives.
func (p *GeminiProvider) CompleteStream(ctx context.Context, req CompletionRequest, callback StreamCallback) error {
	payload, err := p.buildRequest(req)
	if err != nil {
		return err
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}

	model := p.modelFor(req)
	target := p.callURL(model, "streamGenerateContent", url.Values{"alt": []string{"sse"}})

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, target, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")

	resp, err := p.client.Do(httpReq)
	if err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return ctxErr
		}
		return NewTransportError("Gemini", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return NewProviderError("Gemini", resp)
	}

	scanner := bufio.NewScanner(io.LimitReader(resp.Body, MaxResponseBytes))
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)

	index := 0
	for scanner.Scan() {
		if err := ctx.Err(); err != nil {
			return err
		}

		line := strings.TrimSpace(scanner.Text())
		if line == "" || !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "" || data == "[DONE]" {
			continue
		}

		var decoded geminiResponse
		if err := json.Unmarshal([]byte(data), &decoded); err != nil {
			return fmt.Errorf("failed to decode stream chunk: %w", err)
		}
		if decoded.Error != nil {
			return fmt.Errorf("%w: gemini API error: %s", ErrProviderRequest, decoded.Error.Message)
		}

		for _, candidate := range decoded.Candidates {
			var text strings.Builder
			for _, part := range candidate.Content.Parts {
				text.WriteString(part.Text)
			}
			chunk := CompletionResponse{
				ID:      fmt.Sprintf("gemini-stream-%d", index),
				Object:  "chat.completion.chunk",
				Created: time.Now().Unix(),
				Model:   model,
				Choices: []Choice{{
					Index:        candidate.Index,
					Delta:        Message{Role: "assistant", Content: text.String()},
					FinishReason: strings.ToLower(candidate.FinishReason),
				}},
				Usage: Usage{
					PromptTokens:     decoded.UsageMetadata.PromptTokenCount,
					CompletionTokens: decoded.UsageMetadata.CandidatesTokenCount,
					TotalTokens:      decoded.UsageMetadata.TotalTokenCount,
				},
			}
			if err := callback(chunk); err != nil {
				if IsStreamEarlyExit(err) {
					return err
				}
				return fmt.Errorf("callback error: %w", err)
			}
			index++
		}
	}

	if err := scanner.Err(); err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return ctxErr
		}
		return fmt.Errorf("failed to read stream: %w", err)
	}
	return nil
}

// IsHealthy checks if the provider is healthy
func (p *GeminiProvider) IsHealthy(ctx context.Context) error {
	target := fmt.Sprintf("%s/models?key=%s", p.endpoint, url.QueryEscape(p.config.APIKey))

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, target, nil)
	if err != nil {
		return fmt.Errorf("failed to create health request: %w", err)
	}

	resp, err := p.client.Do(req)
	if err != nil {
		return NewTransportError("Gemini", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return NewProviderError("Gemini", resp)
	}
	return nil
}

// GetConfig returns provider configuration
func (p *GeminiProvider) GetConfig() map[string]interface{} {
	return map[string]interface{}{
		"name":        p.GetName(),
		"api_key":     "***masked***",
		"model":       p.config.Model,
		"temperature": p.config.Temperature,
		"max_tokens":  p.config.MaxTokens,
	}
}

// SetConfig updates provider configuration
func (p *GeminiProvider) SetConfig(config map[string]interface{}) error {
	if apiKey, ok := config["api_key"].(string); ok {
		p.config.APIKey = apiKey
	}
	if model, ok := config["model"].(string); ok {
		p.config.Model = model
	}
	if temp, ok := config["temperature"].(float64); ok {
		p.config.Temperature = temp
	}
	if maxTokens, ok := config["max_tokens"].(int); ok {
		p.config.MaxTokens = maxTokens
	}
	return nil
}

// Close closes the provider
func (p *GeminiProvider) Close() error {
	return nil
}

// Helper methods for real implementation (when Google API is available)

// GetDefaultModels returns the default Gemini models
func (p *GeminiProvider) GetDefaultModels() []string {
	return []string{
		"gemini-pro",
		"gemini-pro-vision",
		"gemini-1.5-pro",
		"gemini-1.5-flash",
	}
}

// SupportsStreaming returns whether the provider supports streaming
func (p *GeminiProvider) SupportsStreaming() bool {
	return true
}

// GetStreamingConfig returns the current streaming configuration
func (p *GeminiProvider) GetStreamingConfig() *StreamingConfig {
	if p.config.Streaming == nil {
		p.config.Streaming = DefaultStreamingConfig()
	}
	return p.config.Streaming
}

// SetStreamingConfig updates the streaming configuration
func (p *GeminiProvider) SetStreamingConfig(config *StreamingConfig) error {
	if config == nil {
		return fmt.Errorf("streaming config cannot be nil")
	}
	p.config.Streaming = config
	return nil
}

// CompleteWithMode generates a completion with explicit streaming mode
func (p *GeminiProvider) CompleteWithMode(ctx context.Context, req CompletionRequest, mode StreamMode) (*CompletionResponse, error) {
	switch mode {
	case StreamModeNone:
		// Force non-streaming mode
		return p.completeNonStreaming(ctx, req)
	case StreamModeForced:
		// Force streaming mode but collect all chunks
		return p.completeStreamingCollected(ctx, req)
	case StreamModeAuto:
		// Auto-detect based on request.Stream flag
		if req.Stream {
			return p.completeStreamingCollected(ctx, req)
		}
		return p.completeNonStreaming(ctx, req)
	default:
		return p.Complete(ctx, req)
	}
}

// CompleteStreamWithMode generates a streaming completion with explicit mode
func (p *GeminiProvider) CompleteStreamWithMode(ctx context.Context, req CompletionRequest, callback StreamCallback, mode StreamMode) error {
	switch mode {
	case StreamModeNone:
		// Convert to non-streaming
		resp, err := p.completeNonStreaming(ctx, req)
		if err != nil {
			return err
		}
		// Send as single chunk
		return callback(*resp)
	case StreamModeForced, StreamModeAuto:
		// Use normal streaming
		return p.CompleteStream(ctx, req, callback)
	default:
		return p.CompleteStream(ctx, req, callback)
	}
}

// completeNonStreaming forces non-streaming completion (current Complete method)
func (p *GeminiProvider) completeNonStreaming(ctx context.Context, req CompletionRequest) (*CompletionResponse, error) {
	return p.Complete(ctx, req)
}

// completeStreamingCollected forces streaming but collects all chunks into single response.
// Supports req.EarlyExit to cancel wasted tokens once a complete JSON/tool result is formed.
func (p *GeminiProvider) completeStreamingCollected(ctx context.Context, req CompletionRequest) (*CompletionResponse, error) {
	return CollectStream(ctx, p.CompleteStream, req)
}

// SupportsToolCalls returns whether the provider supports tool calls
func (p *GeminiProvider) SupportsToolCalls() bool {
	return true
}

// GetMaxTokens returns the maximum tokens for a model
func (p *GeminiProvider) GetMaxTokens(model string) int {
	switch model {
	case "gemini-pro":
		return 32768
	case "gemini-pro-vision":
		return 16384
	case "gemini-1.5-pro":
		return 128000
	case "gemini-1.5-flash":
		return 32768
	default:
		return 32768
	}
}

// Note: This is a mock implementation for demonstration purposes.
// In a real implementation, you would:
// 1. Use the official Google AI Go SDK when available
// 2. Make actual HTTP requests to the Gemini API
// 3. Handle authentication, rate limiting, and error handling properly
// 4. Implement proper streaming support
// 5. Support all Gemini features like vision, function calling, etc.
