// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package llm

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/sashabaranov/go-openai"
	"github.com/sirupsen/logrus"
)

// DefaultOpenAIEndpoint is the public OpenAI API base URL. Any OpenAI-compatible
// deployment (Azure OpenAI, vLLM, a corporate gateway) is reached by setting
// ProviderConfig.Endpoint instead.
const DefaultOpenAIEndpoint = "https://api.openai.com/v1"

// defaultOpenAITimeout bounds a request when the configuration does not.
const defaultOpenAITimeout = 60 * time.Second

// OpenAIProvider implements the Provider interface for OpenAI
type OpenAIProvider struct {
	// mu guards everything below it. SetConfig can rebuild the client while
	// requests are in flight, and the model cache is shared between callers,
	// so both were data races before.
	mu         sync.RWMutex
	client     *openai.Client
	httpClient *http.Client
	config     *ProviderConfig
	models     []string
	lastSync   time.Time

	logger *logrus.Logger
}

// openAITransport injects the configured headers on every request and bounds
// the response body.
//
// ProviderConfig.Headers was accepted and then dropped, so the extra headers
// an Azure deployment or a gateway needs never reached the wire. The size cap
// gives this provider the same protection the others get from limitedBody: the
// SDK reads response bodies itself, so the only place to apply it is here.
type openAITransport struct {
	base    http.RoundTripper
	headers map[string]string
}

func (t *openAITransport) RoundTrip(req *http.Request) (*http.Response, error) {
	base := t.base
	if base == nil {
		base = http.DefaultTransport
	}

	// RoundTrip must not mutate the request it is handed.
	if len(t.headers) > 0 {
		req = req.Clone(req.Context())
		for key, value := range t.headers {
			req.Header.Set(key, value)
		}
	}

	resp, err := base.RoundTrip(req)
	if err != nil {
		return nil, err
	}
	resp.Body = cappedBody{Reader: io.LimitReader(resp.Body, MaxResponseBytes), Closer: resp.Body}
	return resp, nil
}

// cappedBody reads at most MaxResponseBytes while still closing the underlying
// body.
type cappedBody struct {
	io.Reader
	io.Closer
}

// NewOpenAIProvider creates a new OpenAI provider
func NewOpenAIProvider(config *ProviderConfig) (*OpenAIProvider, error) {
	if config == nil {
		return nil, fmt.Errorf("openai configuration is required")
	}
	if config.APIKey == "" {
		return nil, fmt.Errorf("OpenAI API key is required")
	}

	// Record the effective endpoint so GetConfig reports what the client
	// actually talks to rather than an empty string.
	if config.Endpoint == "" {
		config.Endpoint = DefaultOpenAIEndpoint
	}

	provider := &OpenAIProvider{
		config: config,
		logger: logrus.New(),
		models: []string{},
	}
	provider.rebuildClient()

	return provider, nil
}

// rebuildClient constructs the SDK client from the current configuration. The
// caller must hold the write lock (or hold no references yet, at construction).
func (p *OpenAIProvider) rebuildClient() {
	timeout := p.config.Timeout
	if timeout <= 0 {
		timeout = defaultOpenAITimeout
	}

	headers := make(map[string]string, len(p.config.Headers))
	for key, value := range p.config.Headers {
		headers[key] = value
	}

	// openai.DefaultConfig installs a bare &http.Client{}, which has no
	// timeout: ProviderConfig.Timeout was ignored and an endpoint that
	// accepted a connection and then went silent hung the caller forever.
	httpClient := &http.Client{
		Timeout:   timeout,
		Transport: &openAITransport{base: http.DefaultTransport, headers: headers},
	}

	clientConfig := openai.DefaultConfig(p.config.APIKey)
	if p.config.Endpoint != "" {
		clientConfig.BaseURL = p.config.Endpoint
	}
	clientConfig.HTTPClient = httpClient

	p.httpClient = httpClient
	p.client = openai.NewClientWithConfig(clientConfig)
}

// state returns the client together with a snapshot of the configuration, so a
// concurrent SetConfig cannot swap them halfway through a request.
func (p *OpenAIProvider) state() (*openai.Client, ProviderConfig) {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.client, *p.config
}

// isReasoningModel reports whether a model belongs to the o-series, which
// accepts a different parameter set from the chat models.
func isReasoningModel(model string) bool {
	return strings.HasPrefix(model, "o1") ||
		strings.HasPrefix(model, "o3") ||
		strings.HasPrefix(model, "o4")
}

// classifyOpenAIError maps an SDK failure onto the shared sentinels.
//
// Errors were previously wrapped with fmt.Errorf and nothing else, so a caller
// could not tell a rate limit from a malformed request and every failure was
// equally (un)retryable. The SDK surfaces the HTTP status on *openai.APIError
// and *openai.RequestError; network failures arrive as *url.Error.
func classifyOpenAIError(ctx context.Context, err error) error {
	if err == nil {
		return nil
	}

	// The caller's context ending is never the provider's fault, and retrying
	// it cannot help.
	if ctxErr := ctx.Err(); ctxErr != nil {
		return ctxErr
	}

	var apiErr *openai.APIError
	if errors.As(err, &apiErr) {
		return openAIStatusError(apiErr.HTTPStatusCode, apiErr.Error())
	}

	var reqErr *openai.RequestError
	if errors.As(err, &reqErr) {
		return openAIStatusError(reqErr.HTTPStatusCode, reqErr.Error())
	}

	// A transport failure (connection refused, DNS, TLS, client timeout) is
	// worth another attempt.
	var urlErr *url.Error
	var netErr net.Error
	if errors.As(err, &urlErr) || errors.As(err, &netErr) {
		return NewTransportError("OpenAI", err)
	}

	// Anything left is local: request validation inside the SDK, or a body
	// that did not decode. Neither improves on a retry, so it stays
	// unclassified and IsRetryable reports false.
	return fmt.Errorf("OpenAI request failed: %w", err)
}

// openAIStatusError builds a classified error from an HTTP status. The SDK
// discards the response headers, so a provider-supplied Retry-After cannot be
// honored here; WithRetry falls back to its own backoff.
func openAIStatusError(status int, message string) *ProviderError {
	kind := classifyStatus(status)
	if kind == nil {
		kind = ErrProviderUnavailable
	}
	return &ProviderError{
		Provider:   "OpenAI",
		StatusCode: status,
		Body:       message,
		kind:       kind,
	}
}

// GetName returns the provider name
func (p *OpenAIProvider) GetName() string {
	return "openai"
}

// GetModels returns available models
func (p *OpenAIProvider) GetModels(ctx context.Context) ([]string, error) {
	client, cfg := p.state()

	// Cache models for 5 minutes
	p.mu.RLock()
	cached := append([]string(nil), p.models...)
	fresh := time.Since(p.lastSync) < 5*time.Minute
	p.mu.RUnlock()

	if fresh && len(cached) > 0 {
		return cached, nil
	}

	var models []string
	attempt := func(ctx context.Context) error {
		list, err := client.ListModels(ctx)
		if err != nil {
			return classifyOpenAIError(ctx, err)
		}
		models = make([]string, len(list.Models))
		for i, model := range list.Models {
			models[i] = model.ID
		}
		return nil
	}

	if err := WithRetry(ctx, &cfg, attempt); err != nil {
		return nil, fmt.Errorf("failed to list models: %w", err)
	}

	p.mu.Lock()
	p.models = models
	p.lastSync = time.Now()
	p.mu.Unlock()

	// Hand back a copy: the cache must not be mutable through the caller.
	return append([]string(nil), models...), nil
}

// Complete generates a completion
func (p *OpenAIProvider) Complete(ctx context.Context, req CompletionRequest) (*CompletionResponse, error) {
	client, cfg := p.state()

	openaiReq, err := p.convertToOpenAIRequest(&cfg, req)
	if err != nil {
		return nil, err
	}

	// This is the non-streaming path. The SDK rejects a request that carries
	// Stream with ErrChatCompletionStreamNotSupported, so passing the flag
	// through turned an ordinary completion into a local error; the streaming
	// decision belongs to CompleteWithMode.
	openaiReq.Stream = false

	p.logger.WithFields(logrus.Fields{
		"endpoint": cfg.Endpoint,
		"model":    openaiReq.Model,
	}).Debug("Sending request to OpenAI")

	var result *CompletionResponse

	// Transient failures (network errors, 5xx, rate limits) are retried
	// according to the provider configuration; permanent errors fail fast.
	// RetryCount and RetryDelay were accepted and then ignored.
	attempt := func(ctx context.Context) error {
		resp, err := client.CreateChatCompletion(ctx, openaiReq)
		if err != nil {
			return classifyOpenAIError(ctx, err)
		}
		result = p.convertFromOpenAIResponse(resp)
		return nil
	}

	if err := WithRetry(ctx, &cfg, attempt); err != nil {
		return nil, err
	}

	return result, nil
}

// CompleteStream generates a streaming completion
func (p *OpenAIProvider) CompleteStream(ctx context.Context, req CompletionRequest, callback StreamCallback) error {
	client, cfg := p.state()

	openaiReq, err := p.convertToOpenAIRequest(&cfg, req)
	if err != nil {
		return err
	}
	openaiReq.Stream = true

	// Only opening the stream is retried: once a chunk has reached the
	// callback the caller has seen partial output, and replaying the request
	// would duplicate it.
	var stream *openai.ChatCompletionStream
	open := func(ctx context.Context) error {
		s, err := client.CreateChatCompletionStream(ctx, openaiReq)
		if err != nil {
			return classifyOpenAIError(ctx, err)
		}
		stream = s
		return nil
	}

	if err := WithRetry(ctx, &cfg, open); err != nil {
		return err
	}
	defer func() { _ = stream.Close() }()

	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		response, err := stream.Recv()
		if err != nil {
			// The end of a stream was detected by comparing err.Error() to
			// "EOF", which silently swallowed any wrapped error whose text
			// happened to match and missed a wrapped io.EOF.
			if errors.Is(err, io.EOF) || errors.Is(err, context.Canceled) {
				break
			}
			return classifyOpenAIError(ctx, err)
		}

		// Convert to our format and call callback
		converted := p.convertFromOpenAIStreamResponse(response)
		if err := callback(converted); err != nil {
			// The deferred Close tears down the HTTP body, so abandoning the
			// stream here does not leak the connection.
			//
			// Propagate early-exit unwrapped so CollectStream can treat it as success.
			if IsStreamEarlyExit(err) {
				return err
			}
			return fmt.Errorf("callback error: %w", err)
		}
	}

	return nil
}

// IsHealthy checks if the provider is healthy
func (p *OpenAIProvider) IsHealthy(ctx context.Context) error {
	client, _ := p.state()

	// Try to list models as a health check
	if _, err := client.ListModels(ctx); err != nil {
		return fmt.Errorf("OpenAI health check failed: %w", classifyOpenAIError(ctx, err))
	}
	return nil
}

// GetConfig returns provider configuration
func (p *OpenAIProvider) GetConfig() map[string]interface{} {
	p.mu.RLock()
	defer p.mu.RUnlock()

	return map[string]interface{}{
		"name":     p.config.Name,
		"type":     p.config.Type,
		"endpoint": p.config.Endpoint,
		// Never hand back the credential: this map is logged and serialized by
		// callers.
		"api_key":     "***masked***",
		"model":       p.config.Model,
		"temperature": p.config.Temperature,
		"max_tokens":  p.config.MaxTokens,
		"timeout":     p.config.Timeout,
		"retry_count": p.config.RetryCount,
		"retry_delay": p.config.RetryDelay,
	}
}

// SetConfig updates provider configuration
func (p *OpenAIProvider) SetConfig(config map[string]interface{}) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	// Endpoint, credential and timeout are baked into the SDK client at
	// construction, so changing them used to have no effect at all; the client
	// is rebuilt below when any of them moves.
	rebuild := false

	if name, ok := config["name"].(string); ok {
		p.config.Name = name
	}
	if endpoint, ok := config["endpoint"].(string); ok && endpoint != p.config.Endpoint {
		p.config.Endpoint = endpoint
		rebuild = true
	}
	if apiKey, ok := config["api_key"].(string); ok && apiKey != "" && apiKey != p.config.APIKey {
		p.config.APIKey = apiKey // pragma: allowlist secret
		rebuild = true
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
	if timeout, ok := config["timeout"].(time.Duration); ok && timeout != p.config.Timeout {
		p.config.Timeout = timeout
		rebuild = true
	}
	if retryCount, ok := config["retry_count"].(int); ok {
		p.config.RetryCount = retryCount
	}
	if retryDelay, ok := config["retry_delay"].(time.Duration); ok {
		p.config.RetryDelay = retryDelay
	}
	if headers, ok := config["headers"].(map[string]string); ok {
		p.config.Headers = headers
		rebuild = true
	}

	if rebuild {
		p.rebuildClient()
	}

	return nil
}

// Close closes the provider and cleans up resources
func (p *OpenAIProvider) Close() error {
	p.mu.RLock()
	defer p.mu.RUnlock()

	// Release keep-alive connections rather than leaving them to the finaliser.
	if p.httpClient != nil {
		p.httpClient.CloseIdleConnections()
	}
	return nil
}

// convertToOpenAIRequest converts our request format to OpenAI format
func (p *OpenAIProvider) convertToOpenAIRequest(cfg *ProviderConfig, req CompletionRequest) (openai.ChatCompletionRequest, error) {
	// Use default model if not specified
	model := req.Model
	if model == "" {
		model = cfg.Model
		if model == "" {
			model = "gpt-3.5-turbo"
		}
	}

	messages := make([]openai.ChatCompletionMessage, 0, len(req.Messages)+1)

	// CompletionRequest.SystemPrompt was dropped on the floor: a caller that
	// set it (the field every other provider honors) had its instructions
	// silently discarded.
	if req.SystemPrompt != "" {
		messages = append(messages, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleSystem,
			Content: req.SystemPrompt,
		})
	}

	for _, msg := range req.Messages {
		converted := openai.ChatCompletionMessage{
			Role:       msg.Role,
			Content:    msg.Content,
			Name:       msg.Name,
			ToolCallID: msg.ToolCallID,
		}

		// Convert tool calls
		if len(msg.ToolCalls) > 0 {
			toolCalls := make([]openai.ToolCall, len(msg.ToolCalls))
			for j, tc := range msg.ToolCalls {
				toolCalls[j] = openai.ToolCall{
					ID:   tc.ID,
					Type: openai.ToolType(tc.Type),
					Function: openai.FunctionCall{
						Name:      tc.Function.Name,
						Arguments: tc.Function.Arguments,
					},
				}
			}
			converted.ToolCalls = toolCalls
		}

		messages = append(messages, converted)
	}

	// An empty message list is a 400 from the API and a wasted round trip.
	if len(messages) == 0 {
		return openai.ChatCompletionRequest{}, fmt.Errorf("%w: no messages provided", ErrProviderRequest)
	}

	// Use default temperature and max tokens if not specified
	temperature := req.Temperature
	if temperature == 0 {
		temperature = cfg.Temperature
	}
	maxTokens := req.MaxTokens
	if maxTokens == 0 {
		maxTokens = cfg.MaxTokens
	}

	openaiReq := openai.ChatCompletionRequest{
		Model:    model,
		Messages: messages,
		Stream:   req.Stream,
		Stop:     req.StopSequences,
	}

	if isReasoningModel(model) {
		// The o-series rejects max_tokens (max_completion_tokens replaces it)
		// and any temperature other than 1. The SDK enforces both locally, so
		// sending the chat-model parameter set meant every call to a reasoning
		// model failed without ever reaching the API.
		openaiReq.MaxCompletionTokens = maxTokens
	} else {
		openaiReq.MaxTokens = maxTokens
		openaiReq.Temperature = float32(temperature)
	}

	// Convert tools
	if len(req.Tools) > 0 {
		tools := make([]openai.Tool, len(req.Tools))
		for i, tool := range req.Tools {
			tools[i] = openai.Tool{
				Type: openai.ToolType(tool.Type),
				Function: &openai.FunctionDefinition{
					Name:        tool.Function.Name,
					Description: tool.Function.Description,
					Parameters:  tool.Function.Parameters,
				},
			}
		}
		openaiReq.Tools = tools
	}

	// Handle tool choice
	if req.ToolChoice != nil {
		switch tc := req.ToolChoice.(type) {
		case string:
			// Only "auto" and "none" used to survive, so "required" — a value
			// the API accepts — was silently downgraded to the default.
			if tc != "" {
				openaiReq.ToolChoice = tc
			}
		case map[string]interface{}:
			if toolType, ok := tc["type"].(string); ok && toolType == "function" {
				if function, ok := tc["function"].(map[string]interface{}); ok {
					if name, ok := function["name"].(string); ok {
						openaiReq.ToolChoice = openai.ToolChoice{
							Type: "function",
							Function: openai.ToolFunction{
								Name: name,
							},
						}
					}
				}
			}
		}
	}

	return openaiReq, nil
}

// convertFromOpenAIResponse converts OpenAI response to our format
func (p *OpenAIProvider) convertFromOpenAIResponse(resp openai.ChatCompletionResponse) *CompletionResponse {
	choices := make([]Choice, len(resp.Choices))
	for i, choice := range resp.Choices {
		message := Message{
			Role:    choice.Message.Role,
			Content: choice.Message.Content,
			Name:    choice.Message.Name,
		}

		// Convert tool calls
		if len(choice.Message.ToolCalls) > 0 {
			toolCalls := make([]ToolCall, len(choice.Message.ToolCalls))
			for j, tc := range choice.Message.ToolCalls {
				toolCalls[j] = ToolCall{
					ID:   tc.ID,
					Type: string(tc.Type),
					Function: FunctionCall{
						Name:      tc.Function.Name,
						Arguments: tc.Function.Arguments,
					},
				}
			}
			message.ToolCalls = toolCalls
		}

		choices[i] = Choice{
			Index:        choice.Index,
			Message:      message,
			FinishReason: string(choice.FinishReason),
		}
	}

	return &CompletionResponse{
		ID:      resp.ID,
		Object:  resp.Object,
		Created: resp.Created,
		Model:   resp.Model,
		Choices: choices,
		Usage: Usage{
			PromptTokens:     resp.Usage.PromptTokens,
			CompletionTokens: resp.Usage.CompletionTokens,
			TotalTokens:      resp.Usage.TotalTokens,
		},
		SystemFingerprint: resp.SystemFingerprint,
	}
}

// convertFromOpenAIStreamResponse converts OpenAI stream response to our format
func (p *OpenAIProvider) convertFromOpenAIStreamResponse(resp openai.ChatCompletionStreamResponse) CompletionResponse {
	choices := make([]Choice, len(resp.Choices))
	for i, choice := range resp.Choices {
		delta := Message{
			Role:    choice.Delta.Role,
			Content: choice.Delta.Content,
		}

		// Convert tool calls in delta (preserve Index for stream accumulation)
		if len(choice.Delta.ToolCalls) > 0 {
			toolCalls := make([]ToolCall, len(choice.Delta.ToolCalls))
			for j, tc := range choice.Delta.ToolCalls {
				idx := j
				if tc.Index != nil {
					idx = *tc.Index
				}
				toolCalls[j] = ToolCall{
					ID:    tc.ID,
					Type:  string(tc.Type),
					Index: idx,
					Function: FunctionCall{
						Name:      tc.Function.Name,
						Arguments: tc.Function.Arguments,
					},
				}
			}
			delta.ToolCalls = toolCalls
		}

		choices[i] = Choice{
			Index:        choice.Index,
			Delta:        delta,
			FinishReason: string(choice.FinishReason),
		}
	}

	converted := CompletionResponse{
		ID:                resp.ID,
		Object:            resp.Object,
		Created:           resp.Created,
		Model:             resp.Model,
		Choices:           choices,
		SystemFingerprint: resp.SystemFingerprint,
	}

	// Usage arrives on the final chunk when stream_options.include_usage is
	// set; it used to be discarded, leaving callers with no token counts.
	if resp.Usage != nil {
		converted.Usage = Usage{
			PromptTokens:     resp.Usage.PromptTokens,
			CompletionTokens: resp.Usage.CompletionTokens,
			TotalTokens:      resp.Usage.TotalTokens,
		}
	}

	return converted
}

// GetDefaultModels returns commonly used OpenAI models
func (p *OpenAIProvider) GetDefaultModels() []string {
	return []string{
		"gpt-4",
		"gpt-4-turbo",
		"gpt-4-turbo-preview",
		"gpt-3.5-turbo",
		"gpt-3.5-turbo-16k",
		"gpt-4-vision-preview",
		"gpt-4-1106-preview",
		"gpt-3.5-turbo-1106",
	}
}

// SupportsStreaming returns true if the provider supports streaming
func (p *OpenAIProvider) SupportsStreaming() bool {
	return true
}

// GetStreamingConfig returns the current streaming configuration
func (p *OpenAIProvider) GetStreamingConfig() *StreamingConfig {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.config.Streaming == nil {
		p.config.Streaming = DefaultStreamingConfig()
	}
	return p.config.Streaming
}

// SetStreamingConfig updates the streaming configuration
func (p *OpenAIProvider) SetStreamingConfig(config *StreamingConfig) error {
	if config == nil {
		return fmt.Errorf("streaming config cannot be nil")
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	p.config.Streaming = config
	return nil
}

// CompleteWithMode generates a completion with explicit streaming mode
func (p *OpenAIProvider) CompleteWithMode(ctx context.Context, req CompletionRequest, mode StreamMode) (*CompletionResponse, error) {
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
func (p *OpenAIProvider) CompleteStreamWithMode(ctx context.Context, req CompletionRequest, callback StreamCallback, mode StreamMode) error {
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
func (p *OpenAIProvider) completeNonStreaming(ctx context.Context, req CompletionRequest) (*CompletionResponse, error) {
	return p.Complete(ctx, req)
}

// completeStreamingCollected forces streaming but collects all chunks into single response.
// When req.EarlyExit fires, the remainder of the token stream is canceled and the
// accumulated content/tool-calls are returned successfully (FinishReason=early_exit).
func (p *OpenAIProvider) completeStreamingCollected(ctx context.Context, req CompletionRequest) (*CompletionResponse, error) {
	return CollectStream(ctx, p.CompleteStream, req)
}

// SupportsToolCalls returns true if the provider supports tool calls
func (p *OpenAIProvider) SupportsToolCalls() bool {
	return true
}

// GetMaxTokens returns the maximum tokens for a model
func (p *OpenAIProvider) GetMaxTokens(model string) int {
	switch {
	case strings.Contains(model, "gpt-4-turbo"):
		return 128000
	case strings.Contains(model, "gpt-4"):
		return 8192
	case strings.Contains(model, "gpt-3.5-turbo-16k"):
		return 16384
	case strings.Contains(model, "gpt-3.5-turbo"):
		return 4096
	default:
		return 4096
	}
}

// GetTokenLimit returns the token limit for a model
func (p *OpenAIProvider) GetTokenLimit(model string) int {
	return p.GetMaxTokens(model)
}

// EstimateTokens estimates the number of tokens in a text
func (p *OpenAIProvider) EstimateTokens(text string) int {
	// Rough approximation: 1 token ≈ 4 characters for English text
	return len(text) / 4
}

// EstimateMessagesTokens estimates the number of tokens in messages
func (p *OpenAIProvider) EstimateMessagesTokens(messages []Message) int {
	total := 0
	for _, msg := range messages {
		// Add some overhead for message formatting
		total += p.EstimateTokens(msg.Content) + 10

		// Add tokens for tool calls
		for _, tc := range msg.ToolCalls {
			total += p.EstimateTokens(tc.Function.Name) + p.EstimateTokens(tc.Function.Arguments) + 20
		}
	}
	return total
}

// ValidateModel checks if a model is valid for this provider
func (p *OpenAIProvider) ValidateModel(model string) error {
	if model == "" {
		return fmt.Errorf("model cannot be empty")
	}

	// Check if it's a known OpenAI model pattern
	validPrefixes := []string{"gpt-", "text-", "code-", "davinci", "curie", "babbage", "ada", "o1", "o3", "o4", "chatgpt-"}
	for _, prefix := range validPrefixes {
		if strings.HasPrefix(model, prefix) {
			return nil
		}
	}

	return fmt.Errorf("model %s does not appear to be a valid OpenAI model", model)
}

// GetProviderInfo returns information about the provider
func (p *OpenAIProvider) GetProviderInfo() map[string]interface{} {
	return map[string]interface{}{
		"name":               "OpenAI",
		"type":               "openai",
		"supports_streaming": true,
		"supports_tools":     true,
		"supports_vision":    true,
		"max_context_length": 128000,
		"default_model":      "gpt-3.5-turbo",
		"api_version":        "v1",
	}
}
