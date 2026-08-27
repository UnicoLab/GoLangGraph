// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/sashabaranov/go-openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// openAIAPI is a stand-in for the OpenAI API. The provider under test is the
// real one: it builds real requests through the SDK, speaks the real wire
// format and parses real responses.
type openAIAPI struct {
	server   *httptest.Server
	requests atomic.Int32

	// Recorded details of the most recent request. The handler runs on the
	// server's goroutine, so access is guarded.
	mu         sync.Mutex
	lastPath   string
	lastMethod string
	lastHeader http.Header
	lastBody   map[string]interface{}
}

func newOpenAIAPI(t *testing.T, handler http.HandlerFunc) *openAIAPI {
	t.Helper()
	api := &openAIAPI{}
	api.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		api.requests.Add(1)

		var decoded map[string]interface{}
		if r.Body != nil {
			if raw, err := io.ReadAll(r.Body); err == nil && len(raw) > 0 {
				_ = json.Unmarshal(raw, &decoded)
			}
		}

		api.mu.Lock()
		api.lastPath = r.URL.Path
		api.lastMethod = r.Method
		api.lastHeader = r.Header.Clone()
		if decoded != nil {
			api.lastBody = decoded
		}
		api.mu.Unlock()

		handler(w, r)
	}))
	t.Cleanup(api.server.Close)
	return api
}

func (api *openAIAPI) body(t *testing.T) map[string]interface{} {
	t.Helper()
	api.mu.Lock()
	defer api.mu.Unlock()
	require.NotNil(t, api.lastBody, "no request body was recorded")
	return api.lastBody
}

func (api *openAIAPI) header(t *testing.T, name string) string {
	t.Helper()
	api.mu.Lock()
	defer api.mu.Unlock()
	require.NotNil(t, api.lastHeader, "no request was recorded")
	return api.lastHeader.Get(name)
}

func (api *openAIAPI) path(t *testing.T) string {
	t.Helper()
	api.mu.Lock()
	defer api.mu.Unlock()
	return api.lastPath
}

// provider builds an OpenAI provider pointed at the fake API. Pointing the SDK
// client at a test server is only possible because the provider honours
// ProviderConfig.Endpoint.
func (api *openAIAPI) provider(t *testing.T, mutate func(*ProviderConfig)) *OpenAIProvider {
	t.Helper()
	cfg := DefaultProviderConfig()
	cfg.Type = "openai"
	cfg.APIKey = "test-key" // pragma: allowlist secret
	cfg.Model = "gpt-4o-mini"
	cfg.Endpoint = api.server.URL
	cfg.RetryCount = 0
	cfg.RetryDelay = time.Millisecond
	cfg.Timeout = 5 * time.Second
	if mutate != nil {
		mutate(cfg)
	}
	p, err := NewOpenAIProvider(cfg)
	require.NoError(t, err)
	t.Cleanup(func() { _ = p.Close() })
	return p
}

// openAIChatSuccess answers with a well-formed chat completion.
func openAIChatSuccess(content string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"id":      "chatcmpl-test",
			"object":  "chat.completion",
			"created": 1700000000,
			"model":   "gpt-4o-mini",
			"choices": []map[string]interface{}{{
				"index":         0,
				"message":       map[string]interface{}{"role": "assistant", "content": content},
				"finish_reason": "stop",
			}},
			"usage": map[string]int{
				"prompt_tokens": 7, "completion_tokens": 11, "total_tokens": 18,
			},
			"system_fingerprint": "fp_test",
		})
	}
}

// openAIError answers with the API's error envelope, which the SDK decodes
// into *openai.APIError.
func openAIError(status int, message string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"error": map[string]interface{}{
				"message": message,
				"type":    "test_error",
				"code":    "test_code",
			},
		})
	}
}

// openAIStream answers with server-sent events, one content delta per chunk.
func openAIStream(chunks ...string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		for i, chunk := range chunks {
			delta := map[string]interface{}{"content": chunk}
			if i == 0 {
				delta["role"] = "assistant"
			}
			frame, _ := json.Marshal(map[string]interface{}{
				"id":      "chatcmpl-stream",
				"object":  "chat.completion.chunk",
				"created": 1700000000,
				"model":   "gpt-4o-mini",
				"choices": []map[string]interface{}{{"index": 0, "delta": delta}},
			})
			_, _ = fmt.Fprintf(w, "data: %s\n\n", frame)
			if flusher != nil {
				flusher.Flush()
			}
		}
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
		if flusher != nil {
			flusher.Flush()
		}
	}
}

// openAIRoutes dispatches between the chat and models endpoints.
func openAIRoutes(chat, models http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasSuffix(r.URL.Path, "/chat/completions"):
			chat(w, r)
		case strings.HasSuffix(r.URL.Path, "/models"):
			models(w, r)
		default:
			http.NotFound(w, r)
		}
	}
}

func openAITestRequest() CompletionRequest {
	return CompletionRequest{
		Messages: []Message{{Role: "user", Content: "Hello, how are you?"}},
	}
}

// The completion must come from the API, proving the provider actually calls it.
func TestOpenAI_CallsTheAPI(t *testing.T) {
	api := newOpenAIAPI(t, openAIChatSuccess("a genuine model reply"))
	p := api.provider(t, nil)

	resp, err := p.Complete(context.Background(), openAITestRequest())
	require.NoError(t, err)

	assert.EqualValues(t, 1, api.requests.Load(), "the provider must actually call the API")
	assert.Equal(t, "/chat/completions", api.path(t))
	require.NotEmpty(t, resp.Choices)
	assert.Equal(t, "a genuine model reply", resp.Choices[0].Message.Content)
	assert.Equal(t, "assistant", resp.Choices[0].Message.Role)
	assert.Equal(t, "stop", resp.Choices[0].FinishReason)
	assert.Equal(t, "chatcmpl-test", resp.ID)
	assert.Equal(t, "fp_test", resp.SystemFingerprint)
	assert.Equal(t, 7, resp.Usage.PromptTokens)
	assert.Equal(t, 11, resp.Usage.CompletionTokens)
	assert.Equal(t, 18, resp.Usage.TotalTokens)
}

// Model, messages and generation settings must reach the API, along with the
// credential.
func TestOpenAI_RequestShape(t *testing.T) {
	api := newOpenAIAPI(t, openAIChatSuccess("ok"))
	p := api.provider(t, nil)

	_, err := p.Complete(context.Background(), CompletionRequest{
		Model:         "gpt-4o",
		Messages:      []Message{{Role: "user", Content: "hi"}, {Role: "assistant", Content: "hello"}},
		Temperature:   0.25,
		MaxTokens:     512,
		StopSequences: []string{"END"},
	})
	require.NoError(t, err)

	body := api.body(t)
	assert.Equal(t, "gpt-4o", body["model"])
	assert.InDelta(t, 0.25, body["temperature"], 1e-6)
	assert.EqualValues(t, 512, body["max_tokens"])
	assert.Contains(t, fmt.Sprint(body["stop"]), "END")

	messages, ok := body["messages"].([]interface{})
	require.True(t, ok, "messages must be sent: %v", body)
	require.Len(t, messages, 2)
	first := messages[0].(map[string]interface{})
	assert.Equal(t, "user", first["role"])
	assert.Equal(t, "hi", first["content"])
	assert.Equal(t, "assistant", messages[1].(map[string]interface{})["role"])

	assert.Equal(t, "Bearer test-key", api.header(t, "Authorization"))
}

// The configured model is used when the request does not name one.
func TestOpenAI_FallsBackToConfiguredModel(t *testing.T) {
	api := newOpenAIAPI(t, openAIChatSuccess("ok"))
	p := api.provider(t, func(c *ProviderConfig) {
		c.Model = "gpt-4-turbo"
		c.MaxTokens = 321
		c.Temperature = 0.9
	})

	_, err := p.Complete(context.Background(), openAITestRequest())
	require.NoError(t, err)

	body := api.body(t)
	assert.Equal(t, "gpt-4-turbo", body["model"])
	assert.EqualValues(t, 321, body["max_tokens"])
	assert.InDelta(t, 0.9, body["temperature"], 1e-6)
}

// CompletionRequest.SystemPrompt is part of the shared request type and was
// dropped entirely by this provider, so the caller's instructions never
// reached the model.
func TestOpenAI_SystemPromptIsSent(t *testing.T) {
	api := newOpenAIAPI(t, openAIChatSuccess("ok"))
	p := api.provider(t, nil)

	_, err := p.Complete(context.Background(), CompletionRequest{
		SystemPrompt: "You are terse.",
		Messages:     []Message{{Role: "user", Content: "hi"}},
	})
	require.NoError(t, err)

	messages, ok := api.body(t)["messages"].([]interface{})
	require.True(t, ok)
	require.Len(t, messages, 2, "the system prompt must be prepended as a message")
	first := messages[0].(map[string]interface{})
	assert.Equal(t, "system", first["role"])
	assert.Equal(t, "You are terse.", first["content"])
}

// Reasoning models reject max_tokens and a non-default temperature; the SDK
// enforces that locally, so sending the chat-model parameter set meant every
// o-series call failed without ever reaching the API.
func TestOpenAI_ReasoningModelParameters(t *testing.T) {
	api := newOpenAIAPI(t, openAIChatSuccess("thought about it"))
	p := api.provider(t, nil)

	resp, err := p.Complete(context.Background(), CompletionRequest{
		Model:       "o3-mini",
		Messages:    []Message{{Role: "user", Content: "hi"}},
		MaxTokens:   256,
		Temperature: 0.7,
	})
	require.NoError(t, err, "a reasoning model request must reach the API")
	assert.Equal(t, "thought about it", resp.Choices[0].Message.Content)

	body := api.body(t)
	assert.EqualValues(t, 256, body["max_completion_tokens"])
	assert.NotContains(t, body, "max_tokens", "max_tokens is rejected by reasoning models")
	assert.NotContains(t, body, "temperature", "reasoning models only accept the default temperature")
}

// Complete is the non-streaming path. The SDK refuses a request that carries
// the stream flag, so passing it through turned an ordinary completion into a
// local error before any request was made.
func TestOpenAI_CompleteIgnoresStreamFlag(t *testing.T) {
	api := newOpenAIAPI(t, openAIChatSuccess("ok"))
	p := api.provider(t, nil)

	req := openAITestRequest()
	req.Stream = true

	resp, err := p.Complete(context.Background(), req)
	require.NoError(t, err)
	assert.Equal(t, "ok", resp.Choices[0].Message.Content)
	assert.EqualValues(t, 1, api.requests.Load())

	assert.NotContains(t, api.body(t), "stream", "Complete must not ask for a stream")
}

// Tools go out on the wire and tool calls come back parsed.
func TestOpenAI_ToolCalls(t *testing.T) {
	api := newOpenAIAPI(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"id":     "chatcmpl-tools",
			"object": "chat.completion",
			"model":  "gpt-4o-mini",
			"choices": []map[string]interface{}{{
				"index": 0,
				"message": map[string]interface{}{
					"role": "assistant",
					"tool_calls": []map[string]interface{}{{
						"id":       "call_1",
						"type":     "function",
						"function": map[string]string{"name": "get_weather", "arguments": `{"city":"Paris"}`},
					}},
				},
				"finish_reason": "tool_calls",
			}},
		})
	})
	p := api.provider(t, nil)

	resp, err := p.Complete(context.Background(), CompletionRequest{
		Messages: []Message{{Role: "user", Content: "weather?"}},
		Tools: []ToolDefinition{{
			Type: "function",
			Function: Function{
				Name:        "get_weather",
				Description: "look up the weather",
				Parameters:  map[string]interface{}{"type": "object"},
			},
		}},
		// "required" is a value the API accepts; it used to be discarded
		// because only "auto" and "none" were passed through.
		ToolChoice: "required",
	})
	require.NoError(t, err)

	body := api.body(t)
	tools, ok := body["tools"].([]interface{})
	require.True(t, ok, "tools must be sent: %v", body)
	require.Len(t, tools, 1)
	assert.Equal(t, "required", body["tool_choice"])

	require.NotEmpty(t, resp.Choices)
	require.Len(t, resp.Choices[0].Message.ToolCalls, 1)
	call := resp.Choices[0].Message.ToolCalls[0]
	assert.Equal(t, "call_1", call.ID)
	assert.Equal(t, "function", call.Type)
	assert.Equal(t, "get_weather", call.Function.Name)
	assert.JSONEq(t, `{"city":"Paris"}`, call.Function.Arguments)
	assert.Equal(t, "tool_calls", resp.Choices[0].FinishReason)
}

// Configured headers reach the wire; they used to be accepted and dropped,
// leaving no way to address a gateway that needs them.
func TestOpenAI_CustomHeaders(t *testing.T) {
	api := newOpenAIAPI(t, openAIChatSuccess("ok"))
	p := api.provider(t, func(c *ProviderConfig) {
		c.Headers = map[string]string{"X-Gateway-Tenant": "acme"}
	})

	_, err := p.Complete(context.Background(), openAITestRequest())
	require.NoError(t, err)
	assert.Equal(t, "acme", api.header(t, "X-Gateway-Tenant"))
}

// Errors must be classified so callers can tell a rate limit from a bad
// request instead of matching on message text.
func TestOpenAI_ErrorClassification(t *testing.T) {
	cases := []struct {
		status int
		want   error
	}{
		{http.StatusBadRequest, ErrProviderRequest},
		{http.StatusUnauthorized, ErrProviderAuth},
		{http.StatusForbidden, ErrProviderAuth},
		{http.StatusTooManyRequests, ErrRateLimited},
		{http.StatusInternalServerError, ErrProviderUnavailable},
		{http.StatusServiceUnavailable, ErrProviderUnavailable},
	}

	for _, tc := range cases {
		t.Run(fmt.Sprint(tc.status), func(t *testing.T) {
			api := newOpenAIAPI(t, openAIError(tc.status, "upstream said no"))
			p := api.provider(t, nil)

			_, err := p.Complete(context.Background(), openAITestRequest())
			require.Error(t, err)
			assert.True(t, errors.Is(err, tc.want), "status %d: got %v", tc.status, err)
			assert.Contains(t, err.Error(), "upstream said no", "the provider's message must survive")

			var pe *ProviderError
			require.True(t, errors.As(err, &pe), "the HTTP status must be recoverable")
			assert.Equal(t, tc.status, pe.StatusCode)
		})
	}
}

// A non-JSON error body arrives as a different SDK error type; it must still
// be classified by status.
func TestOpenAI_ErrorClassificationWithoutJSONBody(t *testing.T) {
	api := newOpenAIAPI(t, func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "502 Bad Gateway from the proxy", http.StatusBadGateway)
	})
	p := api.provider(t, nil)

	_, err := p.Complete(context.Background(), openAITestRequest())
	require.Error(t, err)
	assert.True(t, errors.Is(err, ErrProviderUnavailable), "got %v", err)
	assert.True(t, IsRetryable(err))
}

// Transient failures are retried according to the configuration; RetryCount
// and RetryDelay used to be accepted and ignored.
func TestOpenAI_RetriesTransientFailures(t *testing.T) {
	var calls atomic.Int32
	api := newOpenAIAPI(t, func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) < 3 {
			openAIError(http.StatusServiceUnavailable, "try later")(w, r)
			return
		}
		openAIChatSuccess("recovered")(w, r)
	})
	p := api.provider(t, func(c *ProviderConfig) { c.RetryCount = 3 })

	resp, err := p.Complete(context.Background(), openAITestRequest())
	require.NoError(t, err)
	assert.Equal(t, "recovered", resp.Choices[0].Message.Content)
	assert.EqualValues(t, 3, api.requests.Load(), "the failed attempts must be retried")
}

// A rate limit is transient too.
func TestOpenAI_RetriesRateLimit(t *testing.T) {
	var calls atomic.Int32
	api := newOpenAIAPI(t, func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) == 1 {
			openAIError(http.StatusTooManyRequests, "slow down")(w, r)
			return
		}
		openAIChatSuccess("second time lucky")(w, r)
	})
	p := api.provider(t, func(c *ProviderConfig) { c.RetryCount = 2 })

	resp, err := p.Complete(context.Background(), openAITestRequest())
	require.NoError(t, err)
	assert.Equal(t, "second time lucky", resp.Choices[0].Message.Content)
	assert.EqualValues(t, 2, api.requests.Load())
}

// A permanent failure must fail fast: retrying a malformed request only burns
// the budget and delays the error.
func TestOpenAI_DoesNotRetryPermanentFailures(t *testing.T) {
	for _, status := range []int{http.StatusBadRequest, http.StatusUnauthorized} {
		t.Run(fmt.Sprint(status), func(t *testing.T) {
			api := newOpenAIAPI(t, openAIError(status, "nope"))
			p := api.provider(t, func(c *ProviderConfig) { c.RetryCount = 5 })

			_, err := p.Complete(context.Background(), openAITestRequest())
			require.Error(t, err)
			assert.False(t, IsRetryable(err))
			assert.EqualValues(t, 1, api.requests.Load(), "a permanent failure must not be retried")
		})
	}
}

// The retry budget is finite: once it is spent the classified error surfaces.
func TestOpenAI_RetryBudgetExhausted(t *testing.T) {
	api := newOpenAIAPI(t, openAIError(http.StatusServiceUnavailable, "still down"))
	p := api.provider(t, func(c *ProviderConfig) { c.RetryCount = 2 })

	_, err := p.Complete(context.Background(), openAITestRequest())
	require.Error(t, err)
	assert.True(t, errors.Is(err, ErrProviderUnavailable))
	assert.EqualValues(t, 3, api.requests.Load(), "one initial attempt plus two retries")
}

// Cancelling the caller's context must stop the call promptly and must not be
// retried.
func TestOpenAI_Cancellation(t *testing.T) {
	release := make(chan struct{})
	api := newOpenAIAPI(t, func(w http.ResponseWriter, r *http.Request) {
		<-release
	})
	defer close(release)

	p := api.provider(t, func(c *ProviderConfig) {
		c.Timeout = 30 * time.Second
		c.RetryCount = 3
		c.RetryDelay = time.Second
	})

	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	start := time.Now()
	_, err := p.Complete(ctx, openAITestRequest())
	require.Error(t, err)
	assert.True(t, errors.Is(err, context.DeadlineExceeded), "got %v", err)
	assert.Less(t, time.Since(start), 5*time.Second, "cancellation must not wait out the retry budget")
	assert.EqualValues(t, 1, api.requests.Load(), "a cancelled call must not be retried")
}

// ProviderConfig.Timeout must bound the request. The SDK installs a client
// with no timeout of its own, so an endpoint that accepted the connection and
// then went silent hung the caller indefinitely.
func TestOpenAI_TimeoutIsApplied(t *testing.T) {
	api := newOpenAIAPI(t, func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-r.Context().Done():
		case <-time.After(3 * time.Second):
		}
	})
	p := api.provider(t, func(c *ProviderConfig) { c.Timeout = 100 * time.Millisecond })

	start := time.Now()
	_, err := p.Complete(context.Background(), openAITestRequest())
	elapsed := time.Since(start)

	require.Error(t, err)
	assert.Less(t, elapsed, 2*time.Second, "the configured timeout must bound the request")
	assert.True(t, errors.Is(err, ErrProviderUnavailable), "a timeout is transient: %v", err)
}

// A body that does not decode is a real error, and retrying it will not help.
func TestOpenAI_MalformedResponse(t *testing.T) {
	api := newOpenAIAPI(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte("{not json"))
	})
	p := api.provider(t, func(c *ProviderConfig) { c.RetryCount = 3 })

	_, err := p.Complete(context.Background(), openAITestRequest())
	require.Error(t, err)
	assert.Contains(t, err.Error(), "OpenAI")
	assert.Contains(t, err.Error(), "invalid character")
	assert.False(t, IsRetryable(err))
	assert.EqualValues(t, 1, api.requests.Load())
}

// An empty request is rejected locally rather than spending a round trip.
func TestOpenAI_EmptyMessagesRejected(t *testing.T) {
	api := newOpenAIAPI(t, openAIChatSuccess("ok"))
	p := api.provider(t, nil)

	_, err := p.Complete(context.Background(), CompletionRequest{})
	require.Error(t, err)
	assert.True(t, errors.Is(err, ErrProviderRequest))
	assert.Zero(t, api.requests.Load(), "an invalid request must not reach the API")
}

// Streaming must deliver every chunk in order.
func TestOpenAI_Streaming(t *testing.T) {
	api := newOpenAIAPI(t, openAIStream("Hello", " there", "!"))
	p := api.provider(t, nil)

	var chunks []string
	err := p.CompleteStream(context.Background(), openAITestRequest(), func(chunk CompletionResponse) error {
		require.NotEmpty(t, chunk.Choices)
		chunks = append(chunks, chunk.Choices[0].Delta.Content)
		return nil
	})
	require.NoError(t, err)

	assert.Equal(t, []string{"Hello", " there", "!"}, chunks)
	assert.Equal(t, "Hello there!", strings.Join(chunks, ""))
	assert.Equal(t, "/chat/completions", api.path(t))
	assert.Equal(t, true, api.body(t)["stream"], "a streaming call must ask for a stream")
}

// Opening a stream is retried like any other transient failure; no chunk has
// been delivered yet, so replaying the request is safe.
func TestOpenAI_StreamingRetriesOnOpen(t *testing.T) {
	var calls atomic.Int32
	api := newOpenAIAPI(t, func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) == 1 {
			openAIError(http.StatusServiceUnavailable, "warming up")(w, r)
			return
		}
		openAIStream("late", " but", " here")(w, r)
	})
	p := api.provider(t, func(c *ProviderConfig) { c.RetryCount = 2 })

	var got strings.Builder
	err := p.CompleteStream(context.Background(), openAITestRequest(), func(chunk CompletionResponse) error {
		got.WriteString(chunk.Choices[0].Delta.Content)
		return nil
	})
	require.NoError(t, err)
	assert.Equal(t, "late but here", got.String())
	assert.EqualValues(t, 2, api.requests.Load())
}

// A rejected streaming request must surface a classified error, not a bare
// wrapped string.
func TestOpenAI_StreamingErrorStatus(t *testing.T) {
	api := newOpenAIAPI(t, openAIError(http.StatusUnauthorized, "bad key"))
	p := api.provider(t, func(c *ProviderConfig) { c.RetryCount = 2 })

	called := 0
	err := p.CompleteStream(context.Background(), openAITestRequest(), func(chunk CompletionResponse) error {
		called++
		return nil
	})
	require.Error(t, err)
	assert.True(t, errors.Is(err, ErrProviderAuth), "got %v", err)
	assert.Zero(t, called)
	assert.EqualValues(t, 1, api.requests.Load(), "an auth failure must not be retried")
}

// A callback that fails aborts the stream, surfaces its error, and closes the
// HTTP body — the server sees the request context end.
func TestOpenAI_StreamingCallbackError(t *testing.T) {
	serverDone := make(chan struct{})
	abandoned := make(chan struct{})
	var once sync.Once

	api := newOpenAIAPI(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		frame := `{"id":"chatcmpl-stream","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"x"}}]}`
		_, _ = fmt.Fprintf(w, "data: %s\n\n", frame)
		if flusher != nil {
			flusher.Flush()
		}

		// Hold the response open: the client abandoning the stream is what
		// ends the request context.
		select {
		case <-r.Context().Done():
			once.Do(func() { close(abandoned) })
		case <-serverDone:
		}
	})
	defer close(serverDone)

	p := api.provider(t, nil)

	sentinel := errors.New("consumer gave up")
	calls := 0
	err := p.CompleteStream(context.Background(), openAITestRequest(), func(chunk CompletionResponse) error {
		calls++
		return sentinel
	})
	require.Error(t, err)
	assert.ErrorIs(t, err, sentinel)
	assert.Equal(t, 1, calls, "the stream must stop at the first callback error")

	select {
	case <-abandoned:
	case <-time.After(5 * time.Second):
		t.Fatal("the stream was not closed after the callback failed")
	}
}

// The API reports a mid-generation failure as an error frame inside an
// otherwise successful stream. That must surface as a classified error rather
// than a silently truncated answer.
func TestOpenAI_StreamingInBandError(t *testing.T) {
	api := newOpenAIAPI(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		_, _ = fmt.Fprint(w, `data: {"id":"c","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"partial"}}]}`+"\n\n")
		if flusher != nil {
			flusher.Flush()
		}
		_, _ = fmt.Fprint(w, `data: {"error":{"message":"the server had a problem","type":"server_error","code":null}}`+"\n\n")
		if flusher != nil {
			flusher.Flush()
		}
	})
	p := api.provider(t, nil)

	var delivered []string
	err := p.CompleteStream(context.Background(), openAITestRequest(), func(chunk CompletionResponse) error {
		delivered = append(delivered, chunk.Choices[0].Delta.Content)
		return nil
	})
	require.Error(t, err, "a failed generation must not look like a complete one")
	assert.Equal(t, []string{"partial"}, delivered)
	assert.Contains(t, err.Error(), "the server had a problem")

	var pe *ProviderError
	assert.True(t, errors.As(err, &pe), "an in-band stream error must still be classified: %v", err)
}

// Streaming collected into a single response must assemble the whole text and
// carry a usable assistant message.
func TestOpenAI_CompleteWithModeForcedStreaming(t *testing.T) {
	api := newOpenAIAPI(t, openAIStream("one ", "two ", "three"))
	p := api.provider(t, nil)

	resp, err := p.CompleteWithMode(context.Background(), openAITestRequest(), StreamModeForced)
	require.NoError(t, err)
	require.NotNil(t, resp)
	require.NotEmpty(t, resp.Choices)
	assert.Equal(t, "one two three", resp.Choices[0].Message.Content)
	assert.Equal(t, "assistant", resp.Choices[0].Message.Role)
	assert.Equal(t, "chat.completion", resp.Object)
	assert.Empty(t, resp.Choices[0].Delta.Content)
}

// Health must reflect the API rather than a hardcoded success.
func TestOpenAI_Health(t *testing.T) {
	t.Run("healthy", func(t *testing.T) {
		api := newOpenAIAPI(t, openAIRoutes(openAIChatSuccess("ok"), func(w http.ResponseWriter, r *http.Request) {
			_ = json.NewEncoder(w).Encode(map[string]interface{}{
				"object": "list",
				"data":   []map[string]interface{}{{"id": "gpt-4o-mini", "object": "model"}},
			})
		}))
		p := api.provider(t, nil)

		require.NoError(t, p.IsHealthy(context.Background()))
		assert.EqualValues(t, 1, api.requests.Load(), "the health check must call the API")
		assert.Equal(t, "/models", api.path(t))
	})

	t.Run("rejected credentials", func(t *testing.T) {
		api := newOpenAIAPI(t, openAIError(http.StatusUnauthorized, "bad key"))
		p := api.provider(t, nil)

		err := p.IsHealthy(context.Background())
		require.Error(t, err, "an unhealthy provider must not report healthy")
		assert.True(t, errors.Is(err, ErrProviderAuth), "got %v", err)
	})

	t.Run("unreachable", func(t *testing.T) {
		api := newOpenAIAPI(t, openAIChatSuccess("ok"))
		p := api.provider(t, nil)
		api.server.Close() // the endpoint stops answering

		err := p.IsHealthy(context.Background())
		require.Error(t, err)
		assert.True(t, errors.Is(err, ErrProviderUnavailable), "got %v", err)
	})
}

// GetModels reports what the API returns, and caches it.
func TestOpenAI_GetModels(t *testing.T) {
	api := newOpenAIAPI(t, openAIRoutes(openAIChatSuccess("ok"), func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"object": "list",
			"data": []map[string]interface{}{
				{"id": "gpt-4o", "object": "model"},
				{"id": "o3-mini", "object": "model"},
			},
		})
	}))
	p := api.provider(t, nil)

	models, err := p.GetModels(context.Background())
	require.NoError(t, err)
	assert.Equal(t, []string{"gpt-4o", "o3-mini"}, models)

	// Mutating the returned slice must not corrupt the cache.
	models[0] = "tampered"

	again, err := p.GetModels(context.Background())
	require.NoError(t, err)
	assert.Equal(t, []string{"gpt-4o", "o3-mini"}, again)
	assert.EqualValues(t, 1, api.requests.Load(), "the model list must be cached")
}

func TestOpenAI_GetModelsErrorIsClassified(t *testing.T) {
	api := newOpenAIAPI(t, openAIError(http.StatusUnauthorized, "bad key"))
	p := api.provider(t, nil)

	_, err := p.GetModels(context.Background())
	require.Error(t, err)
	assert.True(t, errors.Is(err, ErrProviderAuth), "got %v", err)
}

// The endpoint must be honoured, and reported, so a gateway or Azure
// deployment can be addressed at all.
func TestOpenAI_EndpointIsHonoured(t *testing.T) {
	api := newOpenAIAPI(t, openAIChatSuccess("from the gateway"))
	p := api.provider(t, nil)

	resp, err := p.Complete(context.Background(), openAITestRequest())
	require.NoError(t, err)
	assert.Equal(t, "from the gateway", resp.Choices[0].Message.Content)
	assert.Equal(t, api.server.URL, p.GetConfig()["endpoint"])
}

// Without an endpoint the provider must still report where it points.
func TestOpenAI_DefaultEndpointReported(t *testing.T) {
	p, err := NewOpenAIProvider(&ProviderConfig{Type: "openai", APIKey: "test-key"}) // pragma: allowlist secret
	require.NoError(t, err)
	t.Cleanup(func() { _ = p.Close() })

	assert.Equal(t, DefaultOpenAIEndpoint, p.GetConfig()["endpoint"])
}

// Re-pointing the provider must take effect: the endpoint and credential are
// baked into the SDK client, so updating them without rebuilding it silently
// did nothing.
func TestOpenAI_SetConfigRepointsClient(t *testing.T) {
	first := newOpenAIAPI(t, openAIChatSuccess("from the first server"))
	second := newOpenAIAPI(t, openAIChatSuccess("from the second server"))
	p := first.provider(t, nil)

	resp, err := p.Complete(context.Background(), openAITestRequest())
	require.NoError(t, err)
	assert.Equal(t, "from the first server", resp.Choices[0].Message.Content)

	require.NoError(t, p.SetConfig(map[string]interface{}{
		"endpoint": second.server.URL,
		"api_key":  "rotated-key", // pragma: allowlist secret
	}))

	resp, err = p.Complete(context.Background(), openAITestRequest())
	require.NoError(t, err)
	assert.Equal(t, "from the second server", resp.Choices[0].Message.Content)
	assert.Equal(t, "Bearer rotated-key", second.header(t, "Authorization"))
	assert.EqualValues(t, 1, first.requests.Load(), "no further traffic to the old endpoint")
}

// Credentials must never appear in the configuration the API exposes.
func TestOpenAI_ConfigMasksAPIKey(t *testing.T) {
	api := newOpenAIAPI(t, openAIChatSuccess("ok"))
	p := api.provider(t, nil)

	cfg := p.GetConfig()
	require.Contains(t, cfg, "api_key", "the credential must be reported as masked, not omitted")
	assert.Equal(t, "***masked***", cfg["api_key"])
	assert.NotContains(t, fmt.Sprint(cfg), "test-key")
}

func TestOpenAI_RequiresAPIKey(t *testing.T) {
	_, err := NewOpenAIProvider(&ProviderConfig{Type: "openai", Model: "gpt-4o"})
	assert.Error(t, err, "an empty API key must be rejected at construction")

	_, err = NewOpenAIProvider(nil)
	assert.Error(t, err, "a nil configuration must not panic")
}

// o-series models are valid OpenAI models; they used to be rejected by the
// prefix list.
func TestOpenAI_ValidateModel(t *testing.T) {
	p, err := NewOpenAIProvider(&ProviderConfig{Type: "openai", APIKey: "test-key"}) // pragma: allowlist secret
	require.NoError(t, err)
	t.Cleanup(func() { _ = p.Close() })

	for _, model := range []string{"gpt-4o", "o1-preview", "o3-mini", "o4-mini"} {
		assert.NoError(t, p.ValidateModel(model), "model %s", model)
	}
	assert.Error(t, p.ValidateModel(""))
	assert.Error(t, p.ValidateModel("llama2"))
}

// The three defects above are all consequences of SDK behaviour that is easy
// to get wrong. Pinning it here documents why the provider translates the
// request the way it does, and tells us when a workaround can be dropped.
func TestOpenAI_SDKConstraintsThisProviderWorksAround(t *testing.T) {
	api := newOpenAIAPI(t, openAIChatSuccess("ok"))
	client := openai.NewClientWithConfig(func() openai.ClientConfig {
		c := openai.DefaultConfig("test-key") // pragma: allowlist secret
		c.BaseURL = api.server.URL
		return c
	}())

	t.Run("max_tokens is rejected for reasoning models", func(t *testing.T) {
		_, err := client.CreateChatCompletion(context.Background(), openai.ChatCompletionRequest{
			Model:     "o3-mini",
			Messages:  []openai.ChatCompletionMessage{{Role: "user", Content: "hi"}},
			MaxTokens: 256,
		})
		require.Error(t, err, "sending max_tokens to an o-series model fails before the request leaves the process")
		assert.Zero(t, api.requests.Load())
	})

	t.Run("the stream flag is rejected on the completion call", func(t *testing.T) {
		_, err := client.CreateChatCompletion(context.Background(), openai.ChatCompletionRequest{
			Model:    "gpt-4o-mini",
			Messages: []openai.ChatCompletionMessage{{Role: "user", Content: "hi"}},
			Stream:   true,
		})
		assert.ErrorIs(t, err, openai.ErrChatCompletionStreamNotSupported)
	})

	t.Run("the default client has no timeout", func(t *testing.T) {
		httpClient, ok := openai.DefaultConfig("test-key").HTTPClient.(*http.Client) // pragma: allowlist secret
		require.True(t, ok)
		assert.Zero(t, httpClient.Timeout, "the provider must install its own timeout")
	})
}

// The provider is shared between goroutines by the agent runtime; the model
// cache and the configuration were unsynchronised. Run with -race.
func TestOpenAI_ConcurrentUse(t *testing.T) {
	api := newOpenAIAPI(t, openAIRoutes(openAIChatSuccess("ok"), func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"object": "list",
			"data":   []map[string]interface{}{{"id": "gpt-4o", "object": "model"}},
		})
	}))
	p := api.provider(t, nil)

	var wg sync.WaitGroup
	errs := make(chan error, 64)

	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			if _, err := p.Complete(context.Background(), openAITestRequest()); err != nil {
				errs <- err
			}
			if _, err := p.GetModels(context.Background()); err != nil {
				errs <- err
			}
			_ = p.GetConfig()
			_ = p.GetStreamingConfig()
			// Rebuilds the client while other goroutines are mid-request.
			if err := p.SetConfig(map[string]interface{}{
				"model":   fmt.Sprintf("gpt-4o-%d", i),
				"timeout": time.Duration(5+i) * time.Second,
			}); err != nil {
				errs <- err
			}
		}(i)
	}

	wg.Wait()
	close(errs)
	for err := range errs {
		t.Errorf("concurrent use failed: %v", err)
	}
}
