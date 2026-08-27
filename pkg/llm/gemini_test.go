// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

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
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// geminiAPI is a stand-in for Google's Generative Language API. The provider
// under test is the real one: it builds real requests, speaks the real wire
// format and parses real responses.
type geminiAPI struct {
	server   *httptest.Server
	requests atomic.Int32

	// lastBody is the decoded body of the most recent generateContent call.
	lastBody map[string]interface{}
	lastPath string
	lastKey  string
}

func newGeminiAPI(t *testing.T, handler http.HandlerFunc) *geminiAPI {
	t.Helper()
	api := &geminiAPI{}
	api.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		api.requests.Add(1)
		api.lastPath = r.URL.Path
		api.lastKey = r.URL.Query().Get("key")

		if r.Body != nil {
			if raw, err := io.ReadAll(r.Body); err == nil && len(raw) > 0 {
				var decoded map[string]interface{}
				if json.Unmarshal(raw, &decoded) == nil {
					api.lastBody = decoded
				}
			}
		}
		handler(w, r)
	}))
	t.Cleanup(api.server.Close)
	return api
}

// provider builds a Gemini provider pointed at the fake API.
func (api *geminiAPI) provider(t *testing.T, mutate func(*ProviderConfig)) *GeminiProvider {
	t.Helper()
	cfg := DefaultProviderConfig()
	cfg.Type = "gemini"
	cfg.APIKey = "test-key" // pragma: allowlist secret
	cfg.Model = "gemini-1.5-flash"
	cfg.Endpoint = api.server.URL
	cfg.RetryCount = 0
	cfg.RetryDelay = time.Millisecond
	cfg.Timeout = 5 * time.Second
	if mutate != nil {
		mutate(cfg)
	}
	p, err := NewGeminiProvider(cfg)
	require.NoError(t, err)
	return p
}

func geminiSuccess(text string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"candidates": []map[string]interface{}{{
				"content":      map[string]interface{}{"role": "model", "parts": []map[string]string{{"text": text}}},
				"finishReason": "STOP",
				"index":        0,
			}},
			"usageMetadata": map[string]int{
				"promptTokenCount": 7, "candidatesTokenCount": 11, "totalTokenCount": 18,
			},
		})
	}
}

func geminiTestRequest() CompletionRequest {
	return CompletionRequest{
		Messages: []Message{{Role: "user", Content: "Hello, how are you?"}},
	}
}

// The provider must call the model rather than returning canned text. It
// previously returned hardcoded strings without contacting the API at all.
func TestGemini_CallsTheAPI(t *testing.T) {
	api := newGeminiAPI(t, geminiSuccess("a genuine model reply"))
	p := api.provider(t, nil)

	resp, err := p.Complete(context.Background(), geminiTestRequest())
	require.NoError(t, err)

	assert.EqualValues(t, 1, api.requests.Load(), "the provider must actually call the API")
	require.NotEmpty(t, resp.Choices)
	assert.Equal(t, "a genuine model reply", resp.Choices[0].Message.Content,
		"the reply must come from the API, not from the provider")
	assert.Equal(t, "stop", resp.Choices[0].FinishReason)
	assert.Equal(t, 7, resp.Usage.PromptTokens)
	assert.Equal(t, 11, resp.Usage.CompletionTokens)
	assert.Equal(t, 18, resp.Usage.TotalTokens)
}

// The request must target the configured model and carry the API key.
func TestGemini_RequestTargetsModelAndKey(t *testing.T) {
	api := newGeminiAPI(t, geminiSuccess("ok"))
	p := api.provider(t, func(c *ProviderConfig) { c.Model = "gemini-1.5-pro" })

	_, err := p.Complete(context.Background(), geminiTestRequest())
	require.NoError(t, err)

	assert.Contains(t, api.lastPath, "gemini-1.5-pro")
	assert.Contains(t, api.lastPath, "generateContent")
	assert.Equal(t, "test-key", api.lastKey)
}

// Gemini names the assistant role "model" and carries the system prompt in a
// dedicated field rather than as a message.
func TestGemini_MessageMapping(t *testing.T) {
	api := newGeminiAPI(t, geminiSuccess("ok"))
	p := api.provider(t, nil)

	_, err := p.Complete(context.Background(), CompletionRequest{
		SystemPrompt: "You are terse.",
		Messages: []Message{
			{Role: "system", Content: "Also be polite."},
			{Role: "user", Content: "hi"},
			{Role: "assistant", Content: "hello"},
			{Role: "user", Content: "again"},
		},
	})
	require.NoError(t, err)

	body := api.lastBody
	require.NotNil(t, body)

	system, ok := body["systemInstruction"].(map[string]interface{})
	require.True(t, ok, "system prompts belong in systemInstruction: %v", body)
	assert.Contains(t, fmt.Sprint(system), "You are terse.")
	assert.Contains(t, fmt.Sprint(system), "Also be polite.")

	contents, ok := body["contents"].([]interface{})
	require.True(t, ok)
	require.Len(t, contents, 3, "only user and assistant turns become contents")

	roles := make([]string, 0, len(contents))
	for _, c := range contents {
		roles = append(roles, fmt.Sprint(c.(map[string]interface{})["role"]))
	}
	assert.Equal(t, []string{"user", "model", "user"}, roles,
		"the assistant role must be sent as \"model\"")
}

// Generation settings must reach the API.
func TestGemini_GenerationConfig(t *testing.T) {
	api := newGeminiAPI(t, geminiSuccess("ok"))
	p := api.provider(t, nil)

	_, err := p.Complete(context.Background(), CompletionRequest{
		Messages:      []Message{{Role: "user", Content: "hi"}},
		Temperature:   0.25,
		MaxTokens:     512,
		StopSequences: []string{"END"},
	})
	require.NoError(t, err)

	cfg, ok := api.lastBody["generationConfig"].(map[string]interface{})
	require.True(t, ok, "generation settings must be sent: %v", api.lastBody)
	assert.InDelta(t, 0.25, cfg["temperature"], 1e-9)
	assert.EqualValues(t, 512, cfg["maxOutputTokens"])
	assert.Contains(t, fmt.Sprint(cfg["stopSequences"]), "END")
}

func TestGemini_EmptyMessagesRejected(t *testing.T) {
	api := newGeminiAPI(t, geminiSuccess("ok"))
	p := api.provider(t, nil)

	_, err := p.Complete(context.Background(), CompletionRequest{})
	require.Error(t, err)
	assert.Zero(t, api.requests.Load(), "an invalid request must not reach the API")
}

// API errors must be classified like any other provider's.
func TestGemini_ErrorClassification(t *testing.T) {
	cases := []struct {
		status int
		want   error
	}{
		{http.StatusTooManyRequests, ErrRateLimited},
		{http.StatusUnauthorized, ErrProviderAuth},
		{http.StatusBadRequest, ErrProviderRequest},
		{http.StatusInternalServerError, ErrProviderUnavailable},
	}

	for _, tc := range cases {
		t.Run(fmt.Sprint(tc.status), func(t *testing.T) {
			api := newGeminiAPI(t, func(w http.ResponseWriter, r *http.Request) {
				http.Error(w, "upstream said no", tc.status)
			})
			p := api.provider(t, nil)

			_, err := p.Complete(context.Background(), geminiTestRequest())
			require.Error(t, err)
			assert.True(t, errors.Is(err, tc.want), "status %d: got %v", tc.status, err)
		})
	}
}

// An error carried inside a 200 body is a permanent request error.
func TestGemini_InBodyErrorIsReported(t *testing.T) {
	api := newGeminiAPI(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"error": map[string]interface{}{"code": 400, "message": "API key not valid", "status": "INVALID_ARGUMENT"},
		})
	})
	p := api.provider(t, nil)

	_, err := p.Complete(context.Background(), geminiTestRequest())
	require.Error(t, err)
	assert.Contains(t, err.Error(), "API key not valid")
	assert.True(t, errors.Is(err, ErrProviderRequest))
}

func TestGemini_MalformedResponse(t *testing.T) {
	api := newGeminiAPI(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("{not json"))
	})
	p := api.provider(t, nil)

	_, err := p.Complete(context.Background(), geminiTestRequest())
	require.Error(t, err)
	assert.Contains(t, err.Error(), "decode")
}

// A blocked prompt returns no candidates; that must be a usable response
// rather than an index-out-of-range panic.
func TestGemini_NoCandidates(t *testing.T) {
	api := newGeminiAPI(t, func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]interface{}{"candidates": []interface{}{}})
	})
	p := api.provider(t, nil)

	resp, err := p.Complete(context.Background(), geminiTestRequest())
	require.NoError(t, err)
	require.NotEmpty(t, resp.Choices)
	assert.Empty(t, resp.Choices[0].Message.Content)
}

func TestGemini_RetriesTransientFailures(t *testing.T) {
	var calls atomic.Int32
	api := newGeminiAPI(t, func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) < 3 {
			http.Error(w, "unavailable", http.StatusServiceUnavailable)
			return
		}
		geminiSuccess("recovered")(w, r)
	})
	p := api.provider(t, func(c *ProviderConfig) { c.RetryCount = 3 })

	resp, err := p.Complete(context.Background(), geminiTestRequest())
	require.NoError(t, err)
	assert.Equal(t, "recovered", resp.Choices[0].Message.Content)
}

func TestGemini_Cancellation(t *testing.T) {
	release := make(chan struct{})
	api := newGeminiAPI(t, func(w http.ResponseWriter, r *http.Request) {
		<-release
	})
	defer close(release)

	p := api.provider(t, func(c *ProviderConfig) { c.Timeout = 30 * time.Second })

	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	start := time.Now()
	_, err := p.Complete(ctx, geminiTestRequest())
	require.Error(t, err)
	assert.Less(t, time.Since(start), 10*time.Second)
}

// Streaming must consume the server-sent event stream and deliver each chunk.
func TestGemini_Streaming(t *testing.T) {
	api := newGeminiAPI(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		for _, text := range []string{"Hello", " there", "!"} {
			frame, _ := json.Marshal(map[string]interface{}{
				"candidates": []map[string]interface{}{{
					"content": map[string]interface{}{"role": "model", "parts": []map[string]string{{"text": text}}},
					"index":   0,
				}},
			})
			_, _ = fmt.Fprintf(w, "data: %s\n\n", frame)
			if flusher != nil {
				flusher.Flush()
			}
		}
	})
	p := api.provider(t, nil)

	var chunks []string
	err := p.CompleteStream(context.Background(), geminiTestRequest(), func(chunk CompletionResponse) error {
		require.NotEmpty(t, chunk.Choices)
		chunks = append(chunks, chunk.Choices[0].Delta.Content)
		return nil
	})
	require.NoError(t, err)

	assert.Equal(t, []string{"Hello", " there", "!"}, chunks)
	assert.Contains(t, api.lastPath, "streamGenerateContent")
	assert.Equal(t, strings.Join(chunks, ""), "Hello there!")
}

// A callback that fails must stop the stream and surface its error.
func TestGemini_StreamingCallbackError(t *testing.T) {
	api := newGeminiAPI(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for i := 0; i < 5; i++ {
			frame, _ := json.Marshal(map[string]interface{}{
				"candidates": []map[string]interface{}{{
					"content": map[string]interface{}{"parts": []map[string]string{{"text": "x"}}},
				}},
			})
			_, _ = fmt.Fprintf(w, "data: %s\n\n", frame)
		}
	})
	p := api.provider(t, nil)

	sentinel := errors.New("consumer gave up")
	err := p.CompleteStream(context.Background(), geminiTestRequest(), func(chunk CompletionResponse) error {
		return sentinel
	})
	assert.ErrorIs(t, err, sentinel)
}

func TestGemini_StreamingErrorStatus(t *testing.T) {
	api := newGeminiAPI(t, func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "nope", http.StatusUnauthorized)
	})
	p := api.provider(t, nil)

	err := p.CompleteStream(context.Background(), geminiTestRequest(), func(chunk CompletionResponse) error {
		return nil
	})
	require.Error(t, err)
	assert.True(t, errors.Is(err, ErrProviderAuth))
}

// Health must reflect the API, not a hardcoded success.
func TestGemini_Health(t *testing.T) {
	t.Run("healthy", func(t *testing.T) {
		api := newGeminiAPI(t, func(w http.ResponseWriter, r *http.Request) {
			_ = json.NewEncoder(w).Encode(map[string]interface{}{"models": []interface{}{}})
		})
		assert.NoError(t, api.provider(t, nil).IsHealthy(context.Background()))
	})

	t.Run("rejected credentials", func(t *testing.T) {
		api := newGeminiAPI(t, func(w http.ResponseWriter, r *http.Request) {
			http.Error(w, "bad key", http.StatusUnauthorized)
		})
		err := api.provider(t, nil).IsHealthy(context.Background())
		require.Error(t, err, "an unhealthy provider must not report healthy")
		assert.True(t, errors.Is(err, ErrProviderAuth))
	})
}

func TestGemini_RequiresAPIKey(t *testing.T) {
	cfg := DefaultProviderConfig()
	cfg.Type = "gemini"
	_, err := NewGeminiProvider(cfg)
	assert.Error(t, err)

	_, err = NewGeminiProvider(nil)
	assert.Error(t, err)
}

// Credentials must never appear in the configuration the API exposes.
func TestGemini_ConfigMasksAPIKey(t *testing.T) {
	api := newGeminiAPI(t, geminiSuccess("ok"))
	p := api.provider(t, nil)

	cfg := p.GetConfig()
	assert.NotEqual(t, "test-key", cfg["api_key"])
	assert.NotContains(t, fmt.Sprint(cfg), "test-key")
}
