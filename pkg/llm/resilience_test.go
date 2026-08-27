// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ollamaChatResponse writes one terminal chat frame.
func ollamaChatResponse(w http.ResponseWriter, content string) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]interface{}{
		"model":      "test-model",
		"created_at": time.Now(),
		"message":    map[string]string{"role": "assistant", "content": content},
		"done":       true,
	})
}

func testOllama(t *testing.T, endpoint string, mutate func(*ProviderConfig)) *OllamaProvider {
	t.Helper()
	cfg := DefaultProviderConfig()
	cfg.Name = "ollama"
	cfg.Type = "ollama"
	cfg.Endpoint = endpoint
	cfg.Model = "test-model"
	cfg.RetryCount = 0
	cfg.RetryDelay = time.Millisecond
	cfg.Timeout = 5 * time.Second
	if mutate != nil {
		mutate(cfg)
	}
	p, err := NewOllamaProvider(cfg)
	require.NoError(t, err)
	return p
}

func simpleRequest() CompletionRequest {
	return CompletionRequest{
		Model:    "test-model",
		Messages: []Message{{Role: "user", Content: "hello"}},
	}
}

// ---------------------------------------------------------------------------
// Error classification
// ---------------------------------------------------------------------------

func TestProvider_ClassifiesHTTPStatuses(t *testing.T) {
	cases := []struct {
		status int
		want   error
		retry  bool
	}{
		{http.StatusInternalServerError, ErrProviderUnavailable, true},
		{http.StatusBadGateway, ErrProviderUnavailable, true},
		{http.StatusServiceUnavailable, ErrProviderUnavailable, true},
		{http.StatusTooManyRequests, ErrRateLimited, true},
		{http.StatusUnauthorized, ErrProviderAuth, false},
		{http.StatusForbidden, ErrProviderAuth, false},
		{http.StatusBadRequest, ErrProviderRequest, false},
		{http.StatusNotFound, ErrProviderRequest, false},
	}

	for _, tc := range cases {
		t.Run(fmt.Sprint(tc.status), func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				http.Error(w, "upstream said no", tc.status)
			}))
			defer srv.Close()

			p := testOllama(t, srv.URL, nil)
			_, err := p.Complete(context.Background(), simpleRequest())
			require.Error(t, err)
			assert.True(t, errors.Is(err, tc.want), "status %d should classify as %v, got %v", tc.status, tc.want, err)
			assert.Equal(t, tc.retry, IsRetryable(err), "retryability for status %d", tc.status)
		})
	}
}

// ---------------------------------------------------------------------------
// Retry behaviour
// ---------------------------------------------------------------------------

func TestProvider_RetriesTransientFailures(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) < 3 {
			http.Error(w, "temporarily down", http.StatusServiceUnavailable)
			return
		}
		ollamaChatResponse(w, "recovered")
	}))
	defer srv.Close()

	p := testOllama(t, srv.URL, func(c *ProviderConfig) { c.RetryCount = 3 })
	resp, err := p.Complete(context.Background(), simpleRequest())
	require.NoError(t, err, "a transient outage must be retried")
	assert.Equal(t, "recovered", resp.Choices[0].Message.Content)
	assert.EqualValues(t, 3, calls.Load())
}

func TestProvider_DoesNotRetryPermanentErrors(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		http.Error(w, "bad model", http.StatusBadRequest)
	}))
	defer srv.Close()

	p := testOllama(t, srv.URL, func(c *ProviderConfig) { c.RetryCount = 5 })
	_, err := p.Complete(context.Background(), simpleRequest())
	require.Error(t, err)
	assert.EqualValues(t, 1, calls.Load(), "a 4xx must not be retried")
}

func TestProvider_ExhaustsRetryBudget(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		http.Error(w, "still down", http.StatusInternalServerError)
	}))
	defer srv.Close()

	p := testOllama(t, srv.URL, func(c *ProviderConfig) { c.RetryCount = 2 })
	_, err := p.Complete(context.Background(), simpleRequest())
	require.Error(t, err)
	assert.True(t, errors.Is(err, ErrProviderUnavailable))
	assert.EqualValues(t, 3, calls.Load(), "initial attempt plus RetryCount retries")
}

func TestProvider_HonoursRetryAfter(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) == 1 {
			w.Header().Set("Retry-After", "1")
			http.Error(w, "slow down", http.StatusTooManyRequests)
			return
		}
		ollamaChatResponse(w, "ok")
	}))
	defer srv.Close()

	p := testOllama(t, srv.URL, func(c *ProviderConfig) {
		c.RetryCount = 2
		c.RetryDelay = time.Millisecond
	})

	start := time.Now()
	_, err := p.Complete(context.Background(), simpleRequest())
	require.NoError(t, err)
	assert.GreaterOrEqual(t, time.Since(start), time.Second,
		"a provider-supplied Retry-After must be respected over the configured delay")
}

// A cancelled context must abandon retries immediately.
func TestProvider_CancellationStopsRetries(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		http.Error(w, "down", http.StatusInternalServerError)
	}))
	defer srv.Close()

	p := testOllama(t, srv.URL, func(c *ProviderConfig) {
		c.RetryCount = 100
		c.RetryDelay = 50 * time.Millisecond
	})

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Millisecond)
	defer cancel()

	done := make(chan struct{})
	go func() {
		defer close(done)
		_, _ = p.Complete(ctx, simpleRequest())
	}()

	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("retries did not stop when the context was cancelled")
	}
	assert.Less(t, calls.Load(), int32(20), "retries must stop promptly on cancellation")
}

// ---------------------------------------------------------------------------
// Network and payload failures
// ---------------------------------------------------------------------------

func TestProvider_NetworkFailureIsRetryable(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	endpoint := srv.URL
	srv.Close() // nothing is listening now

	p := testOllama(t, endpoint, nil)
	_, err := p.Complete(context.Background(), simpleRequest())
	require.Error(t, err)
	assert.True(t, errors.Is(err, ErrProviderUnavailable), "a refused connection is transient, got %v", err)
	assert.True(t, IsRetryable(err))
}

func TestProvider_MalformedResponseIsReported(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte("{not json at all"))
	}))
	defer srv.Close()

	p := testOllama(t, srv.URL, nil)
	_, err := p.Complete(context.Background(), simpleRequest())
	require.Error(t, err)
	assert.Contains(t, err.Error(), "decode")
}

func TestProvider_TruncatedResponseIsReported(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		// A frame that never completes, then the connection ends.
		_, _ = w.Write([]byte(`{"model":"test-model","message":{"role":"assistant","content":"par`))
	}))
	defer srv.Close()

	p := testOllama(t, srv.URL, nil)
	_, err := p.Complete(context.Background(), simpleRequest())
	require.Error(t, err, "a truncated response must not be reported as success")
}

func TestProvider_APILevelErrorIsPermanent(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{"error": "model not found"})
	}))
	defer srv.Close()

	p := testOllama(t, srv.URL, func(c *ProviderConfig) { c.RetryCount = 3 })
	_, err := p.Complete(context.Background(), simpleRequest())
	require.Error(t, err)
	assert.Contains(t, err.Error(), "model not found")
	assert.EqualValues(t, 1, calls.Load(), "an API-level error must not be retried")
}

// A response body larger than the cap must not be buffered without bound.
func TestProvider_OversizedResponseIsBounded(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		flusher, _ := w.(http.Flusher)
		chunk := strings.Repeat("x", 64*1024)
		// Stream frames that never set done, far beyond any sane response.
		for i := 0; i < 200; i++ {
			_, _ = fmt.Fprintf(w, `{"model":"m","message":{"role":"assistant","content":%q},"done":false}`, chunk)
			if flusher != nil {
				flusher.Flush()
			}
		}
	}))
	defer srv.Close()

	p := testOllama(t, srv.URL, nil)
	done := make(chan struct{})
	go func() {
		defer close(done)
		_, _ = p.Complete(context.Background(), simpleRequest())
	}()

	select {
	case <-done:
	case <-time.After(30 * time.Second):
		t.Fatal("an endless response was not bounded")
	}
}

func TestProvider_TimeoutIsReported(t *testing.T) {
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-release
	}))
	defer func() {
		close(release)
		srv.Close()
	}()

	p := testOllama(t, srv.URL, func(c *ProviderConfig) { c.Timeout = 100 * time.Millisecond })
	_, err := p.Complete(context.Background(), simpleRequest())
	require.Error(t, err, "a hung provider must not block indefinitely")
}

// ---------------------------------------------------------------------------
// Manager behaviour
// ---------------------------------------------------------------------------

func TestProviderManager_ConcurrentUse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ollamaChatResponse(w, "ok")
	}))
	defer srv.Close()

	mgr := NewProviderManager()
	require.NoError(t, mgr.RegisterProvider("ollama", testOllama(t, srv.URL, nil)))

	var wg sync.WaitGroup
	errs := make([]error, 16)
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			_, errs[i] = mgr.Complete(context.Background(), "ollama", simpleRequest())
			_ = mgr.ListProviders()
			_ = mgr.HealthCheck(context.Background())
		}(i)
	}
	wg.Wait()
	require.NoError(t, errors.Join(errs...))
}

func TestProviderManager_UnknownProvider(t *testing.T) {
	mgr := NewProviderManager()
	_, err := mgr.Complete(context.Background(), "nope", simpleRequest())
	require.Error(t, err)
}

func TestProviderManager_RejectsDuplicateRegistration(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ollamaChatResponse(w, "ok")
	}))
	defer srv.Close()

	mgr := NewProviderManager()
	require.NoError(t, mgr.RegisterProvider("p", testOllama(t, srv.URL, nil)))
	assert.Error(t, mgr.RegisterProvider("p", testOllama(t, srv.URL, nil)))
}

// HealthCheck must report per-provider status rather than failing as a whole.
func TestProviderManager_HealthCheckReportsPerProvider(t *testing.T) {
	good := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]interface{}{"models": []interface{}{}})
	}))
	defer good.Close()

	bad := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "down", http.StatusInternalServerError)
	}))
	badURL := bad.URL
	bad.Close()

	mgr := NewProviderManager()
	require.NoError(t, mgr.RegisterProvider("good", testOllama(t, good.URL, nil)))
	require.NoError(t, mgr.RegisterProvider("bad", testOllama(t, badURL, nil)))

	health := mgr.HealthCheck(context.Background())
	assert.NoError(t, health["good"])
	assert.Error(t, health["bad"], "an unreachable provider must be reported unhealthy")
}
