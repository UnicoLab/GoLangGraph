// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

package main

import (
	"net"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// A container with no dependencies configured is healthy. The previous
// implementation exited non-zero whenever OPENAI_API_KEY was unset, which made
// every deployment without an OpenAI key permanently unhealthy.
func TestHealth_NoDependenciesConfiguredIsHealthy(t *testing.T) {
	t.Setenv("POSTGRES_HOST", "")
	t.Setenv("REDIS_HOST", "")
	t.Setenv("OLLAMA_URL", "")
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("GEMINI_API_KEY", "")

	code := runHealthCheck(healthOptions{Timeout: time.Second})
	assert.Equal(t, 0, code, "a missing optional credential must not fail the check")
}

// With --strict, warnings do fail.
func TestHealth_StrictTreatsWarningsAsFailures(t *testing.T) {
	t.Setenv("POSTGRES_HOST", "")
	t.Setenv("REDIS_HOST", "")
	t.Setenv("OLLAMA_URL", "")
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("GEMINI_API_KEY", "")

	code := runHealthCheck(healthOptions{Timeout: time.Second, Strict: true})
	assert.Equal(t, 1, code)
}

// A configured dependency that is not listening must fail. The previous
// implementation printed "Reachable" without connecting to anything.
func TestHealth_UnreachableDependencyFails(t *testing.T) {
	// Bind and release a port so nothing is listening on it.
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	port := listener.Addr().(*net.TCPAddr).Port
	require.NoError(t, listener.Close())

	t.Setenv("POSTGRES_HOST", "127.0.0.1")
	t.Setenv("POSTGRES_PORT", itoa(port))
	t.Setenv("REDIS_HOST", "")
	t.Setenv("OLLAMA_URL", "")

	code := runHealthCheck(healthOptions{Timeout: time.Second})
	assert.Equal(t, 1, code, "an unreachable configured dependency must be reported")
}

// A dependency that is listening passes.
func TestHealth_ReachableDependencyPasses(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	defer func() { _ = listener.Close() }()
	go func() {
		for {
			conn, err := listener.Accept()
			if err != nil {
				return
			}
			_ = conn.Close()
		}
	}()

	port := listener.Addr().(*net.TCPAddr).Port
	t.Setenv("POSTGRES_HOST", "127.0.0.1")
	t.Setenv("POSTGRES_PORT", itoa(port))
	t.Setenv("REDIS_HOST", "")
	t.Setenv("OLLAMA_URL", "")

	code := runHealthCheck(healthOptions{Timeout: 2 * time.Second})
	assert.Equal(t, 0, code)
}

// The server probe reports what the endpoint says, which is what the container
// health check depends on.
func TestHealth_ServerProbe(t *testing.T) {
	t.Run("healthy", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != "/api/v1/health" {
				w.WriteHeader(http.StatusNotFound)
				return
			}
			w.WriteHeader(http.StatusOK)
		}))
		defer srv.Close()

		assert.Equal(t, 0, runHealthCheck(healthOptions{ServerURL: srv.URL, Timeout: 2 * time.Second}))
	})

	t.Run("erroring", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusServiceUnavailable)
		}))
		defer srv.Close()

		assert.Equal(t, 1, runHealthCheck(healthOptions{ServerURL: srv.URL, Timeout: 2 * time.Second}))
	})

	t.Run("unreachable", func(t *testing.T) {
		listener, err := net.Listen("tcp", "127.0.0.1:0")
		require.NoError(t, err)
		addr := listener.Addr().String()
		require.NoError(t, listener.Close())

		assert.Equal(t, 1, runHealthCheck(healthOptions{
			ServerURL: "http://" + addr, Timeout: time.Second,
		}))
	})

	t.Run("invalid URL", func(t *testing.T) {
		assert.Equal(t, 1, runHealthCheck(healthOptions{ServerURL: "://nonsense", Timeout: time.Second}))
	})
}

// Resource probes must report real measurements, not fixed strings.
func TestHealth_ResourceProbesAreMeasured(t *testing.T) {
	results := probeResources(healthOptions{MinFreeDiskBytes: 1})
	require.NotEmpty(t, results)

	for _, r := range results {
		assert.NotEmpty(t, r.Detail)
		assert.NotEqual(t, "✓ Sufficient", r.Detail)
		assert.NotEqual(t, "✓ Available", r.Detail)
	}

	// An impossible free-space requirement must fail rather than pass anyway.
	strictResults := probeResources(healthOptions{MinFreeDiskBytes: 1 << 62})
	var sawFailure bool
	for _, r := range strictResults {
		if r.Name == "Disk space" && !r.OK && !r.Warning {
			sawFailure = true
		}
	}
	assert.True(t, sawFailure, "disk space must be measured, not assumed")
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var digits []byte
	for n > 0 {
		digits = append([]byte{byte('0' + n%10)}, digits...)
		n /= 10
	}
	return string(digits)
}
