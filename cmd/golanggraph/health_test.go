// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package main

import (
	"math"
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

// A statfs block size is a signed int64. Converting it straight to uint64 and
// multiplying turned an implausible value into an enormous "free" figure, so a
// health check meant to refuse a full disk would have reported terabytes and
// passed.
func TestAvailableBytes_RefusesImplausibleGeometry(t *testing.T) {
	for name, tc := range map[string]struct {
		bavail uint64
		bsize  int64
		want   uint64
		ok     bool
	}{
		"ordinary":        {bavail: 1000, bsize: 4096, want: 4096000, ok: true},
		"zero blocks":     {bavail: 0, bsize: 4096, want: 0, ok: true},
		"negative bsize":  {bavail: 1000, bsize: -1, ok: false},
		"zero bsize":      {bavail: 1000, bsize: 0, ok: false},
		"would overflow":  {bavail: math.MaxUint64, bsize: 4096, ok: false},
		"exactly maximum": {bavail: math.MaxUint64, bsize: 1, want: math.MaxUint64, ok: true},
	} {
		t.Run(name, func(t *testing.T) {
			got, ok := availableBytes(tc.bavail, tc.bsize)
			if ok != tc.ok {
				t.Fatalf("ok = %v, want %v", ok, tc.ok)
			}
			if ok && got != tc.want {
				t.Errorf("bytes = %d, want %d", got, tc.want)
			}
			if !ok && got != 0 {
				t.Errorf("a refused conversion must report 0, got %d", got)
			}
		})
	}
}

// A misconfigured dependency address should be reported as the
// misconfiguration it is, not as an opaque dial error.
func TestValidateProbeTarget(t *testing.T) {
	for _, good := range []string{"localhost:5432", "127.0.0.1:6379", "[::1]:11434", "db.internal:1"} {
		if err := validateProbeTarget(good); err != nil {
			t.Errorf("%q should be accepted: %v", good, err)
		}
	}
	for _, bad := range []string{"", "localhost", "localhost:", ":5432", "localhost:0", "localhost:70000", "localhost:pgsql"} {
		if err := validateProbeTarget(bad); err == nil {
			t.Errorf("%q should be rejected", bad)
		}
	}
}
