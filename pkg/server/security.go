// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package server

import (
	"crypto/subtle"
	"net/http"
	"runtime/debug"
	"strings"
	"sync"

	"github.com/sirupsen/logrus"
)

// SecurityConfig controls authentication, cross-origin access and request
// limits. The zero value is permissive so existing embedded uses keep working;
// production deployments should set RequireAuth and AllowedOrigins.
type SecurityConfig struct {
	// RequireAuth rejects requests without a valid X-API-Key.
	RequireAuth bool `json:"require_auth" yaml:"require_auth"`
	// APIKeys are the accepted values for the X-API-Key header.
	APIKeys []string `json:"api_keys" yaml:"api_keys"`
	// AllowedOrigins restricts CORS and WebSocket origins. Empty means any
	// origin is accepted, which is only appropriate for local development.
	AllowedOrigins []string `json:"allowed_origins" yaml:"allowed_origins"`
	// MaxRequestBytes caps request bodies. Zero applies DefaultMaxRequestBytes.
	MaxRequestBytes int64 `json:"max_request_bytes" yaml:"max_request_bytes"`
	// PublicPaths bypass authentication (health checks, readiness probes).
	PublicPaths []string `json:"public_paths" yaml:"public_paths"`
}

// DefaultMaxRequestBytes bounds request bodies so a single client cannot
// exhaust server memory with an oversized payload.
const DefaultMaxRequestBytes int64 = 4 << 20 // 4 MiB

// DefaultSecurityConfig returns a development-friendly configuration: no auth,
// any origin, but with a request size limit already in place.
func DefaultSecurityConfig() *SecurityConfig {
	return &SecurityConfig{
		RequireAuth:     false,
		MaxRequestBytes: DefaultMaxRequestBytes,
		PublicPaths:     []string{"/api/v1/health", "/health"},
	}
}

// maxBytes returns the effective request size limit.
func (c *SecurityConfig) maxBytes() int64 {
	if c == nil || c.MaxRequestBytes <= 0 {
		return DefaultMaxRequestBytes
	}
	return c.MaxRequestBytes
}

// isPublic reports whether a path bypasses authentication.
func (c *SecurityConfig) isPublic(path string) bool {
	if c == nil {
		return false
	}
	for _, p := range c.PublicPaths {
		if p == path {
			return true
		}
	}
	return false
}

// authorized reports whether a presented key matches a configured key. The
// comparison is constant time so a caller cannot recover a key by timing.
func (c *SecurityConfig) authorized(presented string) bool {
	if c == nil || !c.RequireAuth {
		return true
	}
	if presented == "" || len(c.APIKeys) == 0 {
		return false
	}
	var ok bool
	for _, key := range c.APIKeys {
		if subtle.ConstantTimeCompare([]byte(key), []byte(presented)) == 1 {
			ok = true
		}
	}
	return ok
}

// originAllowed reports whether an Origin header may access the API.
//
// An empty AllowedOrigins list accepts any origin, matching the previous
// permissive behaviour for local development. A request with no Origin header
// is not a browser cross-origin request and is always allowed.
func (c *SecurityConfig) originAllowed(origin string) bool {
	if origin == "" {
		return true
	}
	if c == nil || len(c.AllowedOrigins) == 0 {
		return true
	}
	for _, allowed := range c.AllowedOrigins {
		if allowed == "*" {
			return true
		}
		if strings.EqualFold(allowed, origin) {
			return true
		}
	}
	return false
}

// corsOrigin returns the value to echo in Access-Control-Allow-Origin.
func (c *SecurityConfig) corsOrigin(origin string) string {
	if c == nil || len(c.AllowedOrigins) == 0 {
		return "*"
	}
	if c.originAllowed(origin) && origin != "" {
		return origin
	}
	return ""
}

// ---------------------------------------------------------------------------
// Middleware
// ---------------------------------------------------------------------------

// recoveryMiddleware converts a panicking handler into a 500 response instead
// of tearing down the connection, and logs the stack for diagnosis. The stack
// is never written to the response.
func recoveryMiddleware(logger *logrus.Logger) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			defer func() {
				if rec := recover(); rec != nil {
					if logger != nil {
						logger.WithFields(logrus.Fields{
							"panic":  rec,
							"path":   r.URL.Path,
							"method": r.Method,
							"stack":  string(debug.Stack()),
						}).Error("Recovered from handler panic")
					}
					w.Header().Set("Content-Type", "application/json")
					w.WriteHeader(http.StatusInternalServerError)
					_, _ = w.Write([]byte(`{"error":"internal server error"}`))
				}
			}()
			next.ServeHTTP(w, r)
		})
	}
}

// bodyLimitMiddleware caps request body size.
func bodyLimitMiddleware(limit int64) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Body != nil {
				r.Body = http.MaxBytesReader(w, r.Body, limit)
			}
			next.ServeHTTP(w, r)
		})
	}
}

// securityHeadersMiddleware sets conservative response headers.
func securityHeadersMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.Header().Set("X-Frame-Options", "DENY")
		w.Header().Set("Referrer-Policy", "no-referrer")
		next.ServeHTTP(w, r)
	})
}

// ---------------------------------------------------------------------------
// Concurrency-safe WebSocket writing
// ---------------------------------------------------------------------------

// wsWriter serialises writes to a WebSocket connection.
//
// gorilla/websocket permits only one concurrent writer; the streaming handlers
// write from a goroutine while the read loop continues, so without this the
// connection can interleave frames and corrupt the stream.
type wsWriter struct {
	mu   sync.Mutex
	conn wsConn
}

// wsConn is the subset of *websocket.Conn used by the writer, which keeps this
// testable without a live connection.
type wsConn interface {
	WriteJSON(v interface{}) error
	Close() error
}

func newWSWriter(conn wsConn) *wsWriter { return &wsWriter{conn: conn} }

// WriteJSON writes a message, serialised against other writers.
func (w *wsWriter) WriteJSON(v interface{}) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.conn.WriteJSON(v)
}

// Close closes the underlying connection.
func (w *wsWriter) Close() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.conn.Close()
}
