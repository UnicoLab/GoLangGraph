// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package server

import (
	"context"
	"crypto/subtle"
	"fmt"
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
	// Deprecated for production use: legacy keys are granted the admin role.
	// Use APIKeyCredentials to issue named, least-privilege keys instead.
	APIKeys []string `json:"api_keys" yaml:"api_keys"`
	// APIKeyCredentials contains named keys and their least-privilege role.
	// Store Key values in a secret manager or mounted secret, never in source.
	APIKeyCredentials []APIKeyCredential `json:"api_key_credentials" yaml:"api_key_credentials"`
	// AllowedOrigins restricts CORS and WebSocket origins. Empty means any
	// origin is accepted, which is only appropriate for local development.
	AllowedOrigins []string `json:"allowed_origins" yaml:"allowed_origins"`
	// MaxRequestBytes caps request bodies. Zero applies DefaultMaxRequestBytes.
	MaxRequestBytes int64 `json:"max_request_bytes" yaml:"max_request_bytes"`
	// PublicPaths bypass authentication (health checks, readiness probes).
	PublicPaths []string `json:"public_paths" yaml:"public_paths"`
}

// APIKeyRole is an ordered permission level for the public control plane.
// Viewer can inspect; executor can invoke and interrupt runs; author can also
// create or modify agents and pipelines; admin is reserved for legacy keys and
// future administrative APIs. // pragma: allowlist secret
type APIKeyRole string

const (
	RoleViewer   APIKeyRole = "viewer"   // pragma: allowlist secret
	RoleExecutor APIKeyRole = "executor" // pragma: allowlist secret
	RoleAuthor   APIKeyRole = "author"   // pragma: allowlist secret
	RoleAdmin    APIKeyRole = "admin"    // pragma: allowlist secret
)

// APIKeyCredential is deliberately named so audit records never need to log
// the secret value in order to identify who changed a deployment. // pragma: allowlist secret
type APIKeyCredential struct {
	Name string     `json:"name" yaml:"name"`
	Key  string     `json:"key" yaml:"key"` // pragma: allowlist secret
	Role APIKeyRole `json:"role" yaml:"role"`
}

// Principal is the authenticated identity carried through a request context
// and returned by /api/v1/whoami. It never includes a credential.
type Principal struct {
	Name string     `json:"name"`
	Role APIKeyRole `json:"role"`
}

type principalContextKey struct{}

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
	_, ok := c.authenticate(presented)
	return ok
}

// authenticate checks all configured candidates using constant-time compares
// and returns an audit-safe principal. Legacy APIKeys remain compatible as
// admins, while new credentials can be scoped down to a single capability.
func (c *SecurityConfig) authenticate(presented string) (Principal, bool) {
	if c == nil || !c.RequireAuth {
		return Principal{Name: "local-development", Role: RoleAdmin}, true
	}
	if presented == "" {
		return Principal{}, false
	}
	matched := false
	principal := Principal{}
	for index, key := range c.APIKeys {
		if key != "" && subtle.ConstantTimeCompare([]byte(key), []byte(presented)) == 1 {
			matched = true
			principal = Principal{Name: fmt.Sprintf("legacy-key-%d", index+1), Role: RoleAdmin}
		}
	}
	for index, credential := range c.APIKeyCredentials {
		if credential.Key == "" {
			continue
		}
		if subtle.ConstantTimeCompare([]byte(credential.Key), []byte(presented)) == 1 {
			matched = true
			role := normalizeRole(credential.Role)
			name := strings.TrimSpace(credential.Name)
			if name == "" {
				name = fmt.Sprintf("key-%d", index+1)
			}
			// If an accidental duplicate key appears, retain the most privileged
			// matching role so behavior is deterministic; deployments should
			// never reuse secrets across principals.
			if principal.Name == "" || roleRank(role) > roleRank(principal.Role) {
				principal = Principal{Name: name, Role: role}
			}
		}
	}
	return principal, matched
}

func normalizeRole(role APIKeyRole) APIKeyRole {
	switch role {
	case RoleViewer, RoleExecutor, RoleAuthor, RoleAdmin:
		return role
	default:
		return RoleViewer
	}
}

func roleRank(role APIKeyRole) int {
	switch normalizeRole(role) {
	case RoleAdmin:
		return 4
	case RoleAuthor:
		return 3
	case RoleExecutor:
		return 2
	default:
		return 1
	}
}

func hasRole(principal Principal, required APIKeyRole) bool {
	return roleRank(principal.Role) >= roleRank(required)
}

func withPrincipal(ctx context.Context, principal Principal) context.Context {
	return context.WithValue(ctx, principalContextKey{}, principal)
}

func principalFromContext(ctx context.Context) (Principal, bool) {
	principal, ok := ctx.Value(principalContextKey{}).(Principal)
	return principal, ok
}

// requiredRole maps the public API to least privilege. Read-only catalog and
// observability requests need viewer; invokes and stop controls need executor;
// deployment mutations need author. New endpoints should choose explicitly
// rather than relying on a generic HTTP-method rule.
func requiredRole(r *http.Request) APIKeyRole {
	if r.Method == http.MethodGet || r.Method == http.MethodHead || r.Method == http.MethodOptions {
		return RoleViewer
	}
	path := r.URL.Path
	if strings.Contains(path, "/execute") || strings.Contains(path, "/interrupt") ||
		strings.Contains(path, "/ws/") || strings.Contains(path, "/playground/") ||
		strings.HasPrefix(path, "/api/v1/sessions") || strings.HasPrefix(path, "/api/v1/threads") {
		return RoleExecutor
	}
	return RoleAuthor
}

// originAllowed reports whether an Origin header may access the API.
//
// An empty AllowedOrigins list accepts any origin, matching the previous
// permissive behavior for local development. A request with no Origin header
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

// wsWriter serializes writes to a WebSocket connection.
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

// WriteJSON writes a message, serialized against other writers.
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
