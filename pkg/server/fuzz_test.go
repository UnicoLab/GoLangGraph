// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

package server

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
)

// FuzzAPIRequestBodies feeds arbitrary payloads to the write endpoints. Request
// bodies come from the network, so a malformed one must produce a 4xx, never a
// panic and never a 500 from an unhandled decode.
func FuzzAPIRequestBodies(f *testing.F) {
	seeds := []string{
		`{"input":"hello"}`,
		`{"state":{"a":1}}`,
		`{"input":"x","thread_id":"t"}`,
		`{}`, `[]`, `null`, `true`, ``, `{`,
		`{"input":12345}`,
		`{"state":"not-an-object"}`,
		`{"state":{"deep":{"deep":{"deep":[1,2,3]}}}}`,
		strings.Repeat(`{"a":`, 500),
		"\x00\x01\x02",
	}
	for _, s := range seeds {
		f.Add(s)
	}

	paths := []string{
		"/api/v1/graphs/fuzz/execute",
		"/api/v1/graphs/missing/execute",
		"/api/v1/agents",
		"/api/v1/sessions",
		"/api/v1/threads",
	}

	f.Fuzz(func(t *testing.T, body string) {
		s := fuzzServer(t)

		for _, path := range paths {
			func() {
				defer func() {
					if r := recover(); r != nil {
						t.Fatalf("handler for %s panicked on %q: %v", path, body, r)
					}
				}()

				req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
				req.Header.Set("Content-Type", "application/json")
				rec := httptest.NewRecorder()
				s.router.ServeHTTP(rec, req)

				if rec.Code == http.StatusInternalServerError {
					t.Fatalf("%s returned 500 for body %q: %s", path, body, rec.Body.String())
				}
			}()
		}
	})
}

// FuzzAPIPaths checks routing against arbitrary path segments: an identifier
// from a URL must never panic a handler or escape into a server error.
func FuzzAPIPaths(f *testing.F) {
	seeds := []string{
		"fuzz", "missing", "", "..", "../../etc/passwd", "%2e%2e",
		strings.Repeat("a", 4096), "with space", "unicode-é",
	}
	for _, s := range seeds {
		f.Add(s)
	}

	f.Fuzz(func(t *testing.T, id string) {
		s := fuzzServer(t)

		for _, tmpl := range []string{
			"/api/v1/graphs/%s",
			"/api/v1/graphs/%s/topology",
			"/api/v1/agents/%s",
			"/api/v1/tools/%s",
		} {
			func() {
				defer func() {
					if r := recover(); r != nil {
						t.Fatalf("path %q panicked with id %q: %v", tmpl, id, r)
					}
				}()

				req := httptest.NewRequest(http.MethodGet, safePath(tmpl, id), nil)
				rec := httptest.NewRecorder()
				s.router.ServeHTTP(rec, req)

				if rec.Code == http.StatusInternalServerError {
					t.Fatalf("%q with id %q returned 500: %s", tmpl, id, rec.Body.String())
				}
			}()
		}
	})
}

// safePath substitutes an identifier into a path template, percent-encoding it
// the way a real HTTP client would. Without this, httptest.NewRequest panics
// while *building* the request, which would report a harness failure rather
// than a server one.
func safePath(tmpl, id string) string {
	return strings.Replace(tmpl, "%s", url.PathEscape(id), 1)
}

// fuzzServer builds a server with one registered graph, without the static
// catch-all so route resolution is exercised directly.
func fuzzServer(t *testing.T) *Server {
	t.Helper()
	s := newTestServer(t, nil)
	if g, ok := s.GraphManager().Get("demo"); ok {
		s.GraphManager().Register("fuzz", g)
	}
	return s
}
