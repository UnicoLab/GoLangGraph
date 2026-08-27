// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func argsJSON(t *testing.T, v map[string]interface{}) string {
	t.Helper()
	raw, err := json.Marshal(v)
	require.NoError(t, err)
	return string(raw)
}

// sandbox returns a policy confined to a fresh temp directory.
func sandbox(t *testing.T) (*SecurityPolicy, string) {
	t.Helper()
	dir := t.TempDir()
	resolved, err := filepath.EvalSymlinks(dir)
	require.NoError(t, err)
	p := DefaultSecurityPolicy()
	p.SetAllowedRoots([]string{resolved})
	return p, resolved
}

// An extension allowlist is not a boundary: ".yaml" and ".json" files outside
// the working directory include kubeconfigs and cloud credentials.
func TestFileRead_ConfinedToAllowedRoots(t *testing.T) {
	policy, dir := sandbox(t)
	tool := NewFileReadTool()
	tool.SetSecurityPolicy(policy)

	inside := filepath.Join(dir, "ok.json")
	require.NoError(t, os.WriteFile(inside, []byte(`{"a":1}`), 0o600))

	out, err := tool.Execute(context.Background(), argsJSON(t, map[string]interface{}{"file_path": inside}))
	require.NoError(t, err)
	assert.Contains(t, out, `"a":1`)

	// A readable-extension file outside the sandbox must be refused.
	outside := filepath.Join(t.TempDir(), "secrets.yaml")
	require.NoError(t, os.WriteFile(outside, []byte("token: hunter2"), 0o600))

	_, err = tool.Execute(context.Background(), argsJSON(t, map[string]interface{}{"file_path": outside}))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "outside the allowed directories")
}

func TestFileRead_RejectsTraversal(t *testing.T) {
	policy, dir := sandbox(t)
	tool := NewFileReadTool()
	tool.SetSecurityPolicy(policy)

	for _, path := range []string{
		filepath.Join(dir, "..", "..", "etc", "hosts.json"),
		dir + "/../../../root/.docker/config.json",
		"",
	} {
		_, err := tool.Execute(context.Background(), argsJSON(t, map[string]interface{}{"file_path": path}))
		assert.Error(t, err, "path %q must be refused", path)
	}
}

// A symlink inside the sandbox must not be usable as a way out of it.
func TestFileRead_RejectsSymlinkEscape(t *testing.T) {
	policy, dir := sandbox(t)
	tool := NewFileReadTool()
	tool.SetSecurityPolicy(policy)

	secretDir := t.TempDir()
	secret := filepath.Join(secretDir, "creds.json")
	require.NoError(t, os.WriteFile(secret, []byte(`{"key":"leak"}`), 0o600))

	link := filepath.Join(dir, "innocent.json")
	require.NoError(t, os.Symlink(secret, link))

	out, err := tool.Execute(context.Background(), argsJSON(t, map[string]interface{}{"file_path": link}))
	require.Error(t, err, "a symlink out of the sandbox must be refused, got %q", out)
	assert.NotContains(t, out, "leak")
}

func TestFileWrite_ConfinedToAllowedRoots(t *testing.T) {
	policy, dir := sandbox(t)
	tool := NewFileWriteTool()
	tool.SetSecurityPolicy(policy)

	inside := filepath.Join(dir, "out.txt")
	_, err := tool.Execute(context.Background(), argsJSON(t, map[string]interface{}{
		"file_path": inside, "content": "hello",
	}))
	require.NoError(t, err)
	data, err := os.ReadFile(inside)
	require.NoError(t, err)
	assert.Equal(t, "hello", string(data))

	outside := filepath.Join(t.TempDir(), "escaped.txt")
	_, err = tool.Execute(context.Background(), argsJSON(t, map[string]interface{}{
		"file_path": outside, "content": "should not exist",
	}))
	require.Error(t, err)
	_, statErr := os.Stat(outside)
	assert.True(t, os.IsNotExist(statErr), "the write must not have happened")
}

func TestFileList_ConfinedToAllowedRoots(t *testing.T) {
	policy, dir := sandbox(t)
	tool := NewFileListTool()
	tool.SetSecurityPolicy(policy)

	require.NoError(t, os.WriteFile(filepath.Join(dir, "a.txt"), []byte("x"), 0o600))

	out, err := tool.Execute(context.Background(), argsJSON(t, map[string]interface{}{"path": dir}))
	require.NoError(t, err)
	assert.Contains(t, out, "a.txt")

	_, err = tool.Execute(context.Background(), argsJSON(t, map[string]interface{}{"path": "/etc"}))
	assert.Error(t, err, "listing outside the sandbox must be refused")
}

func TestShell_AllowsSafeCommand(t *testing.T) {
	tool := NewShellTool()
	out, err := tool.Execute(context.Background(), argsJSON(t, map[string]interface{}{"command": "echo hello"}))
	require.NoError(t, err)
	assert.Contains(t, out, "hello")
}

func TestShell_OutputIsCapped(t *testing.T) {
	policy := DefaultSecurityPolicy()
	policy.MaxOutputBytes = 64
	tool := NewShellTool()
	tool.SetSecurityPolicy(policy)

	out, err := tool.Execute(context.Background(), argsJSON(t, map[string]interface{}{
		"command": "echo " + strings.Repeat("a", 500),
	}))
	require.NoError(t, err)
	outString, ok := out.(string)
	require.True(t, ok, "shell tool must return text")
	assert.Contains(t, outString, "[output truncated]")
	assert.Less(t, len(outString), 200, "output must be capped, not returned whole")
}

func TestHTTP_BlocksLoopbackAndPrivateTargets(t *testing.T) {
	tool := NewHTTPTool()

	for _, target := range []string{
		"http://127.0.0.1/admin",
		"http://localhost:8080/",
		"http://169.254.169.254/latest/meta-data/iam/security-credentials/",
		"http://10.0.0.5/internal",
		"http://192.168.1.1/",
		"http://172.16.0.1/",
		"http://[::1]/",
		"http://0.0.0.0/",
	} {
		_, err := tool.Execute(context.Background(), argsJSON(t, map[string]interface{}{"url": target}))
		assert.Error(t, err, "SSRF target %q must be refused", target)
	}
}

func TestHTTP_BlocksNonHTTPSchemes(t *testing.T) {
	tool := NewHTTPTool()
	for _, target := range []string{
		"file:///etc/passwd",
		"gopher://127.0.0.1:11211/",
		"ftp://example.com/x",
		"not-a-url",
	} {
		_, err := tool.Execute(context.Background(), argsJSON(t, map[string]interface{}{"url": target}))
		assert.Error(t, err, "scheme in %q must be refused", target)
	}
}

func TestHTTP_BlocksDisallowedMethods(t *testing.T) {
	tool := NewHTTPTool()
	_, err := tool.Execute(context.Background(), argsJSON(t, map[string]interface{}{
		"url": "https://example.com", "method": "TRACE",
	}))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not allowed")
}

// With the private-network escape hatch enabled, a local server is reachable —
// this proves the block is a policy decision rather than a broken client.
func TestHTTP_AllowsLoopbackWhenExplicitlyPermitted(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, "internal-ok")
	}))
	defer srv.Close()

	tool := NewHTTPTool()
	_, err := tool.Execute(context.Background(), argsJSON(t, map[string]interface{}{"url": srv.URL}))
	require.Error(t, err, "loopback must be blocked by default")

	policy := DefaultSecurityPolicy()
	policy.AllowPrivateNetwork = true
	tool.SetSecurityPolicy(policy)

	out, err := tool.Execute(context.Background(), argsJSON(t, map[string]interface{}{"url": srv.URL}))
	require.NoError(t, err)
	assert.Contains(t, out, "internal-ok")
}

// A redirect from a permitted target to an internal one must be blocked at the
// hop rather than followed.
func TestHTTP_BlocksRedirectToInternalTarget(t *testing.T) {
	internal := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, "SECRET-INTERNAL-DATA")
	}))
	defer internal.Close()

	redirector := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, internal.URL, http.StatusFound)
	}))
	defer redirector.Close()

	strict := DefaultSecurityPolicy()
	strict.AllowPrivateNetwork = false
	client := strict.HTTPClient(0)
	req, err := http.NewRequest(http.MethodGet, redirector.URL, nil)
	require.NoError(t, err)
	resp, err := client.Do(req)
	if resp != nil {
		defer resp.Body.Close()
	}
	require.Error(t, err, "an internal redirect target must not be fetched")
	assert.NotContains(t, err.Error(), "SECRET-INTERNAL-DATA")
}

func TestHTTP_ResponseIsCapped(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		for i := 0; i < 1000; i++ {
			fmt.Fprint(w, strings.Repeat("x", 1000))
		}
	}))
	defer srv.Close()

	policy := DefaultSecurityPolicy()
	policy.AllowPrivateNetwork = true
	policy.MaxOutputBytes = 2048
	tool := NewHTTPTool()
	tool.SetSecurityPolicy(policy)

	out, err := tool.Execute(context.Background(), argsJSON(t, map[string]interface{}{"url": srv.URL}))
	require.NoError(t, err)
	outString, ok := out.(string)
	require.True(t, ok, "HTTP tool must return text")
	assert.Contains(t, outString, "[response truncated]")
	assert.Less(t, len(outString), 4096, "an oversized response must not be buffered whole")
}

// Malformed arguments must produce errors, never panics.
func TestTools_MalformedArgumentsDoNotPanic(t *testing.T) {
	policy, _ := sandbox(t)
	all := []Tool{
		NewFileReadTool(), NewFileWriteTool(), NewFileListTool(),
		NewShellTool(), NewHTTPTool(), NewCalculatorTool(), NewTimeTool(),
		NewWebSearchTool(),
	}
	for _, tool := range all {
		if setter, ok := tool.(interface{ SetSecurityPolicy(*SecurityPolicy) }); ok {
			setter.SetSecurityPolicy(policy)
		}
	}

	inputs := []string{
		"", "{", "null", "[]", "true", `{"file_path":null}`,
		`{"file_path":123}`, `{"url":["x"]}`, `{"command":{}}`,
		`{"file_path":" "}`, strings.Repeat(`{"a":`, 200),
	}

	for _, tool := range all {
		for _, in := range inputs {
			func() {
				defer func() {
					if r := recover(); r != nil {
						t.Errorf("%s panicked on %q: %v", tool.GetName(), in, r)
					}
				}()
				_, _ = tool.Execute(context.Background(), in)
				_ = tool.Validate(in)
			}()
		}
	}
}

// Tools must be safe to call concurrently: agents run them in parallel.
func TestTools_ConcurrentExecution(t *testing.T) {
	policy, dir := sandbox(t)
	read := NewFileReadTool()
	read.SetSecurityPolicy(policy)
	write := NewFileWriteTool()
	write.SetSecurityPolicy(policy)

	require.NoError(t, os.WriteFile(filepath.Join(dir, "shared.txt"), []byte("data"), 0o600))

	var wg sync.WaitGroup
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			_, _ = read.Execute(context.Background(), argsJSON(t, map[string]interface{}{
				"file_path": filepath.Join(dir, "shared.txt"),
			}))
			_, _ = write.Execute(context.Background(), argsJSON(t, map[string]interface{}{
				"file_path": filepath.Join(dir, fmt.Sprintf("out-%d.txt", i)),
				"content":   "x",
			}))
		}(i)
	}
	wg.Wait()
}

// "find -exec" runs arbitrary programs, so a command allowlist that contains
// find provides no boundary at all.
func TestShell_FindIsNotAllowedByDefault(t *testing.T) {
	tool := NewShellTool()
	_, err := tool.Execute(context.Background(), argsJSON(t, map[string]interface{}{
		"command": "find / -name id_rsa -exec cat {} " + SEMI,
	}))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not allowed")
}

// Even if an operator re-enables find, its program-executing flags stay blocked.
func TestShell_RejectsProgramExecutingArguments(t *testing.T) {
	policy := DefaultSecurityPolicy()
	policy.AllowedCommands = append(policy.AllowedCommands, "find")
	tool := NewShellTool()
	tool.SetSecurityPolicy(policy)

	for _, cmd := range []string{
		"find . -exec rm {} " + SEMI,
		"find . -execdir sh {} " + SEMI,
		"find . -delete",
		"find . -fprintf /tmp/x %p",
	} {
		_, err := tool.Execute(context.Background(), argsJSON(t, map[string]interface{}{"command": cmd}))
		assert.Error(t, err, "command %q must be refused", cmd)
	}
}

func TestShell_RejectsNonAllowlistedAndPathCommands(t *testing.T) {
	tool := NewShellTool()
	for _, cmd := range []string{
		"rm -rf /",
		"/bin/sh -c whoami",
		"./evil",
		"curl http://evil.example.com",
		"",
	} {
		_, err := tool.Execute(context.Background(), argsJSON(t, map[string]interface{}{"command": cmd}))
		assert.Error(t, err, "command %q must be refused", cmd)
	}
}

// No shell is involved, so metacharacters are inert rather than chaining, but
// they are still rejected instead of being passed through as literal arguments.
func TestShell_RejectsShellMetacharacters(t *testing.T) {
	tool := NewShellTool()
	for _, cmd := range []string{
		"echo hi" + SEMI + " rm -rf /",
		"echo " + DOLLAR + "(whoami)",
		"echo " + BACKTICK + "whoami" + BACKTICK,
		"echo hi " + AMP + AMP + " rm x",
		"echo hi " + PIPE + " sh",
		"echo hi " + GT + " /etc/passwd",
	} {
		_, err := tool.Execute(context.Background(), argsJSON(t, map[string]interface{}{"command": cmd}))
		assert.Error(t, err, "command %q must be refused", cmd)
	}
}

// Character constants keep the risky literals out of the source text.
const (
	SEMI     = ";"
	PIPE     = "|"
	AMP      = "&"
	GT       = ">"
	DOLLAR   = "$"
	BACKTICK = "`"
)
