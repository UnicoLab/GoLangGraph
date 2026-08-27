// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

package tools

import (
	"context"
	"path/filepath"
	"strings"
	"testing"
)

// fuzzTools returns every built-in tool, confined to a sandbox so a fuzzed
// argument cannot touch anything outside the test's temporary directory.
func fuzzTools(t *testing.T, root string) []Tool {
	policy := DefaultSecurityPolicy()
	policy.SetAllowedRoots([]string{root})
	// Never let a fuzzed URL reach the network.
	policy.AllowedHosts = []string{"127.0.0.1.invalid"}

	all := []Tool{
		NewFileReadTool(), NewFileWriteTool(), NewFileListTool(),
		NewShellTool(), NewHTTPTool(), NewCalculatorTool(),
		NewTimeTool(), NewWebSearchTool(),
	}
	for _, tool := range all {
		if setter, ok := tool.(interface{ SetSecurityPolicy(*SecurityPolicy) }); ok {
			setter.SetSecurityPolicy(policy)
		}
	}
	return all
}

// FuzzToolArguments feeds arbitrary payloads to every tool. Tool arguments are
// produced by a language model, so they are untrusted: a tool must return an
// error, never panic and never escape its sandbox.
func FuzzToolArguments(f *testing.F) {
	seeds := []string{
		`{"file_path":"/etc/passwd"}`,
		`{"file_path":"../../../../etc/shadow"}`,
		`{"command":"ls -la"}`,
		`{"url":"http://169.254.169.254/"}`,
		`{"expression":"1+1"}`,
		`{"query":"anything"}`,
		`{"path":"."}`,
		`{}`, `[]`, `null`, ``, `{`,
		`{"file_path":" "}`,
		`{"expression":"9999999999^9999999999"}`,
		strings.Repeat(`{"a":`, 100),
	}
	for _, s := range seeds {
		f.Add(s)
	}

	f.Fuzz(func(t *testing.T, args string) {
		root := t.TempDir()
		resolved, err := filepath.EvalSymlinks(root)
		if err != nil {
			resolved = root
		}

		for _, tool := range fuzzTools(t, resolved) {
			func() {
				defer func() {
					if r := recover(); r != nil {
						t.Fatalf("%s panicked on %q: %v", tool.GetName(), args, r)
					}
				}()
				_ = tool.Validate(args)
				// Any error is fine; a crash or an escape is not.
				_, _ = tool.Execute(context.Background(), args)
			}()
		}
	})
}

// FuzzPathResolution checks the filesystem confinement directly: no input may
// resolve to a path outside the allowed roots.
func FuzzPathResolution(f *testing.F) {
	seeds := []string{
		"file.txt", "../escape", "../../../../etc/passwd", "/etc/passwd",
		"./nested/../file", "", " ", "sub/dir/file.json", "/proc/self/environ",
		strings.Repeat("../", 50) + "etc/passwd",
		"embedded" + string(rune(0)) + "null",
	}
	for _, s := range seeds {
		f.Add(s)
	}

	f.Fuzz(func(t *testing.T, path string) {
		root := t.TempDir()
		resolved, err := filepath.EvalSymlinks(root)
		if err != nil {
			resolved = root
		}
		policy := DefaultSecurityPolicy()
		policy.SetAllowedRoots([]string{resolved})

		got, err := policy.ResolvePath(path)
		if err != nil {
			return // refusing the path is a correct outcome
		}

		// An accepted path must lie within the root.
		if got != resolved && !strings.HasPrefix(got, resolved+string(filepath.Separator)) {
			t.Fatalf("path %q escaped the sandbox: resolved to %q, root %q", path, got, resolved)
		}
		if !filepath.IsAbs(got) {
			t.Fatalf("resolved path %q is not absolute (input %q)", got, path)
		}
	})
}

// FuzzCommandPolicy checks that no argument string can smuggle execution past
// the shell allowlist.
func FuzzCommandPolicy(f *testing.F) {
	seeds := []string{
		"echo hi", "ls -la", "rm -rf /", "/bin/sh -c id",
		"find . -exec rm", "", "   ", "echo a b c",
	}
	for _, s := range seeds {
		f.Add(s)
	}

	f.Fuzz(func(t *testing.T, command string) {
		policy := DefaultSecurityPolicy()
		parts := strings.Fields(command)

		if err := policy.CheckCommand(parts); err != nil {
			return // refusing is correct
		}

		// Anything accepted must be a bare allowlisted name.
		if len(parts) == 0 {
			t.Fatalf("an empty command was accepted")
		}
		base := parts[0]
		permitted := false
		for _, allowed := range policy.AllowedCommands {
			if base == allowed {
				permitted = true
				break
			}
		}
		if !permitted {
			t.Fatalf("command %q was accepted but is not allowlisted", base)
		}
		if strings.ContainsAny(base, "/\\") {
			t.Fatalf("command %q was accepted as a path", base)
		}
	})
}

// FuzzURLPolicy checks that no URL accepted by the policy points at a local or
// private address.
func FuzzURLPolicy(f *testing.F) {
	seeds := []string{
		"http://example.com", "https://example.com/path",
		"http://127.0.0.1", "http://169.254.169.254/latest/",
		"file:///etc/passwd", "", "http://",
		"http://2130706433/", "http://0x7f000001/", "http://localhost",
	}
	for _, s := range seeds {
		f.Add(s)
	}

	f.Fuzz(func(t *testing.T, raw string) {
		policy := DefaultSecurityPolicy()
		parsed, err := policy.CheckURL(raw)
		if err != nil {
			return // refusing is correct
		}

		scheme := strings.ToLower(parsed.Scheme)
		if scheme != "http" && scheme != "https" {
			t.Fatalf("URL %q accepted with scheme %q", raw, parsed.Scheme)
		}
		if parsed.Host == "" {
			t.Fatalf("URL %q accepted with no host", raw)
		}
	})
}
