// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package tools

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"
)

// SecurityPolicy bounds what the built-in tools may touch.
//
// Tool arguments are chosen by a language model, which may be steered by
// untrusted input, so every tool that reaches the filesystem, a shell or the
// network is treated as attacker-controlled and constrained here.
type SecurityPolicy struct {
	mu sync.RWMutex

	// AllowedRoots are absolute directories the file tools may operate within.
	// Empty means the process working directory and the system temp directory.
	AllowedRoots []string

	// MaxOutputBytes caps command output and HTTP response bodies.
	MaxOutputBytes int64

	// AllowedCommands is the shell command allowlist.
	AllowedCommands []string

	// DeniedArgSubstrings reject shell arguments that can execute other
	// programs even when the base command is allowed.
	DeniedArgSubstrings []string

	// AllowPrivateNetwork permits requests to loopback, link-local and private
	// address ranges. Off by default: those ranges hold cloud metadata services
	// and internal admin endpoints.
	AllowPrivateNetwork bool

	// AllowedHosts, when non-empty, restricts HTTP requests to these hosts.
	AllowedHosts []string

	// MaxRedirects bounds HTTP redirect following.
	MaxRedirects int
}

// DefaultMaxToolOutputBytes caps how much a single tool call may return.
const DefaultMaxToolOutputBytes int64 = 1 << 20 // 1 MiB

// DefaultSecurityPolicy returns the policy applied to tools built with the
// New*Tool constructors.
func DefaultSecurityPolicy() *SecurityPolicy {
	return &SecurityPolicy{
		MaxOutputBytes: DefaultMaxToolOutputBytes,
		// "find" is deliberately absent: -exec, -execdir and -ok run arbitrary
		// programs, which defeats any command allowlist.
		AllowedCommands: []string{"ls", "pwd", "echo", "wc", "head", "tail"},
		DeniedArgSubstrings: []string{
			"-exec", "-execdir", "-ok", "-okdir", "-fprintf", "-delete",
			"--eval", "-e", "$(", "`", ";", "|", "&&", "||", ">", "<",
		},
		MaxRedirects: 5,
	}
}

// roots returns the effective allowed roots, resolved to absolute paths.
func (p *SecurityPolicy) roots() []string {
	p.mu.RLock()
	configured := append([]string(nil), p.AllowedRoots...)
	p.mu.RUnlock()

	if len(configured) == 0 {
		cwd, err := os.Getwd()
		if err == nil {
			configured = append(configured, cwd)
		}
		configured = append(configured, os.TempDir())
	}

	out := make([]string, 0, len(configured))
	for _, root := range configured {
		abs, err := filepath.Abs(root)
		if err != nil {
			continue
		}
		if resolved, err := filepath.EvalSymlinks(abs); err == nil {
			abs = resolved
		}
		out = append(out, abs)
	}
	return out
}

// SetAllowedRoots replaces the directories file tools may operate within.
func (p *SecurityPolicy) SetAllowedRoots(roots []string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.AllowedRoots = append([]string(nil), roots...)
}

// maxOutput returns the effective output cap.
func (p *SecurityPolicy) maxOutput() int64 {
	if p == nil || p.MaxOutputBytes <= 0 {
		return DefaultMaxToolOutputBytes
	}
	return p.MaxOutputBytes
}

// ResolvePath validates a caller-supplied path and returns its cleaned absolute
// form. Symlinks are resolved so a link inside an allowed root cannot be used
// to reach a file outside one; for a path that does not exist yet, the nearest
// existing parent is checked instead so new files can still be created.
func (p *SecurityPolicy) ResolvePath(path string) (string, error) {
	if strings.TrimSpace(path) == "" {
		return "", fmt.Errorf("path must not be empty")
	}
	if strings.ContainsRune(path, 0) {
		return "", fmt.Errorf("path must not contain a null byte")
	}

	abs, err := filepath.Abs(path)
	if err != nil {
		return "", fmt.Errorf("invalid path: %w", err)
	}
	abs = filepath.Clean(abs)

	// Resolve symlinks on the deepest existing ancestor.
	probe := abs
	var trailing []string
	for {
		if resolved, err := filepath.EvalSymlinks(probe); err == nil {
			probe = resolved
			break
		}
		parent := filepath.Dir(probe)
		if parent == probe {
			break
		}
		trailing = append([]string{filepath.Base(probe)}, trailing...)
		probe = parent
	}
	resolved := filepath.Join(append([]string{probe}, trailing...)...)

	roots := p.roots()
	if len(roots) == 0 {
		return "", fmt.Errorf("no allowed roots are configured")
	}
	for _, root := range roots {
		if resolved == root || strings.HasPrefix(resolved, root+string(os.PathSeparator)) {
			return resolved, nil
		}
	}

	return "", fmt.Errorf("path %q is outside the allowed directories", path)
}

// CheckCommand validates a shell invocation against the allowlist and rejects
// arguments that would let an allowed command run something else.
func (p *SecurityPolicy) CheckCommand(parts []string) error {
	if len(parts) == 0 {
		return fmt.Errorf("empty command")
	}

	p.mu.RLock()
	allowed := append([]string(nil), p.AllowedCommands...)
	denied := append([]string(nil), p.DeniedArgSubstrings...)
	p.mu.RUnlock()

	base := parts[0]
	if strings.ContainsAny(base, "/\\") {
		return fmt.Errorf("command %q must be a bare name, not a path", base)
	}

	permitted := false
	for _, cmd := range allowed {
		if base == cmd {
			permitted = true
			break
		}
	}
	if !permitted {
		return fmt.Errorf("command %s not allowed", base)
	}

	for _, arg := range parts[1:] {
		for _, bad := range denied {
			if strings.Contains(arg, bad) {
				return fmt.Errorf("argument %q is not allowed (contains %q)", arg, bad)
			}
		}
	}

	return nil
}

// ---------------------------------------------------------------------------
// Network policy
// ---------------------------------------------------------------------------

// CheckURL validates a request target before any connection is made.
func (p *SecurityPolicy) CheckURL(raw string) (*url.URL, error) {
	parsed, err := url.Parse(raw)
	if err != nil {
		return nil, fmt.Errorf("invalid URL: %w", err)
	}
	switch strings.ToLower(parsed.Scheme) {
	case "http", "https":
	default:
		return nil, fmt.Errorf("URL scheme %q is not allowed", parsed.Scheme)
	}
	if parsed.Host == "" {
		return nil, fmt.Errorf("URL must include a host")
	}

	p.mu.RLock()
	allowedHosts := append([]string(nil), p.AllowedHosts...)
	allowPrivate := p.AllowPrivateNetwork
	p.mu.RUnlock()

	host := parsed.Hostname()

	if len(allowedHosts) > 0 {
		ok := false
		for _, h := range allowedHosts {
			if strings.EqualFold(h, host) {
				ok = true
				break
			}
		}
		if !ok {
			return nil, fmt.Errorf("host %q is not in the allowed list", host)
		}
	}

	// A literal IP can be checked now; hostnames are checked at dial time,
	// which also defeats DNS rebinding.
	if ip := net.ParseIP(host); ip != nil && !allowPrivate {
		if err := checkPublicIP(ip); err != nil {
			return nil, err
		}
	}

	return parsed, nil
}

// checkPublicIP rejects addresses that reach the host itself or the local
// network, including cloud metadata endpoints.
func checkPublicIP(ip net.IP) error {
	switch {
	case ip.IsLoopback():
		return fmt.Errorf("address %s is loopback", ip)
	case ip.IsPrivate():
		return fmt.Errorf("address %s is in a private range", ip)
	case ip.IsLinkLocalUnicast(), ip.IsLinkLocalMulticast():
		// 169.254.169.254 is the cloud instance metadata service.
		return fmt.Errorf("address %s is link-local", ip)
	case ip.IsUnspecified():
		return fmt.Errorf("address %s is unspecified", ip)
	case ip.IsMulticast():
		return fmt.Errorf("address %s is multicast", ip)
	case ip.IsInterfaceLocalMulticast():
		return fmt.Errorf("address %s is interface-local", ip)
	}
	// Unique local IPv6 (fc00::/7) is not covered by IsPrivate on all versions.
	if len(ip) == net.IPv6len && ip[0]&0xfe == 0xfc {
		return fmt.Errorf("address %s is a unique local address", ip)
	}
	return nil
}

// HTTPClient builds a client that enforces the policy on every connection and
// every redirect hop.
func (p *SecurityPolicy) HTTPClient(timeout time.Duration) *http.Client {
	p.mu.RLock()
	allowPrivate := p.AllowPrivateNetwork
	maxRedirects := p.MaxRedirects
	p.mu.RUnlock()
	if maxRedirects <= 0 {
		maxRedirects = 5
	}

	dialer := &net.Dialer{Timeout: 10 * time.Second, KeepAlive: 30 * time.Second}
	if !allowPrivate {
		// Control runs after DNS resolution with the address about to be used,
		// so a hostname that resolves to an internal address is caught here even
		// if it resolved to a public address a moment earlier.
		dialer.Control = func(network, address string, _ syscall.RawConn) error {
			host, _, err := net.SplitHostPort(address)
			if err != nil {
				return fmt.Errorf("unexpected address %q: %w", address, err)
			}
			ip := net.ParseIP(host)
			if ip == nil {
				return fmt.Errorf("could not parse resolved address %q", host)
			}
			return checkPublicIP(ip)
		}
	}

	return &http.Client{
		Timeout: timeout,
		Transport: &http.Transport{
			DialContext:           dialer.DialContext,
			TLSHandshakeTimeout:   10 * time.Second,
			ResponseHeaderTimeout: timeout,
			MaxIdleConns:          10,
			IdleConnTimeout:       30 * time.Second,
		},
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if len(via) >= maxRedirects {
				return fmt.Errorf("stopped after %d redirects", maxRedirects)
			}
			// Re-validate each hop: a public URL may redirect to an internal one.
			if _, err := p.CheckURL(req.URL.String()); err != nil {
				return fmt.Errorf("redirect blocked: %w", err)
			}
			return nil
		},
	}
}

// LimitedRead reads at most the policy's output cap, reporting truncation.
func (p *SecurityPolicy) LimitedRead(r interface{ Read([]byte) (int, error) }) ([]byte, bool, error) {
	limit := p.maxOutput()
	buf := make([]byte, 0, 4096)
	tmp := make([]byte, 32*1024)
	truncated := false
	for {
		n, err := r.Read(tmp)
		if n > 0 {
			remaining := limit - int64(len(buf))
			if int64(n) > remaining {
				buf = append(buf, tmp[:remaining]...)
				truncated = true
				break
			}
			buf = append(buf, tmp[:n]...)
		}
		if err != nil {
			if err.Error() == "EOF" {
				break
			}
			return buf, truncated, err
		}
	}
	return buf, truncated, nil
}

// truncateOutput caps a byte slice to the policy limit.
func (p *SecurityPolicy) truncateOutput(b []byte) (string, bool) {
	limit := p.maxOutput()
	if int64(len(b)) <= limit {
		return string(b), false
	}
	return string(b[:limit]), true
}

// ensure context import is used by tools that pass one through.
var _ = context.Background
