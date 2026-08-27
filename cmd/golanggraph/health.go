// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package main

import (
	"context"
	"errors"
	"fmt"
	"math"
	"net"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"syscall"
	"time"
)

// healthOptions configures a health check run.
type healthOptions struct {
	// ServerURL, when set, probes a running server's HTTP health endpoint
	// instead of inspecting local dependencies. This is what a container
	// health check should use.
	ServerURL string
	// Strict turns warnings into failures.
	Strict bool
	// Timeout bounds each individual probe.
	Timeout time.Duration
	// MinFreeDiskBytes is the free space below which the check fails.
	MinFreeDiskBytes uint64
}

// checkResult is the outcome of a single probe.
type checkResult struct {
	Name    string
	OK      bool
	Warning bool
	Detail  string
}

func (r checkResult) symbol() string {
	switch {
	case r.OK:
		return "✓"
	case r.Warning:
		return "⚠"
	default:
		return "✗"
	}
}

// runHealthCheck performs a real health check and exits with a status an
// orchestrator can act on.
//
// The previous implementation printed "✓ Reachable" for PostgreSQL, Redis and
// Ollama without ever connecting to them, and reported disk and memory as fine
// unconditionally — so it reported a healthy system no matter what was actually
// running. It also exited non-zero when the optional OPENAI_API_KEY was unset,
// which combined with the container HEALTHCHECK marked every deployment without
// an OpenAI key permanently unhealthy.
func runHealthCheck(opts healthOptions) int {
	if opts.Timeout <= 0 {
		opts.Timeout = 3 * time.Second
	}
	if opts.MinFreeDiskBytes == 0 {
		opts.MinFreeDiskBytes = 64 << 20 // 64 MiB
	}

	fmt.Printf("Running GoLangGraph health check...\n")

	var results []checkResult

	if opts.ServerURL != "" {
		results = append(results, probeServer(opts.ServerURL, opts.Timeout))
	} else {
		results = append(results, probeDependencies(opts)...)
		results = append(results, probeResources(opts)...)
	}

	failed, warned := 0, 0
	for _, r := range results {
		fmt.Printf("  %s %s: %s\n", r.symbol(), r.Name, r.Detail)
		switch {
		case !r.OK && !r.Warning:
			failed++
		case r.Warning:
			warned++
		}
	}

	fmt.Printf("\n")
	switch {
	case failed > 0:
		fmt.Printf("✗ System is unhealthy: %d failed, %d warnings\n", failed, warned)
		return 1
	case warned > 0 && opts.Strict:
		fmt.Printf("✗ System has %d warnings and --strict is set\n", warned)
		return 1
	case warned > 0:
		fmt.Printf("✅ System is healthy (%d warnings)\n", warned)
		return 0
	default:
		fmt.Printf("✅ System is healthy\n")
		return 0
	}
}

// probeServer checks a running server's health endpoint.
func probeServer(rawURL string, timeout time.Duration) checkResult {
	name := "server " + rawURL

	parsed, err := url.Parse(rawURL)
	if err != nil || parsed.Host == "" {
		return checkResult{Name: name, Detail: fmt.Sprintf("invalid server URL: %v", err)}
	}
	endpoint := strings.TrimRight(rawURL, "/") + "/api/v1/health"

	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return checkResult{Name: name, Detail: err.Error()}
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return checkResult{Name: name, Detail: fmt.Sprintf("unreachable: %v", err)}
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return checkResult{Name: name, Detail: fmt.Sprintf("returned status %d", resp.StatusCode)}
	}
	return checkResult{Name: name, OK: true, Detail: "responding"}
}

// probeDependencies checks the services that are actually configured.
//
// A dependency is only probed when its environment variable is set: defaulting
// to localhost would report a failure in every deployment that does not use
// that service.
func probeDependencies(opts healthOptions) []checkResult {
	var results []checkResult

	if host := os.Getenv("POSTGRES_HOST"); host != "" {
		port := envOr("POSTGRES_PORT", "5432")
		results = append(results, probeTCP("PostgreSQL", net.JoinHostPort(host, port), opts.Timeout, false))
	}

	if host := os.Getenv("REDIS_HOST"); host != "" {
		port := envOr("REDIS_PORT", "6379")
		results = append(results, probeTCP("Redis", net.JoinHostPort(host, port), opts.Timeout, false))
	}

	if raw := os.Getenv("OLLAMA_URL"); raw != "" {
		if parsed, err := url.Parse(raw); err == nil && parsed.Host != "" {
			host := parsed.Hostname()
			port := parsed.Port()
			if port == "" {
				port = map[bool]string{true: "443", false: "80"}[parsed.Scheme == "https"]
			}
			results = append(results, probeTCP("Ollama", net.JoinHostPort(host, port), opts.Timeout, false))
		} else {
			results = append(results, checkResult{Name: "Ollama", Detail: "invalid OLLAMA_URL"})
		}
	}

	// Credentials are informational: a deployment may legitimately use only one
	// provider, so a missing key is never a failure on its own.
	for _, cred := range []struct{ name, env string }{
		{"OpenAI credentials", "OPENAI_API_KEY"},
		{"Gemini credentials", "GEMINI_API_KEY"},
	} {
		if os.Getenv(cred.env) != "" {
			results = append(results, checkResult{Name: cred.name, OK: true, Detail: "configured"})
		} else {
			results = append(results, checkResult{Name: cred.name, Warning: true, Detail: cred.env + " is not set"})
		}
	}

	if len(results) == 0 {
		results = append(results, checkResult{
			Name: "dependencies", OK: true,
			Detail: "none configured; set POSTGRES_HOST, REDIS_HOST or OLLAMA_URL to probe them",
		})
	}
	return results
}

// probeTCP opens a TCP connection to verify a service is actually listening.
//
// The address comes from the operator's environment (POSTGRES_HOST and
// friends), so it is argv-level trust, not request input — but it is still
// validated before the dial: a malformed value should be reported as the
// misconfiguration it is, rather than as an opaque dial error.
func probeTCP(name, address string, timeout time.Duration, optional bool) checkResult {
	if err := validateProbeTarget(address); err != nil {
		return checkResult{
			Name:    name,
			Warning: optional,
			Detail:  fmt.Sprintf("%s is not a usable address: %v", address, err),
		}
	}

	conn, err := net.DialTimeout("tcp", address, timeout) // #nosec G704 -- operator-configured dependency address, validated above
	if err != nil {
		return checkResult{
			Name:    name,
			Warning: optional,
			Detail:  fmt.Sprintf("%s unreachable: %v", address, err),
		}
	}
	_ = conn.Close()
	return checkResult{Name: name, OK: true, Detail: address + " reachable"}
}

// probeResources measures real disk and memory availability.
func probeResources(opts healthOptions) []checkResult {
	var results []checkResult

	wd, err := os.Getwd()
	if err != nil {
		wd = "."
	}

	var stat syscall.Statfs_t
	if err := syscall.Statfs(wd, &stat); err != nil {
		results = append(results, checkResult{Name: "Disk space", Warning: true, Detail: "unavailable: " + err.Error()})
	} else {
		// Bsize is a signed int64. Converting a negative or absurd value
		// straight to uint64 wraps to an enormous number, and the multiply
		// then overflows — so a health check meant to refuse a full disk
		// would report terabytes free and pass.
		free, ok := availableBytes(stat.Bavail, stat.Bsize)
		if !ok {
			results = append(results, checkResult{
				Name: "Disk space", Warning: true,
				Detail: fmt.Sprintf("unavailable: implausible filesystem geometry (bavail=%d bsize=%d)", stat.Bavail, stat.Bsize),
			})
			results = append(results, probeMemory())
			return results
		}
		detail := fmt.Sprintf("%s free at %s", humanBytes(free), wd)
		if free < opts.MinFreeDiskBytes {
			results = append(results, checkResult{Name: "Disk space", Detail: detail + " (below the minimum)"})
		} else {
			results = append(results, checkResult{Name: "Disk space", OK: true, Detail: detail})
		}
	}

	results = append(results, probeMemory())
	return results
}

// probeMemory reports available memory from the kernel where it is exposed.
func probeMemory() checkResult {
	data, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return checkResult{Name: "Memory", Warning: true, Detail: "unavailable on this platform"}
	}

	for _, line := range strings.Split(string(data), "\n") {
		if !strings.HasPrefix(line, "MemAvailable:") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 2 {
			break
		}
		var kb uint64
		if _, err := fmt.Sscanf(fields[1], "%d", &kb); err != nil {
			break
		}
		available := kb * 1024
		detail := humanBytes(available) + " available"
		if available < 64<<20 {
			return checkResult{Name: "Memory", Detail: detail + " (low)"}
		}
		return checkResult{Name: "Memory", OK: true, Detail: detail}
	}
	return checkResult{Name: "Memory", Warning: true, Detail: "MemAvailable not reported"}
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func humanBytes(n uint64) string {
	const unit = 1024
	if n < unit {
		return fmt.Sprintf("%d B", n)
	}
	div, exp := uint64(unit), 0
	for v := n / unit; v >= unit && exp < 4; v /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %ciB", float64(n)/float64(div), "KMGTP"[exp])
}

// validateProbeTarget reports whether address is a dialable "host:port".
func validateProbeTarget(address string) error {
	host, port, err := net.SplitHostPort(address)
	if err != nil {
		return fmt.Errorf("expected host:port: %w", err)
	}
	if host == "" {
		return errors.New("empty host")
	}
	n, err := strconv.Atoi(port)
	if err != nil {
		return fmt.Errorf("port %q is not a number", port)
	}
	if n < 1 || n > 65535 {
		return fmt.Errorf("port %d is out of range", n)
	}
	return nil
}

// availableBytes converts a statfs block count and block size to a byte count,
// reporting false rather than a wrapped or overflowed result.
func availableBytes(bavail uint64, bsize int64) (uint64, bool) {
	if bsize <= 0 {
		return 0, false
	}
	size := uint64(bsize)
	if bavail != 0 && size > math.MaxUint64/bavail {
		return 0, false
	}
	return bavail * size, true
}
