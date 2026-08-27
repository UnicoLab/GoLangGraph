// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package llm

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"time"
)

// Sentinel errors classifying provider failures, so callers can branch with
// errors.Is instead of matching on message text.
var (
	// ErrProviderUnavailable indicates a transient failure: a network error or
	// a 5xx response. Retrying may succeed.
	ErrProviderUnavailable = errors.New("provider unavailable")
	// ErrRateLimited indicates the provider asked the caller to slow down.
	ErrRateLimited = errors.New("provider rate limited")
	// ErrProviderRequest indicates a permanent client error (4xx). Retrying
	// the same request will not help.
	ErrProviderRequest = errors.New("invalid provider request")
	// ErrProviderAuth indicates rejected credentials.
	ErrProviderAuth = errors.New("provider authentication failed")
	// ErrResponseTooLarge indicates the provider returned more data than the
	// configured limit allows.
	ErrResponseTooLarge = errors.New("provider response too large")
)

// MaxResponseBytes bounds how much of a provider response is buffered, so a
// broken or hostile endpoint cannot exhaust memory.
const MaxResponseBytes int64 = 32 << 20 // 32 MiB

// ProviderError carries the provider's HTTP status and body alongside a
// classification of whether the request is worth retrying.
type ProviderError struct {
	Provider   string
	StatusCode int
	Body       string
	// RetryAfter is set when the provider supplied a Retry-After header.
	RetryAfter time.Duration
	kind       error
}

func (e *ProviderError) Error() string {
	if e.StatusCode > 0 {
		return fmt.Sprintf("%s: %s API error: status %d: %s", e.kind, e.Provider, e.StatusCode, e.Body)
	}
	return fmt.Sprintf("%s: %s: %s", e.kind, e.Provider, e.Body)
}

// Unwrap exposes the classification so errors.Is matches the sentinels.
func (e *ProviderError) Unwrap() error { return e.kind }

// Retryable reports whether another attempt could succeed.
func (e *ProviderError) Retryable() bool {
	return errors.Is(e.kind, ErrProviderUnavailable) || errors.Is(e.kind, ErrRateLimited)
}

// classifyStatus maps an HTTP status onto a sentinel.
func classifyStatus(status int) error {
	switch {
	case status == http.StatusTooManyRequests:
		return ErrRateLimited
	case status == http.StatusUnauthorized, status == http.StatusForbidden:
		return ErrProviderAuth
	case status >= 500:
		return ErrProviderUnavailable
	case status >= 400:
		return ErrProviderRequest
	}
	return nil
}

// NewProviderError builds a classified error from an HTTP response. The body is
// read up to a bounded size so an oversized error page cannot exhaust memory.
func NewProviderError(provider string, resp *http.Response) *ProviderError {
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 8<<10))
	pe := &ProviderError{
		Provider:   provider,
		StatusCode: resp.StatusCode,
		Body:       string(body),
		kind:       classifyStatus(resp.StatusCode),
	}
	if pe.kind == nil {
		pe.kind = ErrProviderUnavailable
	}
	if after := resp.Header.Get("Retry-After"); after != "" {
		if seconds, err := strconv.Atoi(after); err == nil && seconds >= 0 {
			pe.RetryAfter = time.Duration(seconds) * time.Second
		}
	}
	return pe
}

// NewTransportError wraps a network-level failure as a retryable provider error.
func NewTransportError(provider string, err error) *ProviderError {
	return &ProviderError{
		Provider: provider,
		Body:     err.Error(),
		kind:     ErrProviderUnavailable,
	}
}

// IsRetryable reports whether an error is worth another attempt. Context
// cancellation is never retryable.
func IsRetryable(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return false
	}
	var pe *ProviderError
	if errors.As(err, &pe) {
		return pe.Retryable()
	}
	return false
}

// retryAfter returns a provider-requested delay, if any.
func retryAfter(err error) time.Duration {
	var pe *ProviderError
	if errors.As(err, &pe) {
		return pe.RetryAfter
	}
	return 0
}

// WithRetry runs fn, retrying transient failures with exponential backoff.
//
// RetryCount and RetryDelay were previously accepted as configuration and then
// ignored, so a transient provider blip failed the whole call. Attempts stop
// early on a permanent error or a cancelled context, and a provider-supplied
// Retry-After is honored.
func WithRetry(ctx context.Context, config *ProviderConfig, fn func(ctx context.Context) error) error {
	attempts := 0
	delay := time.Second
	if config != nil {
		attempts = config.RetryCount
		if config.RetryDelay > 0 {
			delay = config.RetryDelay
		}
	}
	if attempts < 0 {
		attempts = 0
	}

	var err error
	for attempt := 0; ; attempt++ {
		if ctxErr := ctx.Err(); ctxErr != nil {
			if err != nil {
				return err
			}
			return ctxErr
		}

		err = fn(ctx)
		if err == nil {
			return nil
		}
		if attempt >= attempts || !IsRetryable(err) {
			return err
		}

		wait := delay
		if requested := retryAfter(err); requested > wait {
			wait = requested
		}

		select {
		case <-ctx.Done():
			return err
		case <-time.After(wait):
		}

		// Exponential backoff, capped so a long retry budget cannot stall a
		// request for an unbounded time.
		delay *= 2
		if delay > 30*time.Second {
			delay = 30 * time.Second
		}
	}
}

// limitedBody wraps a response body with a size cap, so a provider that streams
// endlessly cannot exhaust memory.
func limitedBody(body io.ReadCloser) io.Reader {
	return io.LimitReader(body, MaxResponseBytes)
}
