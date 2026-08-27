//go:build windows

// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package main

import "errors"

var errInvalidFilesystemCapacity = errors.New("invalid filesystem capacity")

// diskFreeBytes leaves the local disk probe as a warning on Windows until the
// Windows-specific implementation is available. Server-mode health checks,
// which are what deployed containers use, remain fully supported.
func diskFreeBytes(string) (uint64, error) {
	return 0, errors.New("disk space probe unavailable on Windows")
}
