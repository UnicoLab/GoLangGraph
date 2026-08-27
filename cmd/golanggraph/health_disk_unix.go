//go:build !windows

// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package main

import (
	"errors"
	"syscall"
)

var errInvalidFilesystemCapacity = errors.New("invalid filesystem capacity")

// diskFreeBytes returns available filesystem capacity for Unix-like targets.
// Keeping the syscall behind a build constraint lets the release workflow
// produce Windows binaries as well as Unix binaries.
func diskFreeBytes(path string) (uint64, error) {
	var stat syscall.Statfs_t
	if err := syscall.Statfs(path, &stat); err != nil {
		return 0, err
	}
	free, ok := availableDiskBytes(stat.Bavail, stat.Bsize)
	if !ok {
		return 0, errInvalidFilesystemCapacity
	}
	return free, nil
}
