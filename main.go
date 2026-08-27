// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package main

import (
	"fmt"
	"io"
	"os"
)

// This package used to start an unrelated mock "Habitat 2035" HTTP server.
// That made `go run .` appear to run GoLangGraph while serving placeholder
// responses and permissive CORS instead of the supported API. The production
// CLI entry point is deliberately explicit until the command package is
// extracted into a reusable library.
func main() {
	os.Exit(runRootEntrypoint(os.Stderr))
}

func runRootEntrypoint(out io.Writer) int {
	_, _ = fmt.Fprintln(out, "GoLangGraph is run with: go run ./cmd/golanggraph [command]")
	return 2
}
