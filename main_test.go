// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package main

import (
	"bytes"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestRootEntrypointDirectsOperatorsToTheSupportedCLI(t *testing.T) {
	var output bytes.Buffer

	assert.Equal(t, 2, runRootEntrypoint(&output))
	assert.Equal(t, "GoLangGraph is run with: go run ./cmd/golanggraph [command]\n", output.String())
}
