// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package agentspec_test

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/UnicoLab/GoLangGraph/pkg/agentspec"
)

func FuzzManifest(f *testing.F) {
	definition, err := agentspec.NewHandler[constrained, constrained](agentspec.HandlerConfig{Name: "echo", Description: "Echo"}, func(_ context.Context, _ agentspec.Execution, i constrained) (constrained, error) { return i, nil })
	if err != nil {
		f.Fatal(err)
	}
	pkg, err := agentspec.NewPackage("example.com/echo", "v0.1.0", "Echo", definition)
	if err != nil {
		f.Fatal(err)
	}
	data, err := json.Marshal(pkg.Manifest())
	if err != nil {
		f.Fatal(err)
	}
	f.Add(data)
	f.Add([]byte(`{"module":"a","module":"b"}`))
	f.Add([]byte(`{"agents":[{"input_schema":{"items":{}}}]}`))
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) > 1<<20 {
			t.Skip()
		}
		manifest, err := agentspec.ParseManifest(data)
		if err != nil {
			return
		}
		encoded, err := json.Marshal(manifest)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := agentspec.ParseManifest(encoded); err != nil {
			t.Fatalf("accepted manifest cannot round-trip: %v", err)
		}
	})
}
