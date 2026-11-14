// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/UnicoLab/GoLangGraph/pkg/agentspec"
)

func TestScaffoldBuildsStandaloneModuleAndInspectsManifest(t *testing.T) {
	sdk, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	dir := filepath.Join(t.TempDir(), "module with spaces")
	command := newPackageCommand()
	var stdout bytes.Buffer
	command.SetOut(&stdout)
	command.SetArgs([]string{"init", "example.com/agent", "--dir", dir, "--sdk-dir", sdk})
	if err := command.Execute(); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(stdout.String(), "Created agent module") {
		t.Fatal(stdout.String())
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	run := func(args ...string) []byte {
		t.Helper()
		cmd := exec.CommandContext(ctx, "go", args...)
		cmd.Dir = dir
		cmd.Env = append(os.Environ(), "GOWORK=off", "GOFLAGS=-mod=readonly")
		data, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("generated module %v: %v\n%s", args, err, data)
		}
		return data
	}
	run("test", "./...")
	manifest := run("run", "./cmd/manifest")
	path := filepath.Join(dir, "manifest.json")
	if err := os.WriteFile(path, manifest, 0600); err != nil {
		t.Fatal(err)
	}
	inspect := newPackageCommand()
	stdout.Reset()
	inspect.SetOut(&stdout)
	inspect.SetArgs([]string{"inspect", path})
	if err := inspect.Execute(); err != nil {
		t.Fatal(err)
	}
	parsed, err := agentspec.ParseManifest(stdout.Bytes())
	if err != nil || parsed.Module != "example.com/agent" || len(parsed.Agents) != 2 {
		t.Fatalf("bad manifest: %+v %v", parsed, err)
	}
	if err := scaffoldPackage("example.com/agent", dir, sdk, ""); err == nil {
		t.Fatal("overwrote existing directory")
	}
	after, err := os.ReadFile(path)
	if err != nil || !bytes.Equal(manifest, after) {
		t.Fatal("existing module changed")
	}
}

func TestScaffoldRejectsInvalidOptionsWithoutCreatingDirectory(t *testing.T) {
	for _, test := range []struct{ module, sdkDir, sdkVersion string }{
		{"example.com/agent", "", ""},
		{"example.com/agent", "missing", "v0.1.0"},
		{"example.com/agent", "missing", ""},
		{"example.com/agent", "", "latest"},
		{"example.com/agent\nreplace injected", "", "v0.1.0"},
		{"example.com/agent\"", "", "v0.1.0"},
	} {
		dir := filepath.Join(t.TempDir(), "agent")
		if err := scaffoldPackage(test.module, dir, test.sdkDir, test.sdkVersion); err == nil {
			t.Fatalf("accepted %+v", test)
		}
		if _, err := os.Stat(dir); !os.IsNotExist(err) {
			t.Fatalf("invalid scaffold wrote files: %v", err)
		}
	}
	dir := filepath.Join(t.TempDir(), "released")
	if err := scaffoldPackage("example.com/agent", dir, "", "v0.2.0"); err != nil {
		t.Fatal(err)
	}
	mod, err := os.ReadFile(filepath.Join(dir, "go.mod"))
	if err != nil || strings.Contains(string(mod), "replace") || !strings.Contains(string(mod), "v0.2.0") {
		t.Fatalf("release scaffold: %s %v", mod, err)
	}
}

func TestInspectRejectsUnknownFieldsAndOversizedFiles(t *testing.T) {
	for _, data := range [][]byte{[]byte(`{"contract_version":"wrong"}`), []byte(strings.Repeat(" ", (4<<20)+1)), []byte(`{"module":"a","module":"b"}`)} {
		path := filepath.Join(t.TempDir(), "bad.json")
		if err := os.WriteFile(path, data, 0600); err != nil {
			t.Fatal(err)
		}
		command := newPackageCommand()
		command.SetArgs([]string{"inspect", path})
		command.SetOut(&bytes.Buffer{})
		command.SetErr(&bytes.Buffer{})
		if err := command.Execute(); err == nil {
			t.Fatal("invalid manifest accepted")
		}
	}
	// Help does not require a provider or execute package code.
	command := newPackageCommand()
	var output bytes.Buffer
	command.SetOut(&output)
	command.SetArgs([]string{"--help"})
	if err := command.Execute(); err != nil {
		t.Fatal(err)
	}
	if json.Valid(output.Bytes()) || !strings.Contains(output.String(), "inspect") {
		t.Fatal("missing package help")
	}
}
