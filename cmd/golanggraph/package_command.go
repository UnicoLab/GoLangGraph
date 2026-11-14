// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package main

import (
	"bytes"
	"embed"
	"encoding/json"
	"fmt"
	"go/format"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"text/template"

	"github.com/UnicoLab/GoLangGraph/pkg/agentspec"
	"github.com/spf13/cobra"
)

//go:embed package_templates
var packageTemplates embed.FS

func newPackageCommand() *cobra.Command {
	command := &cobra.Command{Use: "package", Short: "Create and inspect Go-native agent packages (experimental)"}
	var directory, sdkDir, sdkVersion string
	init := &cobra.Command{
		Use: "init <module-path>", Short: "Create a standalone agent module in a new directory", Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			if err := scaffoldPackage(args[0], directory, sdkDir, sdkVersion); err != nil {
				return err
			}
			cmd.Printf("Created agent module %s in %s\nRun go test ./... and go run ./cmd/manifest from that directory.\n", args[0], directory)
			return nil
		},
	}
	init.Flags().StringVar(&directory, "dir", "", "New directory to create (required)")
	init.Flags().StringVar(&sdkDir, "sdk-dir", "", "Local GoLangGraph checkout containing the agent SDK")
	init.Flags().StringVar(&sdkVersion, "sdk-version", "", "Released GoLangGraph SDK version; mutually exclusive with --sdk-dir")
	inspect := &cobra.Command{
		Use: "inspect <manifest.json>", Short: "Validate a package manifest without executing its code", Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			file, err := os.Open(args[0])
			if err != nil {
				return err
			}
			defer file.Close()
			const maximum = 4 << 20
			data, err := io.ReadAll(io.LimitReader(file, maximum+1))
			if err != nil {
				return err
			}
			if len(data) > maximum {
				return fmt.Errorf("manifest exceeds 4 MiB")
			}
			manifest, err := agentspec.ParseManifest(data)
			if err != nil {
				return err
			}
			encoder := json.NewEncoder(cmd.OutOrStdout())
			encoder.SetIndent("", "  ")
			return encoder.Encode(manifest)
		},
	}
	command.AddCommand(init, inspect)
	return command
}

func scaffoldPackage(module, directory, sdkDir, sdkVersion string) error {
	if err := agentspec.ValidateIdentity(module, "v0.1.0"); err != nil {
		return err
	}
	if directory == "" || (sdkDir == "") == (sdkVersion == "") {
		return fmt.Errorf("provide --dir and exactly one of --sdk-dir or --sdk-version")
	}
	if sdkDir != "" {
		absolute, err := filepath.Abs(sdkDir)
		if err != nil {
			return err
		}
		sdkDir = absolute
		if _, err := os.Stat(filepath.Join(sdkDir, "pkg", "agentspec", "spec.go")); err != nil {
			return fmt.Errorf("SDK directory must contain pkg/agentspec/spec.go: %w", err)
		}
		sdkVersion = "v0.0.0"
	} else if err := agentspec.ValidateIdentity("github.com/UnicoLab/GoLangGraph", sdkVersion); err != nil {
		return err
	}
	data := struct{ Module, ModuleQuote, SDKDirQuote, SDKVersion string }{module, strconv.Quote(module), strconv.Quote(filepath.ToSlash(sdkDir)), sdkVersion}
	// Render and format everything before creating the destination. Mkdir is
	// exclusive, so even a concurrent creator cannot have its files overwritten.
	files := make(map[string][]byte)
	err := fs.WalkDir(packageTemplates, "package_templates", func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if entry.IsDir() {
			return nil
		}
		source, err := packageTemplates.ReadFile(path)
		if err != nil {
			return err
		}
		tmpl, err := template.New(path).Parse(string(source))
		if err != nil {
			return err
		}
		var buffer bytes.Buffer
		if renderErr := tmpl.Execute(&buffer, data); renderErr != nil {
			return renderErr
		}
		name := strings.TrimSuffix(strings.TrimPrefix(path, "package_templates/"), ".tmpl")
		content := buffer.Bytes()
		if strings.HasSuffix(name, ".go") {
			content, err = format.Source(content)
			if err != nil {
				return err
			}
		}
		files[name] = content
		return nil
	})
	if err != nil {
		return err
	}
	if err := os.Mkdir(directory, 0750); err != nil {
		return fmt.Errorf("create new package directory: %w", err)
	}
	complete := false
	defer func() {
		if !complete {
			_ = os.RemoveAll(directory)
		}
	}()
	for name, content := range files {
		path := filepath.Join(directory, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0750); err != nil {
			return err
		}
		if err := os.WriteFile(path, content, 0600); err != nil {
			return err
		}
	}
	complete = true
	return nil
}
